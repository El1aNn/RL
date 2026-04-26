"""
vLLM 多 LoRA 推理封装

功能：
- colocate 模式（训练 + 推理共享同进程，共享 base model 权重，CPU RAM 友好）
- 通过 LoRARequest 切换 buyer / seller / ref
- 支持 adapter 热重载（训练后把新 adapter 写盘 → 下次 generate 自动拿到）

注意：
- 本类封装假设单卡。多卡需扩展。
- 生产环境启动 vLLM 需要传 enable_lora=True, max_lora_rank, max_loras。
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any
import os
import shutil


# 延迟 import，允许在无 vllm 环境下 import 本模块做静态测试
def _lazy_import_vllm():
    from vllm import LLM, SamplingParams  # noqa
    from vllm.lora.request import LoRARequest  # noqa
    return LLM, SamplingParams, LoRARequest


@dataclass
class GenerateOutput:
    """vLLM 推理结果的轻量封装"""
    prompt: str
    text: str                      # 生成的文本（不含 prompt）
    prompt_token_ids: List[int]
    completion_token_ids: List[int]
    finish_reason: Optional[str] = None


class VLLMClient:
    """
    vLLM + multi-LoRA 客户端（单卡 colocate 模式）。

    用法：
        client = VLLMClient(
            base_model="./checkpoints/sft_base",
            adapters={"buyer": "./ckpt/buyer", "seller": "./ckpt/seller"},
            max_lora_rank=64,
            gpu_memory_utilization=0.55,
        )
        outputs = client.generate(
            prompts=["prompt 1", "prompt 2"],
            adapter_name="buyer",
            temperature=0.9,
            max_new_tokens=128,
        )
    """

    def __init__(
        self,
        base_model: str,
        adapters: Optional[Dict[str, str]] = None,
        max_lora_rank: int = 64,
        max_loras: int = 3,
        gpu_memory_utilization: float = 0.55,
        dtype: str = "bfloat16",
        max_model_len: int = 4096,
        seed: int = 42,
        enforce_eager: bool = False,
    ):
        LLM, SamplingParams, LoRARequest = _lazy_import_vllm()

        self._LLM = LLM
        self._SamplingParams = SamplingParams
        self._LoRARequest = LoRARequest

        self.base_model = base_model
        self.adapters = {
            name: self._resolve_adapter_path(name, path)
            for name, path in dict(adapters or {}).items()
        }
        for path in self.adapters.values():
            self._ensure_lora_tokenizer_files(path)
        self.engine = LLM(
            model=base_model,
            enable_lora=True,
            max_lora_rank=max_lora_rank,
            max_loras=max_loras,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            max_model_len=max_model_len,
            seed=seed,
            enforce_eager=enforce_eager,
        )
        # 每个 adapter name 对应一个「当前有效的 int id」
        # 关键：热重载时必须递增 int id，否则 vLLM 会 cache 旧权重
        self._lora_int_ids: Dict[str, int] = {}
        self._next_lora_int_id: int = 1   # 单调递增的 id 分配器
        for name in self.adapters:
            self._lora_int_ids[name] = self._next_lora_int_id
            self._next_lora_int_id += 1

    # ------------------------------------------------------------
    # Adapter 管理
    # ------------------------------------------------------------

    def register_adapter(self, name: str, path: str) -> None:
        """注册一个新 adapter（如果 name 已存在会递增 id + 覆盖路径，下次 generate 生效）"""
        path = self._resolve_adapter_path(name, path)
        self._ensure_lora_tokenizer_files(path)
        self.adapters[name] = path
        # 不管是否已存在，都分配新 id，确保 vLLM 不 cache 旧版本
        self._lora_int_ids[name] = self._next_lora_int_id
        self._next_lora_int_id += 1

    def reload_adapter(self, name: str, new_path: str) -> None:
        """
        训练完成后热更新 adapter 路径。

        关键细节：必须分配一个**新的** lora_int_id，否则 vLLM 会认为是同一个 LoRA
        而直接复用 KV cache / 权重缓存，拿不到最新参数。
        """
        if not os.path.exists(new_path):
            raise FileNotFoundError(f"adapter path not found: {new_path}")
        new_path = self._resolve_adapter_path(name, new_path)
        self._ensure_lora_tokenizer_files(new_path)
        self.adapters[name] = new_path
        self._lora_int_ids[name] = self._next_lora_int_id
        self._next_lora_int_id += 1

    def _resolve_adapter_path(self, name: str, path: str) -> str:
        """兼容旧 checkpoint 中 <path>/<adapter_name>/ 的保存形态。"""
        adapter_path = Path(path)
        if not (adapter_path / "adapter_config.json").exists():
            nested_path = adapter_path / name
            if (nested_path / "adapter_config.json").exists():
                adapter_path = nested_path
        return str(adapter_path)

    def _ensure_lora_tokenizer_files(self, adapter_path: str) -> None:
        """vLLM 0.4 会尝试从 LoRA 目录加载 tokenizer；缺失时补 base tokenizer 文件。"""
        target = Path(adapter_path)
        base = Path(self.base_model)
        if not target.exists() or not base.is_dir():
            return

        for filename in (
            "config.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "tokenizer.json",
            "vocab.json",
            "merges.txt",
        ):
            src = base / filename
            dest = target / filename
            if src.exists() and not dest.exists():
                shutil.copy2(src, dest)

    def _make_lora_request(self, name: Optional[str]):
        if name is None:
            return None
        if name not in self.adapters:
            raise KeyError(f"adapter {name} not registered")
        self._ensure_lora_tokenizer_files(self.adapters[name])
        return self._LoRARequest(
            lora_name=name,
            lora_int_id=self._lora_int_ids[name],
            lora_local_path=self.adapters[name],
        )

    # ------------------------------------------------------------
    # 生成
    # ------------------------------------------------------------

    def generate(
        self,
        prompts: List[str],
        adapter_name: Optional[str] = None,
        temperature: float = 0.9,
        top_p: float = 0.9,
        max_new_tokens: int = 128,
        stop: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ) -> List[GenerateOutput]:
        """
        批量生成。

        adapter_name=None 表示用纯 base（即 SFT-merged base，作为 ref 或 zero-shot）。
        """
        sampling_params = self._SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            stop=stop,
            seed=seed,
        )
        lora_request = self._make_lora_request(adapter_name)

        results = self.engine.generate(
            prompts=prompts,
            sampling_params=sampling_params,
            lora_request=lora_request,
            use_tqdm=False,
        )

        outputs: List[GenerateOutput] = []
        for r in results:
            out = r.outputs[0]
            outputs.append(GenerateOutput(
                prompt=r.prompt,
                text=out.text,
                prompt_token_ids=list(r.prompt_token_ids) if r.prompt_token_ids else [],
                completion_token_ids=list(out.token_ids) if out.token_ids else [],
                finish_reason=out.finish_reason,
            ))
        return outputs

    def generate_with_chat_template(
        self,
        messages_list: List[List[Dict[str, str]]],
        tokenizer,
        adapter_name: Optional[str] = None,
        **kwargs,
    ) -> List[GenerateOutput]:
        """
        便捷入口：传 messages 列表，自动 apply_chat_template 再生成。
        """
        prompts = [
            tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )
            for msgs in messages_list
        ]
        return self.generate(prompts=prompts, adapter_name=adapter_name, **kwargs)
