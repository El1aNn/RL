"""
AdapterManager：统一管理 buyer / seller 两个 LoRA adapter 的生命周期

职责：
- 在同一个 base 模型上创建 / 加载 buyer 和 seller adapter
- 按需激活 / 禁用 adapter（disable 时退化为纯 base = SFT-merged base = KL ref）
- 保存 / 持久化 adapter

典型用法：
    mgr = AdapterManager(base_model_path="./sft_base")
    mgr.ensure_adapter("buyer", lora_config)
    mgr.ensure_adapter("seller", lora_config)

    with mgr.use_adapter("buyer"):
        logp_policy = mgr.model(...)

    with mgr.use_no_adapter():
        logp_ref = mgr.model(...)

    mgr.save_adapter("buyer", path="./ckpt/buyer_step100")
"""
from contextlib import contextmanager
from pathlib import Path
import shutil
from typing import Dict, Optional, Any

from peft import LoraConfig, PeftModel, get_peft_model


class AdapterManager:
    """
    LoRA 多 adapter 管理器。

    假设 base_model 已经是 SFT merge 后的 HF CausalLM，挂两个 LoRA adapter。
    """

    def __init__(
        self,
        model,
        adapters_init: Optional[Dict[str, str]] = None,
    ):
        """
        Args:
            model: 已加载的 HF CausalLM（bf16）
            adapters_init: {"buyer": "/path", "seller": "/path"} 从已有 adapter 目录加载；
                           None 则延后通过 ensure_adapter() 新建
        """
        self.model = model
        self._adapter_configs: Dict[str, LoraConfig] = {}
        self._peft_model: Optional[PeftModel] = None

        if adapters_init:
            # 从已有路径加载每个 adapter
            for name, path in adapters_init.items():
                self.load_adapter(name, path)

    # ------------------------------------------------------------
    # 创建 / 加载
    # ------------------------------------------------------------

    def ensure_adapter(self, name: str, lora_config: LoraConfig) -> None:
        """如果 adapter 不存在，用 lora_config 新建一个（参数随机初始化）"""
        if self._peft_model is None:
            # 首次：用 get_peft_model 包装
            self._peft_model = get_peft_model(self.model, lora_config, adapter_name=name)
            self.model = self._peft_model
        else:
            if name in self._peft_model.peft_config:
                return
            self._peft_model.add_adapter(name, lora_config)
        self._adapter_configs[name] = lora_config

    def load_adapter(self, name: str, path: str) -> None:
        """从磁盘加载已有 adapter 到当前模型"""
        adapter_path = Path(path)
        if not (adapter_path / "adapter_config.json").exists():
            nested_path = adapter_path / name
            if (nested_path / "adapter_config.json").exists():
                adapter_path = nested_path
        path = str(adapter_path)
        if self._peft_model is None:
            # 首次加载：from_pretrained 包装
            self._peft_model = PeftModel.from_pretrained(
                self.model, path, adapter_name=name, is_trainable=True,
            )
            self.model = self._peft_model
        else:
            self._peft_model.load_adapter(path, adapter_name=name, is_trainable=True)

    # ------------------------------------------------------------
    # 激活 / 禁用
    # ------------------------------------------------------------

    @contextmanager
    def use_adapter(self, name: str):
        """临时激活指定 adapter"""
        assert self._peft_model is not None, "no adapters loaded"
        prev_active = self._peft_model.active_adapter
        prev_disabled = getattr(self._peft_model, "disable_adapter_layers", None) is not None \
                        and hasattr(self._peft_model, "_disable_adapters") \
                        and self._peft_model._disable_adapters
        self._peft_model.set_adapter(name)
        # 确保 adapter 层启用
        if hasattr(self._peft_model, "enable_adapter_layers"):
            self._peft_model.enable_adapter_layers()
        try:
            yield self._peft_model
        finally:
            self._peft_model.set_adapter(prev_active)
            if prev_disabled and hasattr(self._peft_model, "disable_adapter_layers"):
                self._peft_model.disable_adapter_layers()

    @contextmanager
    def use_no_adapter(self):
        """临时禁用所有 adapter，退化为纯 base（= SFT-merged base = KL reference）"""
        assert self._peft_model is not None, "no adapters loaded"
        self._peft_model.disable_adapter_layers()
        try:
            yield self._peft_model
        finally:
            self._peft_model.enable_adapter_layers()

    def set_trainable(self, name: str, trainable: bool = True) -> None:
        """把某个 adapter 设为可训练（其它 frozen）"""
        assert self._peft_model is not None
        self._peft_model.set_adapter(name)
        for n, p in self._peft_model.named_parameters():
            if name in n and "lora_" in n:
                p.requires_grad = trainable

    def freeze_all(self) -> None:
        """把所有 adapter 设为 frozen"""
        assert self._peft_model is not None
        for _, p in self._peft_model.named_parameters():
            p.requires_grad = False

    # ------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------

    def save_adapter(self, name: str, path: str) -> None:
        """保存指定 adapter 到 path"""
        assert self._peft_model is not None
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        # PeftModel 的标准保存方式
        self._peft_model.set_adapter(name)
        self._peft_model.save_pretrained(str(target), selected_adapters=[name])

        # PEFT 在多 adapter 场景下会保存到 <path>/<adapter_name>/。
        # vLLM 和训练脚本都按 <path>/adapter_config.json 读取，所以这里铺平目录。
        nested = target / name
        if (nested / "adapter_config.json").exists():
            for child in nested.iterdir():
                dest = target / child.name
                if dest.exists():
                    if dest.is_dir():
                        shutil.rmtree(dest)
                    else:
                        dest.unlink()
                shutil.move(str(child), str(dest))
            nested.rmdir()

    # ------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------

    @property
    def peft_model(self):
        return self._peft_model

    @property
    def active_adapter(self) -> Optional[str]:
        if self._peft_model is None:
            return None
        return self._peft_model.active_adapter
