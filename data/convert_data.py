import json
import re
from tqdm import tqdm


# ========= 基础 =========

def split_think(text):
    think = re.findall(r"<think>(.*?)</think>", text, re.S)
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    return clean, "\n".join(think).strip()


# ========= 核心转换 =========

def convert(messages, keep_think=False, max_history=3):
    system_prompt = ""
    samples = []

    # system
    if messages[0]["role"] == "system":
        system_prompt = messages[0]["content"]
        messages = messages[1:]

    history = []

    for i in range(len(messages)):
        msg = messages[i]

        # 我们只学 assistant 的行为
        if msg["role"] != "assistant":
            continue

        # 必须有上一轮 user
        if i == 0 or messages[i - 1]["role"] != "user":
            continue

        user_msg = messages[i - 1]["content"]
        assistant_msg = msg["content"]

        # 处理 think
        clean_output, think = split_think(assistant_msg)
        instruction, _ = split_think(user_msg)

        sample = {
            "instruction": instruction,
            "input": "",
            "output": clean_output if not keep_think else assistant_msg.strip(),
            "system": system_prompt,
            "history": history[-max_history:]
        }

        if keep_think and think:
            sample["think"] = think

        samples.append(sample)

        # 更新 history（统一用 clean）
        history.append([instruction, clean_output])

    return samples


# ========= 主流程 =========

def process(input_file, out_no_think, out_with_think):
    data_no_think = []
    data_with_think = []

    with open(input_file, "r") as f:
        for line in tqdm(f):
            data = json.loads(line)
            messages = data["messages"]

            data_no_think.extend(convert(messages, keep_think=False))
            data_with_think.extend(convert(messages, keep_think=True))

    # 保存
    with open(out_no_think, "w") as f:
        json.dump(data_no_think, f, ensure_ascii=False, indent=2)

    with open(out_with_think, "w") as f:
        json.dump(data_with_think, f, ensure_ascii=False, indent=2)

    print(f"✅ 无think: {len(data_no_think)}")
    print(f"✅ 含think: {len(data_with_think)}")


# ========= 运行 =========

if __name__ == "__main__":
    # process(
    #     input_file="/root/autodl-tmp/training/data/sft_1k_think_content/train_800.jsonl",
    #     out_no_think="training_mix_no_think.json",
    #     out_with_think="training_mix_with_think.json"
    # )
    # process(
    #     input_file="/root/autodl-tmp/training/data/sft_1k_think_content/test_100.jsonl",
    #     out_no_think="test_mix_no_think.json",
    #     out_with_think="test_mix_with_think.json"
    # )
    process(
        input_file="/root/autodl-tmp/training/data/sft_1k_think_content/val_100.jsonl",
        out_no_think="val_mix_no_think.json",
        out_with_think="val_mix_with_think.json"
    )