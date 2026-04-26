"""
按 role 拆分 SFT 数据集

plan 中的三阶段 GRPO 要求 buyer / seller 分别初始化各自的 adapter，
最干净的做法是先用各自角色的 SFT 数据分别做 role-specific SFT。
因此把 sft_1k / sft_1k_think_content 按 metadata.role 拆成
buyer_*.jsonl 和 seller_*.jsonl。

默认就地写入（在同目录下产出 *_buyer.jsonl / *_seller.jsonl）。

用法：
    # 同时拆分两个数据集
    python data/split_sft_by_role.py

    # 只拆指定目录
    python data/split_sft_by_role.py --dataset-dir data/sft_1k_think_content

    # 自定义输出目录（保持原文件名规则）
    python data/split_sft_by_role.py \
        --dataset-dir data/sft_1k \
        --output-dir  data/sft_1k/by_role
"""
import json
import argparse
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DATASETS = [
    ROOT / "data" / "sft_1k",
    ROOT / "data" / "sft_1k_think_content",
]

SPLIT_FILES = ["train_800.jsonl", "val_100.jsonl", "test_100.jsonl"]
VALID_ROLES = ("buyer", "seller")


def split_one_file(
    input_path: Path,
    output_dir: Path,
    file_stem: str,
) -> dict:
    """
    拆分一个 jsonl 文件。

    输入 <file_stem>.jsonl，输出 <file_stem>_buyer.jsonl 和 <file_stem>_seller.jsonl。

    Returns:
        每个 role 的样本计数
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_paths = {role: output_dir / f"{file_stem}_{role}.jsonl" for role in VALID_ROLES}
    writers = {role: out_paths[role].open("w", encoding="utf-8") for role in VALID_ROLES}

    counts = Counter()
    skipped = 0

    try:
        with input_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"  ⚠ 跳过第 {line_no} 行 JSON 解析失败: {e}")
                    skipped += 1
                    continue

                role = (sample.get("metadata") or {}).get("role")
                if role not in VALID_ROLES:
                    print(f"  ⚠ 跳过第 {line_no} 行：未知 role={role!r}")
                    skipped += 1
                    continue

                writers[role].write(json.dumps(sample, ensure_ascii=False) + "\n")
                counts[role] += 1
    finally:
        for w in writers.values():
            w.close()

    return {
        "counts": dict(counts),
        "skipped": skipped,
        "out_paths": {role: str(p) for role, p in out_paths.items()},
    }


def split_dataset_dir(
    dataset_dir: Path,
    output_dir: Path = None,
):
    """拆分一个数据集目录下的所有 split 文件"""
    if output_dir is None:
        output_dir = dataset_dir

    print(f"\n>>> 处理数据集: {dataset_dir}")
    if not dataset_dir.is_dir():
        print(f"  ✗ 目录不存在，跳过")
        return

    total = Counter()
    for split_file in SPLIT_FILES:
        input_path = dataset_dir / split_file
        if not input_path.exists():
            print(f"  - 跳过 {split_file}（不存在）")
            continue

        stem = Path(split_file).stem  # "train_800"
        result = split_one_file(input_path, output_dir, stem)
        buyer_n = result["counts"].get("buyer", 0)
        seller_n = result["counts"].get("seller", 0)
        print(
            f"  ✓ {split_file}: buyer={buyer_n}, seller={seller_n}"
            + (f", skipped={result['skipped']}" if result["skipped"] else "")
        )
        total["buyer"] += buyer_n
        total["seller"] += seller_n

    print(f"  合计: buyer={total['buyer']}, seller={total['seller']}")


def main():
    parser = argparse.ArgumentParser(description="按 role 拆分 SFT 数据集")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="单个数据集目录。默认同时处理 data/sft_1k 与 data/sft_1k_think_content",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录（默认就地写到 dataset-dir）",
    )
    args = parser.parse_args()

    if args.dataset_dir:
        targets = [Path(args.dataset_dir)]
    else:
        targets = DEFAULT_DATASETS

    output_dir = Path(args.output_dir) if args.output_dir else None

    for ds in targets:
        split_dataset_dir(ds, output_dir=output_dir)


if __name__ == "__main__":
    main()
