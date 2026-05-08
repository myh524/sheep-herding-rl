#!/usr/bin/env python3
"""
从 evaluate_generalization.py 写出的 JSON 重新生成折线图（无需 GPU / 不重跑环境）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from plot_generalization import save_generalization_line_figures  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="从泛化评估 JSON 绘制折线图")
    p.add_argument("json_path", type=str, help="evaluate_generalization 输出的 .json")
    p.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="PNG 输出目录",
    )
    p.add_argument(
        "--figure_prefix",
        type=str,
        default="",
        help="与 evaluate_generalization.py --figure_prefix 相同",
    )
    args = p.parse_args()

    with open(args.json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sheep_list = [int(x) for x in data["sheep_counts"]]
    herder_list = [int(x) for x in data["herder_counts"]]
    cells = data["cells"]
    title = os.path.basename(data.get("model_path", args.json_path))

    paths = save_generalization_line_figures(
        args.output_dir,
        sheep_list,
        herder_list,
        cells,
        figure_prefix=args.figure_prefix,
        suptitle=title,
    )
    for fp in paths:
        print(fp)


if __name__ == "__main__":
    main()
