#!/usr/bin/env python3
"""Run the baseline Qwen model on MMMU test and save submission answers.

Accepts the same arguments as eval/qwen/mmmu.py. Raw completions are saved
alongside the question-ID-keyed submission, without scoring hidden labels.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.qwen.mmmu import main


if __name__ == "__main__":
    raise SystemExit(main(split="test"))
