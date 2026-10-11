#!/usr/bin/env python3
"""Phase 1 -- zero-shot FBBench benchmark under uniform 2 FPS sampling.

This is the floor every later method is measured against, and the feasibility
check: 7B models should land near or below 25% overall. Much higher suggests an
answer-extraction bug or leakage; much lower suggests frames are not loading.
Investigate either before proceeding.
"""
import argparse

# TODO(phase 1): wire bavlm.models.backend + bavlm.eval.fbbench


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2_5_vl_7b")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="results/zeroshot")
    a = ap.parse_args()
    raise NotImplementedError("roadmap phase 1")


if __name__ == "__main__":
    raise SystemExit(main())
