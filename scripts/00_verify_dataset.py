#!/usr/bin/env python3
"""Phase 0 -- dataset integrity. Run this before anything else.

Writes results/dataset_report.md. Every later number depends on this passing.
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from bavlm.data.splits import check_overlap, label_vocabulary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/finebadminton-20k")
    ap.add_argument("--bench", default="data/fbbench")
    ap.add_argument("--out", default="results/dataset_report.md")
    a = ap.parse_args()

    for d in (a.train, a.bench):
        if not pathlib.Path(d).exists():
            print(f"missing: {d}\n  hf download ... --repo-type dataset --local-dir {d}")
            return 1

    rep = check_overlap(a.train, a.bench)
    vocab = label_vocabulary(a.train)

    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as fh:
        fh.write("# Dataset report\n\n")
        fh.write(f"- train clips: {rep['train_clips']}\n")
        fh.write(f"- benchmark clips: {rep['bench_clips']}\n")
        fh.write(f"- overlap: {rep['overlap_count']}\n\n")
        if not rep["clean"]:
            fh.write("## CONTAMINATION\n\nClips appear in both sets. Exclude them "
                     "from training or the evaluation is invalid.\n\n")
            fh.write("```\n" + "\n".join(rep["overlap_sample"]) + "\n```\n\n")
        fh.write("## Label vocabulary\n\n")
        for field, counts in vocab.items():
            fh.write(f"### {field} ({len(counts)} values)\n\n")
            for k, v in counts.most_common():
                fh.write(f"- `{k}` — {v}\n")
            fh.write("\n")

    print(f"wrote {a.out}")
    if not rep["clean"]:
        print(f"!! {rep['overlap_count']} overlapping clips — resolve before training")
        return 2
    print("no overlap detected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
