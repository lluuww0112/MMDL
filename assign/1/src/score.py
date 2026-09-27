"""predictions.jsonl -> 채점(official|ours) -> scores_<method>.jsonl + summary_<method>.json

같은 predictions 파일을 두 방식으로 모두 채점할 수 있다 (채점 방식만의 효과를 분리해서 볼 때 사용).
"""
import argparse
import json
from collections import Counter

from mmmu_common import SUBJECTS, CATEGORIES


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def summarize(preds, results):
    by_subj = {s: [] for s in SUBJECTS}
    for p, r in zip(preds, results):
        by_subj[p["subject"]].append(r["hit"])
    subj_acc = {s: (sum(h) / len(h) if h else None) for s, h in by_subj.items()}
    subj_n = {s: len(h) for s, h in by_subj.items()}
    valid = [a for a in subj_acc.values() if a is not None]
    cat_acc = {}
    for c, subs in CATEGORIES.items():
        hits = [h for s in subs for h in by_subj[s]]
        cat_acc[c] = sum(hits) / len(hits) if hits else None
    return {
        "overall_macro": sum(valid) / len(valid),                       # 과제 규정: 30과목 단순평균
        "overall_micro": sum(r["hit"] for r in results) / len(results),  # 900문제 중 정답 비율 (검산용)
        "num_correct": sum(r["hit"] for r in results),
        "num_questions": len(results),
        "subject_acc": subj_acc, "subject_n": subj_n, "category_acc": cat_acc,
        "extraction_methods": dict(Counter(r["method"] for r in results)),
        "finish_reason": dict(Counter(p.get("finish_reason") for p in preds)),
        "truncated_ids": [p["id"] for p in preds if p.get("finish_reason") == "length"],
        "truncated_hits": sum(r["hit"] for p, r in zip(preds, results) if p.get("finish_reason") == "length"),
        "avg_output_tokens": sum(p.get("num_output_tokens", 0) for p in preds) / len(preds),
        "max_output_tokens": max(p.get("num_output_tokens", 0) for p in preds),
        "by_question_type": {
            qt: {"n": sum(1 for p in preds if p["question_type"] == qt),
                 "acc": (lambda hs: sum(hs) / len(hs) if hs else None)(
                     [r["hit"] for p, r in zip(preds, results) if p["question_type"] == qt])}
            for qt in sorted({p["question_type"] for p in preds})},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_file", required=True)
    ap.add_argument("--method", choices=["official", "ours"], required=True)
    ap.add_argument("--judge_model", default="gpt-3.5-turbo-0125", help="official 모드 GPT 판정 모델")
    ap.add_argument("--no_judge", action="store_true", help="official 모드에서 판정기 없이 규칙만 (디버그용)")
    ap.add_argument("--nproc", type=int, default=8)
    args = ap.parse_args()

    preds = load_jsonl(args.pred_file)
    if args.method == "official":
        import score_official
        results = score_official.score_file(preds, args.judge_model, use_judge=not args.no_judge,
                                            nproc=args.nproc)
    else:
        import score_ours
        results = score_ours.score_file(preds)

    base = args.pred_file.rsplit(".jsonl", 1)[0].replace("predictions", "scores")
    with open(f"{base}_{args.method}.jsonl", "w") as f:
        for p, r in zip(preds, results):
            f.write(json.dumps({"id": p["id"], "subject": p["subject"], **r}, ensure_ascii=False) + "\n")
    summary = summarize(preds, results)
    summary["scoring"] = {"method": args.method,
                          "judge_model": args.judge_model if args.method == "official" and not args.no_judge else None}
    with open(f"{base}_{args.method}_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[score:{args.method}] overall(macro)={summary['overall_macro']*100:.2f}  "
          f"({summary['num_correct']}/{summary['num_questions']})  methods={summary['extraction_methods']}  "
          f"truncated={len(summary['truncated_ids'])}")


if __name__ == "__main__":
    main()
