"""run 폴더(seed*/ 하위 결과) -> 제출 템플릿 형식의 report.md"""
import argparse
import glob
import json
import os
import statistics as st

from mmmu_common import SUBJECTS, CATEGORIES

OFFICIAL = 67.4


def pct(x):
    return f"{x*100:.2f}"


def peak_vram_gb(path):
    try:
        vals = [float(l.strip().split()[0]) for l in open(path) if l.strip() and l.strip()[0].isdigit()]
        return max(vals) / 1024 if vals else None
    except OSError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--method", choices=["official", "ours"], required=True)
    args = ap.parse_args()

    seed_dirs = sorted(glob.glob(os.path.join(args.run_dir, "seed*")))
    runs = []
    for d in seed_dirs:
        sp = os.path.join(d, f"scores_{args.method}_summary.json")
        if os.path.exists(sp):
            runs.append((os.path.basename(d), json.load(open(sp)),
                         json.load(open(os.path.join(d, "predictions_meta.json")))))
    assert runs, f"no summaries for method={args.method} in {args.run_dir}"
    names = [r[0] for r in runs]
    multi = len(runs) > 1

    L = [f"# MMMU-val 결과 — scoring={args.method}, runs={', '.join(names)}\n"]
    hdr = "| No. | Subject | Data Num | " + (" | ".join(names) + " | Acc (mean) | Time (s, mean) |" if multi
                                          else "Acc | Time (s) |")
    L += [hdr, "|" + "---|" * (hdr.count("|") - 1)]
    for i, s in enumerate(SUBJECTS, 1):
        accs = [r[1]["subject_acc"][s] for r in runs]
        t = st.mean(r[2]["timing_sec_by_subject"][s] for r in runs)
        n = runs[0][1]["subject_n"][s]
        cells = (" | ".join(pct(a) for a in accs) + f" | {pct(st.mean(accs))}") if multi else pct(accs[0])
        L.append(f"| {i} | {s} | {n} | {cells} | {t:.1f} |")
    overall = [r[1]["overall_macro"] for r in runs]
    tot_n = sum(runs[0][1]["subject_n"].values())
    if multi:
        L.append(f"| | **Overall (macro avg)** | {tot_n} | " + " | ".join(pct(o) for o in overall)
                 + f" | **{pct(st.mean(overall))} ± {st.stdev(overall)*100:.2f}** | "
                 + f"{st.mean(r[2]['total_sec'] for r in runs):.0f} |")
    else:
        L.append(f"| | **Overall (macro avg)** | {tot_n} | **{pct(overall[0])}** | {runs[0][2]['total_sec']:.0f} |")
    L.append("\n계산식: Overall = mean(30개 과목 accuracy). 과목당 30문제로 균등하므로 900문제 micro 정답률과 같아야 함 "
             "(검산: " + ", ".join(f"{n}: micro={pct(r['overall_micro'])}, {r['num_correct']}/{r['num_questions']}"
                                 for n, r, _ in runs) + ")")
    if multi:
        L.append("± 는 seed 간 표본 표준편차.")

    m = st.mean(overall) * 100
    L += ["\n## 공식 수치와 비교\n", "| | Overall (MMMU val) |", "|---|---|",
          f"| 공식 (Qwen3-VL Technical Report) | {OFFICIAL} |", f"| 우리 재현 결과 | {m:.2f} |",
          f"| 차이 (Δ) | {m - OFFICIAL:+.2f} |"]

    L += ["\n## 분야별 정확도\n", "| Category | #Subj | Acc |", "|---|---|---|"]
    for c, subs in CATEGORIES.items():
        L.append(f"| {c} | {len(subs)} | {pct(st.mean(r[1]['category_acc'][c] for r in runs))} |")

    L += ["\n## 진단 정보 (격차 분석용)\n", "| run | 추출 방식 분포 | finish_reason | 잘린 응답 | 평균/최대 출력 토큰 | 문제유형별 acc | peak VRAM (nvidia-smi) |",
          "|---|---|---|---|---|---|---|"]
    for n, s, meta in runs:
        qt = ", ".join(f"{k}: {pct(v['acc'])} (n={v['n']})" for k, v in s["by_question_type"].items())
        v = peak_vram_gb(os.path.join(args.run_dir, f"vram_{n}.log"))
        L.append(f"| {n} | {s['extraction_methods']} | {s['finish_reason']} | {len(s['truncated_ids'])} (그중 정답 {s.get('truncated_hits', '-')}) | "
                 f"{s['avg_output_tokens']:.0f} / {s['max_output_tokens']} | {qt} | "
                 f"{f'{v:.1f} GB' if v else '-'} |")

    a = runs[0][2]["args"]
    L += ["\n## 사용한 설정 (predictions_meta.json 에서 자동 기록)\n", "```json",
          json.dumps({k: a[k] for k in a if k not in ("output_file",)}, indent=2, ensure_ascii=False), "```"]
    if runs[0][1]["scoring"].get("judge_model"):
        L.append(f"\n판정 모델: `{runs[0][1]['scoring']['judge_model']}`")

    out = os.path.join(args.run_dir, f"report_{args.method}.md")
    open(out, "w").write("\n".join(L) + "\n")
    print(f"[report] -> {out}  overall={m:.2f}")


if __name__ == "__main__":
    main()
