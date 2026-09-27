"""우리 객관식 파서 수동 점검용: 방법별 무작위 샘플 + 공식과 갈린 문제 출력.
사용: python scripts/audit_ours_mc.py results/official/seed42 [샘플 수]"""
import json
import random
import sys
from collections import defaultdict

d = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
load = lambda f: {r["id"]: r for r in map(json.loads, open(f"{d}/{f}"))}
preds, ours, off = load("predictions.jsonl"), load("scores_ours.jsonl"), load("scores_official.jsonl")


def show(i, tag=""):
    p, o = preds[i], ours[i]
    tail = p["response"][-300:].replace("\n", " ⏎ ")
    print(f"--- {i} {tag} | 방법={o['method']} 추출={o['extracted']} 정답={p['answer']} "
          f"공식={off[i]['extracted']}({off[i]['method']}) finish={p['finish_reason']}")
    print(f"    ...{tail}\n")


mc = [i for i, p in preds.items() if p["question_type"] == "multiple-choice"]
by = defaultdict(list)
for i in mc:
    by[ours[i]["method"]].append(i)
rng = random.Random(0)
for m, ids in sorted(by.items()):
    print(f"\n######## {m}: {len(ids)}문제 중 {min(n, len(ids))}개 샘플")
    for i in rng.sample(ids, min(n, len(ids))):
        show(i)

print("\n######## 공식만 맞고 우리는 틀린 객관식 (우리 파서 오류 후보)")
for i in mc:
    if off[i]["hit"] and not ours[i]["hit"]:
        show(i)