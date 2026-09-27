"""GPU 없이 돌아가는 검증: 프롬프트 생성, 두 채점기, 집계 산술.  실행: python -m pytest -q tests"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import mmmu_common as mc  # noqa: E402
import score_official as so  # noqa: E402
import score_ours as su  # noqa: E402

ABCD = {"A": "$6", "B": "$7", "C": "$8", "D": "$9"}


def mcq(resp, gold="B", choices=ABCD):
    return {"idx": 0, "id": "x", "subject": "Accounting", "question_type": "multiple-choice",
            "question": "q", "choices": choices, "answer": gold, "response": resp}


def test_prompt_matches_official_format():
    row = {"id": "validation_Accounting_1", "question": "What is <image 1>?", "options": "['$6', '$7']",
           "answer": "B", "question_type": "multiple-choice", "image_1": "IMG1", "image_2": None,
           **{f"image_{k}": None for k in range(3, 8)}}
    rec = mc.row_to_record(row, "Accounting")
    assert mc.build_prompt_text(rec) == (
        "Question: What is <image 1>?\nOptions:\nA. $6\nB. $7\n"
        "Please select the correct answer from the options above.")
    msgs = mc.build_messages(rec)
    assert [c["type"] for c in msgs[0]["content"]] == ["image", "text"]
    assert msgs[0]["content"][0]["max_pixels"] == 5120 * 28 * 28


def test_open_question_prompt_and_gold():
    row = {"id": "o", "question": "Compute x.", "options": "[]", "answer": "['0.8', '4/5']",
           "question_type": "open", **{f"image_{k}": None for k in range(1, 8)}}
    rec = mc.row_to_record(row, "Math")
    assert mc.build_prompt_text(rec) == "Question: Compute x."
    assert rec["answer"] == ["0.8", "4/5"]


def test_ours_final_answer_beats_later_mentions():
    # MMMU 원본 파서는 마지막 단독 글자(A)를 고르지만, 우리는 선언된 최종답(B)을 고름
    r = "The answer is B. Option A is wrong because the cost is fixed."
    assert su.parse_multi_choice_response(r, list(ABCD), ABCD) == "A"
    assert su.score_one(mcq(r))["extracted"] == "B"


def test_ours_formats():
    for r in ["Answer: B", "**Answer:** (B)", "Final answer: \\boxed{B}", "the correct option is B.",
              "B", "(B) $7", "So the answer is **B**."]:
        assert su.score_one(mcq(r))["extracted"] == "B", r
    # 'a' 소문자/단어의 일부는 선택지로 보지 않음
    assert su.extract_final_answer("The answer is a bit unclear. Approximately", list(ABCD)) is None


def test_ours_no_random_fallback():
    s = su.score_one(mcq("I cannot tell."))
    assert s["extracted"] is None and s["hit"] == 0 and s["method"] == "failed"


def test_ours_open():
    p = {**mcq("Therefore the result is 0.80."), "question_type": "open", "choices": {}, "answer": ["0.8", "4/5"]}
    assert su.score_one(p)["hit"] == 1
    p["response"] = "So x = 3"
    assert su.score_one(p)["hit"] == 0


def test_official_rule():
    assert so.can_infer("B", ABCD) == "B"
    assert so.can_infer("The answer is B", ABCD) == "B"
    # 풀이에 A, B 두 글자가 나오면 규칙 실패 -> 판정기 필요
    assert so.can_infer("A is wrong, so B", ABCD) is False
    s = so.score_one(mcq("A is wrong, so B"), judge=None, rng=__import__("random").Random(0))
    assert s["method"] == "rule_failed_no_judge"


def test_official_open_to_mc():
    p = {**mcq("x = 0.8"), "question_type": "open", "choices": {}, "answer": "0.8"}
    ch, gt = so.to_official_mc(p)
    assert ch == {"A": "0.8", "B": "Other Answers"} and gt == "A"


class FakeJudge:
    model = "fake"

    def generate(self, prompt):
        return "B"


def test_official_judge_path():
    import random
    s = so.score_one(mcq("A is wrong, so B"), judge=FakeJudge(), rng=random.Random(0))
    assert s == {"extracted": "B", "gt": "B", "method": "judge", "hit": 1}


def test_end_to_end_score_and_report(tmp_path):
    """가짜 predictions 로 score.py + report.py 전체 실행, 산술 정합성 확인."""
    import random
    rng = random.Random(0)
    run = tmp_path / "ours"
    truth = {}
    for seed in (1, 2):
        d = run / f"seed{seed}"
        d.mkdir(parents=True)
        preds, correct = [], 0
        for si, subj in enumerate(mc.SUBJECTS):
            for q in range(30):
                ok = rng.random() < 0.6
                correct += ok
                preds.append({**mcq(f"Answer: {'B' if ok else 'C'}"), "idx": si * 30 + q, "id": f"{subj}_{q}",
                              "subject": subj, "finish_reason": "stop", "num_output_tokens": 10})
        truth[seed] = correct
        with open(d / "predictions.jsonl", "w") as f:
            f.writelines(json.dumps(p) + "\n" for p in preds)
        json.dump({"args": {"seed": seed}, "timing_sec_by_subject": {s: 1.0 for s in mc.SUBJECTS},
                   "total_sec": 30.0}, open(d / "predictions_meta.json", "w"))
        env = {**os.environ, "PYTHONPATH": os.path.join(ROOT, "src")}
        subprocess.run([sys.executable, os.path.join(ROOT, "src/score.py"), "--pred_file",
                        str(d / "predictions.jsonl"), "--method", "ours"], check=True, env=env)
        s = json.load(open(d / "scores_ours_summary.json"))
        assert s["num_correct"] == correct
        assert abs(s["overall_macro"] - correct / 900) < 1e-12  # 균등 30문제 -> macro == micro
    subprocess.run([sys.executable, os.path.join(ROOT, "src/report.py"), "--run_dir", str(run),
                    "--method", "ours"], check=True, env=env)
    md = open(run / "report_ours.md").read()
    mean = (truth[1] + truth[2]) / 2 / 900 * 100
    assert f"{mean:.2f}" in md


def test_ours_smoke_regressions():
    """스모크 테스트에서 발견된 실제 응답들."""
    assert su.score_one(mcq("B. is still active today"))["extracted"] == "B"
    assert su.score_one(mcq("B. False", choices={"A": "True", "B": "False"}))["extracted"] == "B"
    assert su.score_one(mcq("so the answer is:\n$$\n\\boxed{\\text{C. Not exist}}\n$$", gold="C"))["extracted"] == "C"
    # 문장 첫 단어 'A' (관사)는 선택지로 보지 않음
    assert su.extract_leading_option("A bar is loaded axially", list(ABCD)) is None
    assert su.extract_leading_option("(C)", list(ABCD)) == "C"
    assert su.extract_leading_option("**D.** 42", list(ABCD)) == "D"


def test_qwen_cot_prompt_matches_official_code():
    """Qwen 공식 run_mmmu.py: prompt.rstrip() 뒤에 cot_prompt 를 그대로 이어 붙임."""
    row = {"id": "x", "question": "Q?", "options": "['a', 'b']", "answer": "A",
           "question_type": "multiple-choice", **{f"image_{k}": None for k in range(1, 8)}}
    rec = mc.row_to_record(row, "Math")
    t = mc.build_prompt_text(rec, use_cot=True)
    assert t.startswith("Question: Q?\nOptions:\nA. a\nB. b\nPlease select the correct answer from the options above. If you")
    assert "Avoid repeating steps indefinitely—provide your best guess even if unsure." in t
    assert t.endswith("considering all relevant information before answering.")
    assert mc.build_prompt_text(rec) == t.split(" If you are uncertain")[0]


def test_open_final_answer_first():
    """공식/우리 채점 불일치 주관식 수동 검토에서 나온 실제 사례."""
    def hit(resp, gold):
        return su.score_one({"question_type": "open", "choices": {}, "answer": gold, "response": resp})["hit"]
    assert hit("✅ **Answer: \\boxed{1464}**", "1464") == 1
    assert hit("### Final Answer: **Step 2**", "2") == 1
    assert hit("so \\( \\boxed{\\frac{24}{7}} \\) ft/s", ["24/7", "3.429"]) == 1
    assert hit("contract 2,000,000 francs ... cost = 500,000.\n\\boxed{500000}", "2000000") == 0
    assert hit("### ✅ Final Answer: **1/16**", "1/64") == 0
    assert hit("### ✅ Final Answer: **1/16**", "1") == 0      # 분자만 따로 맞추지 않음
    assert hit("Therefore the result is 0.80.", ["0.8", "4/5"]) == 1   # 최종답 없으면 MMMU 파서


def test_truncated_is_wrong():
    """잘린 응답(finish_reason=length)은 답이 보여도 오답 처리. 공식 채점은 그대로."""
    p = {**mcq("... I think the correct answer is Option B. But wait"), "finish_reason": "length"}
    assert su.score_one(p) == {"extracted": None, "gt": "B", "method": "truncated", "hit": 0}
    p["finish_reason"] = "stop"
    assert su.score_one(p)["hit"] == 1


def test_cot_heldout_audit_formats():
    """CoT 실행(규칙 설계 때 안 본 답변) 점검에서 놓친 최종답 형식."""
    five = {k: str(i) for i, k in enumerate("ABCDE")}
    r1 = "### ✅ Final Answer: $\\boxed{194}$\n\n**Answer choice: D. 194**\n\n---"
    assert su.score_one(mcq(r1, gold="D", choices=five))["extracted"] == "D"
    r2 = "### Final Answer:\n\n✅ **A. 3.8-hp input, 2.3-hp output**\n\n*(Note: Option C is identical.)*"
    s = su.score_one(mcq(r2, gold="B", choices=five))
    assert s["extracted"] == "A" and s["method"] == "final_answer_pattern"
    # 수식 속 문자는 여전히 무시
    assert su.extract_final_answer("Final Answer: $x = 3$", list(ABCD)) is None
