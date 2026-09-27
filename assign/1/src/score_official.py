"""④단계 - official 모드: Qwen 공식 MMMU 채점 로직 재현.

출처: https://github.com/QwenLM/Qwen3-VL/blob/main/evaluation/mmmu/eval_utils.py (VLMEvalKit 계열)
  1) 주관식은 '객관식'으로 바꿔 채점: A = 정답, B = 'Other Answers', GT = 'A'   (MMMU_preproc)
  2) 규칙 추출 can_infer: 응답에서 선택지 글자가 정확히 1개만 단독으로 등장하면 채택,
     아니면 선택지 '내용'이 응답에 정확히 1개만 포함되면 채택
  3) 규칙 실패 시 GPT 판정기에게 "응답이 어느 선택지와 가장 비슷한가" 질문 (temperature 0, 최대 25회 재시도)
  4) 그래도 실패하면 무작위 선택 (공식 코드는 seed 미고정 -> 여기서는 재현을 위해 seed 고정)
공식 기본 판정 모델: gpt-3.5-turbo-0125 (OpenAI 공지상 2026-10-23 종료 예정)
"""
import copy
import os
import random
import string
import time

import requests

JUDGE_FAIL = "Failed to obtain answer via API."


# ---------- 규칙 기반 (공식 코드 그대로) ----------
def can_infer_option(answer, choices):
    if JUDGE_FAIL in answer:
        return False
    reject_to_answer = [
        "Sorry, I can't help with images of people yet.",
        "I can't process this file.",
        "I'm sorry, but without the image provided",
        "Cannot determine the answer",
    ]
    for err in reject_to_answer:
        if err in answer:
            return "Z"

    def count_choice(splits, choices, prefix="", suffix=""):
        return sum(1 for c in choices if prefix + c + suffix in splits)

    answer_mod = copy.copy(answer)
    for c in ".()[],:;!*#{}":
        answer_mod = answer_mod.replace(c, " ")
    splits = [x.strip() for x in answer_mod.split()]
    count = count_choice(splits, choices)
    if count == 1:
        for ch in choices:
            if "A" in splits and len(splits) > 3:
                return False  # 'A' 가 관사일 수 있음
            if ch in splits:
                return ch
    elif count == 0 and count_choice(splits, {"Z", ""}) == 1:
        return "Z"
    return False


def can_infer_text(answer, choices):
    answer = answer.lower()
    choices = {k: str(v).lower() for k, v in choices.items()}
    cands = [k for k in choices if choices[k] in answer]
    return cands[0] if len(cands) == 1 else False


def can_infer(answer, choices):
    answer = str(answer)
    copt = can_infer_option(answer, choices)
    return copt if copt else can_infer_text(answer, choices)


def build_judge_prompt(question, choices, prediction):
    options = "There are several options: \n" + "".join(f"{c}. {v}\n" for c, v in choices.items())
    tmpl = (
        "You are an AI assistant who will help me to match "
        "an answer with several options of a single-choice question. "
        "You are provided with a question, several options, and an answer, "
        "and you need to find which option is most similar to the answer. "
        "If the meaning of all options are significantly different from the answer, output Z. "
        "Your should output a single uppercase character in A, B, C, D (if they are valid options), and Z. \n"
        "Example 1: \n"
        "Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n"
        "Answer: a cute teddy bear\nYour output: A\n"
        "Example 2: \n"
        "Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n"
        "Answer: Spider\nYour output: Z\n"
        "Example 3: \n"
        "Question: {}?\nOptions: {}\nAnswer: {}\nYour output: "
    )
    return tmpl.format(question, options, prediction)


class OpenAIJudge:
    def __init__(self, model, api_base=None, api_key=None, timeout=60, retry=5, wait=5):
        self.model = model
        self.api_base = api_base or os.environ.get("OPENAI_API_BASE",
                                                   "https://api.openai.com/v1/chat/completions")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.timeout, self.retry, self.wait = timeout, retry, wait

    def generate(self, prompt):
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        payload = {"model": self.model, "temperature": 0, "max_tokens": 4096,  # 공식 코드 값
                   "messages": [{"role": "user", "content": prompt}]}
        for _ in range(self.retry):
            try:
                r = requests.post(self.api_base, headers=headers, json=payload, timeout=self.timeout)
                if r.status_code == 200:
                    return r.json()["choices"][0]["message"]["content"].strip()
                print(f"[judge] HTTP {r.status_code}: {r.text[:200]}")
            except Exception as e:  # noqa: BLE001
                print(f"[judge] error: {e}")
            time.sleep(self.wait)
        return JUDGE_FAIL


def to_official_mc(pred):
    """MMMU_preproc: 주관식 -> {A: 정답, B: 'Other Answers'}, GT='A'."""
    if pred["question_type"] == "multiple-choice" and pred["choices"]:
        return dict(pred["choices"]), pred["answer"]
    gold = pred["answer"]
    gold_text = gold if isinstance(gold, str) else str(gold)  # TSV 에는 리스트도 문자열로 들어있음
    return {"A": gold_text, "B": "Other Answers"}, "A"


def score_one(pred, judge, rng, judge_retry=25):
    response = str(pred["response"]).split("</think>")[-1].strip()
    choices, gt = to_official_mc(pred)
    ret = can_infer(response, choices)
    if ret and ret != "Z":
        method, opt = "rule", ret
    elif ret == "Z":
        method, opt = "rule_Z", "Z"
    elif judge is None:
        method, opt = "rule_failed_no_judge", None
    else:
        prompt = build_judge_prompt(pred["question"], choices, response)
        opt = None
        for _ in range(judge_retry):
            ans = judge.generate(prompt)
            r = can_infer(ans, choices) if JUDGE_FAIL not in ans else False
            if r and r != "Z":
                opt, method = r, "judge"
                break
            time.sleep(rng.random() * 2)
        if opt is None:
            opt, method = rng.choice(list(choices) + ["Z"]), "random"
    return {"extracted": opt, "gt": gt, "method": method, "hit": int(opt == gt)}


def score_file(preds, judge_model="gpt-3.5-turbo-0125", use_judge=True, seed=0, nproc=8):
    from concurrent.futures import ThreadPoolExecutor
    judge = OpenAIJudge(judge_model) if use_judge else None
    rngs = [random.Random(seed * 1000003 + p["idx"]) for p in preds]
    with ThreadPoolExecutor(max_workers=nproc) as ex:
        results = list(ex.map(lambda a: score_one(a[0], judge, a[1]), zip(preds, rngs)))
    return results
