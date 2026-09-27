"""④단계 - ours 모드: 외부 API 없이 결정론적으로 채점.

설계 이유
  - 공식 채점은 GPT 판정기(gpt-3.5-turbo-0125)에 의존 -> 유료, 모델 종료 예정(2026-10-23),
    fine-tuning 후 재평가 시점에 같은 판정기를 쓸 수 없음 => 비교 가능성이 깨짐.
  - 공식/ MMMU 원본 파서는 추출 실패 시 '무작위 선택' -> 같은 응답도 점수가 달라질 수 있음.
  - fine-tuning 데이터(MMFineReason)는 긴 풀이형 응답을 유도할 가능성이 높음 -> 풀이 중간에
    등장한 선택지 글자에 속지 않고 '최종 답' 선언을 우선적으로 잡아야 함.

규칙 (공통)
  0) finish_reason == 'length' (max_new_tokens 에 잘림) -> 추출하지 않고 오답 ('truncated')
규칙 (객관식)
  1) </think> 이후만 사용
  2) 명시적 최종답 패턴('Answer: B', 'The answer is (B)', '\\boxed{B}', '**Answer:** B' 등) 중
     유효한 선택지 글자의 '마지막' 등장
  2-1) 없으면 답변 맨 앞의 선택지 글자 ('B. ...', 'B) ...', '(B)', 'B')
  3) 없으면 MMMU 공식 parse_multi_choice_response (괄호 글자 -> 단독 글자 -> 선택지 내용, 여러 개면 마지막)
     단, 원본의 random.choice fallback 은 제거
  4) 그래도 없으면 추출 실패 = 오답, 건수를 따로 보고
규칙 (주관식): 최종 답(\\boxed{..} / 'Final Answer:' 줄)이 있으면 그것만 eval_open 으로 비교,
  없으면 MMMU 공식 parse_open_response + eval_open
출처: https://github.com/MMMU-Benchmark/MMMU/blob/main/mmmu/utils/eval_utils.py
"""
import re

import numpy as np

FINAL_ANSWER_PATTERNS = [
    re.compile(r"\\boxed\{\s*(?:\\text\{\s*)?\(?([A-Z])(?![A-Za-z0-9])"),   # \boxed{B}, \boxed{\text{C. ...}}
    re.compile(r"(?i:\b(?:final\s+answer|correct\s+answer|correct\s+option|correct\s+choice|answer))"
               r"(?:\s*(?i:choice|option))?\s*(?i:is)?\s*[:：]?"      # 'Answer choice: D' (CoT 점검에서 발견)
               r"[^\w]{0,12}"                                           # '✅ **', 줄바꿈, '> ' 등 기호만 허용
               r"(?i:option\s*|choice\s*)?[\(\[\*]*([A-Z])(?![A-Za-z0-9])"),
]


def extract_final_answer(response, all_choices):
    best = None  # (position, letter)
    for pat in FINAL_ANSWER_PATTERNS:
        for m in pat.finditer(response):
            if m.group(1) in all_choices and (best is None or m.start(1) > best[0]):
                best = (m.start(1), m.group(1))
    return best[1] if best else None


# 답변이 선택지 글자로 바로 시작하는 경우: "B. is still active today", "B) ...", "(B)", "**B.**", "B"
LEADING_OPTION = re.compile(r"^\s*[\*\(\[]*([A-Z])[\)\]\*]*(?:[\.\):]|\s*$)")


def extract_leading_option(response, all_choices):
    m = LEADING_OPTION.match(response)
    return m.group(1) if m and m.group(1) in all_choices else None


# ---------------- MMMU 공식 파서 (random fallback 만 제거) ----------------
def parse_multi_choice_response(response, all_choices, index2ans):
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "
    index_ans, ans_with_brack, candidates = True, False, []
    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True
    if len(candidates) == 0:
        for choice in all_choices:
            if f" {choice} " in response:
                candidates.append(choice)
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False
    if len(candidates) == 0:
        return None  # 원본: random.choice(all_choices)
    if len(candidates) == 1:
        return candidates[0]
    if index_ans:
        key = (lambda c: response.rfind(f"({c})")) if ans_with_brack else (lambda c: response.rfind(f" {c} "))
    else:
        key = lambda c: response.lower().rfind(index2ans[c].lower())  # noqa: E731
    return candidates[int(np.argmax([key(c) for c in candidates]))]


def check_is_number(s):
    try:
        float(s.replace(",", ""))
        return True
    except ValueError:
        return False


def normalize_str(s):
    s = s.strip()
    if check_is_number(s):
        return [round(float(s.replace(",", "")), 2)]
    s = s.lower()
    return [" " + s, s + " "] if len(s) == 1 else [s]


def extract_numbers(s):
    pattern_commas = r"-?\b\d{1,3}(?:,\d{3})+\b"
    pattern_scientific = r"-?\d+(?:\.\d+)?[eE][+-]?\d+"
    pattern_simple = r"-?(?:\d+\.\d+|\.\d+|\d+\b)(?![eE][+-]?\d+)(?![,\d])"
    return re.findall(pattern_commas, s) + re.findall(pattern_scientific, s) + re.findall(pattern_simple, s)


def parse_open_response(response):
    def get_key_subresponses(response):
        response = response.strip().strip(".").lower()
        sub_responses = re.split(r"\.\s(?=[A-Z])|\n", response)
        indicators = ["could be ", "so ", "is ", "thus ", "therefore ", "final ", "answer ", "result "]
        key_responses = []
        for index, resp in enumerate(sub_responses):
            if index == len(sub_responses) - 1:
                indicators = indicators + ["="]
            shortest = None
            for ind in indicators:
                if ind in resp:
                    cand = resp.split(ind)[-1].strip()
                    if shortest is None or len(cand) < len(shortest):
                        shortest = cand
            if shortest and shortest.strip() not in [":", ",", ".", "!", "?", ";", ":", "'"]:
                key_responses.append(shortest)
        return key_responses if key_responses else [response]

    key_responses = get_key_subresponses(response)
    pred_list = key_responses.copy()
    for resp in key_responses:
        pred_list.extend(extract_numbers(resp))
    out = []
    for p in pred_list:
        out.extend(normalize_str(p))
    return list(set(out))


def eval_open(gold, pred_list):
    golds = gold if isinstance(gold, list) else [gold]
    norm_answers = []
    for g in golds:
        norm_answers.extend(normalize_str(g))
    for pred in pred_list:
        if isinstance(pred, str):
            if any(isinstance(a, str) and a in pred for a in norm_answers):
                return True
        elif pred in norm_answers:
            return True
    return False


# ---------------- 주관식: 최종 답 우선 (공식/우리 채점 불일치 29건 수동 검토 후 추가) ----------------
# MMMU 원본 parse_open_response 는 'so/is/answer ' 같은 단어가 있는 줄만 보기 때문에
# '**Answer: \\boxed{1464}**', 'Final Answer: **Step 2**' 같은 최종 답 줄을 놓치고,
# 반대로 풀이 중 등장한 숫자를 잡아 오답을 정답 처리하기도 함 (Finance_10: 최종 500000, 정답 2000000).
# -> 명시적 최종 답(\boxed{..} 또는 'Final Answer:/Answer:' 줄)이 있으면 그것만 비교, 없으면 MMMU 원본 파서.
_ANSWER_LINE = re.compile(r"(?i)\b(?:final\s+answer|answer)\s*\**\s*[:：]\s*(.*)")


def _last_boxed(text):
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + len("\\boxed{"), 1
    for k in range(j, len(text)):
        depth += {"{": 1, "}": -1}.get(text[k], 0)
        if depth == 0:
            return text[j:k]
    return None


def extract_open_final(response):
    boxed = _last_boxed(response)
    if boxed is not None and boxed.strip():
        return boxed
    lines = [l for l in response.splitlines() if l.strip()]
    for n in range(len(lines) - 1, -1, -1):
        m = _ANSWER_LINE.search(lines[n])
        if m:
            rest = m.group(1).strip(" *")
            return rest if rest else (lines[n + 1] if n + 1 < len(lines) else None)
    return None


def parse_open_final(final):
    t = re.sub(r"\\d?frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", final)      # \frac{24}{7} -> 24/7
    t = re.sub(r"\\text\{([^{}]*)\}", r"\1", t)
    t = re.sub(r"\\[,;!quad ]+|[$*`]", " ", t).strip()
    out = normalize_str(t)
    frac = re.compile(r"(-?\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)")
    for a, b in frac.findall(t):          # 분수는 값으로 비교 (1/16 -> 0.06). 분자·분모를 따로 숫자로 뽑지 않음
        if float(b) != 0:
            out.append(round(float(a) / float(b), 2))
    for num in extract_numbers(frac.sub(" ", t)):
        out.extend(normalize_str(num))
    return list(set(out))


# ---------------- 채점 ----------------
def score_one(pred):
    # max_new_tokens 에 잘린 응답은 최종 답을 내지 못한 것으로 보고 오답 처리.
    # (수동 점검: 잘린 응답에서 뽑힌 글자는 풀이 도중 언급된 선택지라 사실상 무작위)
    if pred.get("finish_reason") == "length":
        return {"extracted": None, "gt": pred["answer"], "method": "truncated", "hit": 0}
    response = str(pred["response"]).split("</think>")[-1].strip()
    if pred["question_type"] == "multiple-choice" and pred["choices"]:
        all_choices = list(pred["choices"].keys())
        opt = extract_final_answer(response, all_choices)
        method = "final_answer_pattern"
        if opt is None:
            opt = extract_leading_option(response, all_choices)
            method = "leading_option"
        if opt is None:
            opt = parse_multi_choice_response(response, all_choices, pred["choices"])
            method = "mmmu_parser"
        if opt is None:
            method = "failed"
        return {"extracted": opt, "gt": pred["answer"], "method": method,
                "hit": int(opt is not None and opt == pred["answer"])}
    final = extract_open_final(response)
    if final is not None:
        parsed, method = parse_open_final(final), "open_final_answer"
    else:
        parsed, method = parse_open_response(response), "mmmu_open"
    return {"extracted": sorted({str(x) for x in parsed})[:10], "gt": pred["answer"], "method": method,
            "hit": int(eval_open(pred["answer"], parsed))}


def score_file(preds):
    return [score_one(p) for p in preds]