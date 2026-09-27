"""MMMU-val 데이터 로딩 + 프롬프트 생성 (①②단계). official/ours 두 모드가 공통으로 사용.

프롬프트와 이미지 처리는 Qwen 공식 MMMU 평가 코드를 그대로 따른다:
  https://github.com/QwenLM/Qwen3-VL/blob/main/evaluation/mmmu/run_mmmu.py  (build_mmmu_prompt)
공식 코드는 VLMEvalKit의 MMMU_DEV_VAL.tsv를 읽지만, 과제 규정상 HF MMMU/MMMU(pinned revision)를
읽어서 같은 형태(질문/선택지 A.., 이미지 리스트)로 변환한다.
"""
import ast
import string

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
MODEL_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
DATASET_ID = "MMMU/MMMU"
DATASET_REVISION = "98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68"
SPLIT = "validation"

# Qwen 공식 run_mmmu.py 의 값 그대로 (28 은 Qwen2.5-VL 시절 상수; 픽셀 수 상/하한으로만 쓰임)
MIN_PIXELS = 1280 * 28 * 28  # 1,003,520 px
MAX_PIXELS = 5120 * 28 * 28  # 4,014,080 px

SUBJECTS = [
    "Accounting", "Agriculture", "Architecture_and_Engineering", "Art", "Art_Theory",
    "Basic_Medical_Science", "Biology", "Chemistry", "Clinical_Medicine", "Computer_Science",
    "Design", "Diagnostics_and_Laboratory_Medicine", "Economics", "Electronics", "Energy_and_Power",
    "Finance", "Geography", "History", "Literature", "Manage", "Marketing", "Materials", "Math",
    "Mechanical_Engineering", "Music", "Pharmacy", "Physics", "Psychology", "Public_Health", "Sociology",
]

# MMMU 논문의 6개 분야 분류
CATEGORIES = {
    "Art & Design": ["Art", "Art_Theory", "Design", "Music"],
    "Business": ["Accounting", "Economics", "Finance", "Manage", "Marketing"],
    "Science": ["Biology", "Chemistry", "Geography", "Math", "Physics"],
    "Health & Medicine": ["Basic_Medical_Science", "Clinical_Medicine",
                          "Diagnostics_and_Laboratory_Medicine", "Pharmacy", "Public_Health"],
    "Humanities & Social Science": ["History", "Literature", "Psychology", "Sociology"],
    "Tech & Engineering": ["Agriculture", "Architecture_and_Engineering", "Computer_Science",
                           "Electronics", "Energy_and_Power", "Materials", "Mechanical_Engineering"],
}
assert sorted(s for v in CATEGORIES.values() for s in v) == sorted(SUBJECTS)


def parse_options(raw):
    """HF MMMU 의 options 는 "['$6', '$7']" 같은 문자열 -> list."""
    if raw is None or raw == "":
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw]
    return [str(x) for x in ast.literal_eval(raw)]


def parse_gold(answer, question_type):
    """객관식: 'B'. 주관식: 문자열, 또는 "['0.8', '4/5']" 같은 리스트 문자열이면 list."""
    if question_type == "multiple-choice":
        return str(answer).strip()
    a = str(answer).strip()
    if a.startswith("[") and a.endswith("]"):
        try:
            v = ast.literal_eval(a)
            if isinstance(v, list):
                return [str(x) for x in v]
        except (ValueError, SyntaxError):
            pass
    return a


def row_to_record(row, subject):
    """HF 한 행 -> 파이프라인 내부 레코드 (이미지는 PIL 그대로)."""
    options = parse_options(row.get("options"))
    images = [row[f"image_{k}"] for k in range(1, 8) if row.get(f"image_{k}") is not None]
    return {
        "id": row["id"],
        "subject": subject,
        "question": row["question"],
        "options": options,                       # ['$6', '$7', ...]
        "choices": {string.ascii_uppercase[i]: o for i, o in enumerate(options)},
        "question_type": row["question_type"],    # 'multiple-choice' | 'open'
        "answer": parse_gold(row["answer"], row["question_type"]),
        "images": images,
    }


def load_subject(data_path, subject, cache_dir=None, revision=DATASET_REVISION):
    """data_path: HF repo id(MMMU/MMMU) 또는 같은 revision 을 받아둔 로컬 스냅샷 폴더."""
    from datasets import load_dataset
    kwargs = dict(split=SPLIT, cache_dir=cache_dir)
    if data_path == DATASET_ID:
        kwargs["revision"] = revision
    ds = load_dataset(data_path, subject, **kwargs)
    return [row_to_record(r, subject) for r in ds]


# Qwen 공식 run_mmmu.py 의 --use-cot 기본 문구 (앞 공백 포함, 원문 그대로)
QWEN_COT_PROMPT = (" If you are uncertain or the problem is too complex, make a reasoned guess based on the "
                   "information provided. Avoid repeating steps indefinitely\u2014provide your best guess even if "
                   "unsure. Determine whether to think step by step based on the difficulty of the question, "
                   "considering all relevant information before answering.")


def build_prompt_text(rec, use_cot=False):
    """Qwen 공식 build_mmmu_prompt 와 동일한 텍스트 (MMMU 에는 hint 필드가 없음)."""
    prompt = f"Question: {rec['question']}\n"
    if rec["choices"]:
        prompt += "Options:\n"
        for key, item in rec["choices"].items():
            prompt += f"{key}. {item}\n"
        prompt += "Please select the correct answer from the options above. \n"
    prompt = prompt.rstrip()
    # 공식 코드: rstrip 된 프롬프트 뒤에 cot 문구를 그대로 이어 붙임 (last_content['text'] += cot_prompt)
    return prompt + QWEN_COT_PROMPT if use_cot else prompt


def build_messages(rec, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS, use_cot=False):
    """공식 코드와 동일: 모든 이미지를 먼저, 그 뒤에 텍스트 1개 (인터리브하지 않음)."""
    content = [{"type": "image", "image": img, "min_pixels": min_pixels, "max_pixels": max_pixels}
               for img in rec["images"]]
    content.append({"type": "text", "text": build_prompt_text(rec, use_cot)})
    return [{"role": "user", "content": content}]
