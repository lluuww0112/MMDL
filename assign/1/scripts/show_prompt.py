"""보고서용: 챗 템플릿이 적용된 실제 입력 문자열 출력 (GPU 불필요).
사용: PYTHONPATH=src python scripts/show_prompt.py [과목] [--open] [--use_cot]"""
import sys
from transformers import AutoProcessor
from mmmu_common import MODEL_ID, MODEL_REVISION, load_subject, build_messages

args = [a for a in sys.argv[1:] if not a.startswith("--")]
subj = args[0] if args else "Accounting"
want = "open" if "--open" in sys.argv else "multiple-choice"
rec = next(r for r in load_subject("MMMU/MMMU", subj) if r["question_type"] == want)
proc = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
text = proc.apply_chat_template(build_messages(rec, use_cot="--use_cot" in sys.argv),
                                tokenize=False, add_generation_prompt=True)
print(f"# id={rec['id']}  type={rec['question_type']}  images={len(rec['images'])}")
print(repr(text))
print("-----")
print(text)
