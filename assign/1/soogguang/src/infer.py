"""③단계: vLLM 으로 900문제 응답 생성 -> predictions.jsonl

  --seed_mode global       : 공식 코드와 동일. LLM(seed=S) 한 번만 설정.
                             (배치 구성/스케줄링에 따라 같은 seed 여도 결과가 달라질 수 있음)
  --seed_mode per_request  : 우리 방식. 문제마다 SamplingParams(seed=S*100000+idx)를 고정해서
                             배치 순서와 무관하게 같은 문제엔 같은 난수열이 쓰이도록 함.
"""
import argparse
import json
import os
import time

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from mmmu_common import (MODEL_ID, MODEL_REVISION, DATASET_ID, DATASET_REVISION, SUBJECTS,
                         MIN_PIXELS, MAX_PIXELS, load_subject, build_messages)


def load_resume_state(output_file, expected):
    """이미 끝난 과목의 예측과 소요시간을 불러옴. 과목의 문제 수가 다 채워진 경우만 '완료'로 인정.
    expected: {subject: 문제 수}. 반환: (완료 과목 set, 유지할 줄 list, timing dict)"""
    import re
    from collections import Counter
    if not os.path.exists(output_file):
        return set(), [], {}
    lines = [l for l in open(output_file) if l.strip()]
    cnt = Counter(json.loads(l)["subject"] for l in lines)
    done = {s for s, n in expected.items() if cnt.get(s, 0) == n}
    keep = [l if l.endswith("\n") else l + "\n" for l in lines if json.loads(l)["subject"] in done]
    timing = {}
    log = os.path.join(os.path.dirname(os.path.abspath(output_file)), "infer.log")
    if os.path.exists(log):
        for m in re.finditer(r"\[infer\] (\S+)\s+(\d+) q\s+([\d.]+)s", open(log, errors="ignore").read()):
            if m.group(1) in done:
                timing[m.group(1)] = float(m.group(3))
    return done, keep, timing


def prepare_inputs_for_vllm(messages, processor):
    """Qwen 공식 run_mmmu.py 의 prepare_inputs_for_vllm 과 동일."""
    from qwen_vl_utils import process_vision_info
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True, return_video_metadata=True)
    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs
    return {"prompt": text, "multi_modal_data": mm_data, "mm_processor_kwargs": video_kwargs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=MODEL_ID, help="HF repo id 또는 로컬 체크포인트 폴더")
    ap.add_argument("--model_revision", default=MODEL_REVISION, help="HF repo id 일 때만 사용")
    ap.add_argument("--data_path", default=DATASET_ID, help="HF repo id 또는 MMMU 로컬 스냅샷 폴더")
    ap.add_argument("--data_revision", default=DATASET_REVISION)
    ap.add_argument("--cache_dir", default=None, help="HF datasets 캐시 위치 (선택)")
    ap.add_argument("--output_file", required=True)
    # 생성 설정 (기본값 = Qwen 공식 instruct recipe / infer_instruct.sh)
    ap.add_argument("--max_new_tokens", type=int, default=16384)  # HF 모델 카드 VL out_seq_length (공식 스크립트는 32768)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--repetition_penalty", type=float, default=1.0)
    ap.add_argument("--presence_penalty", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seed_mode", choices=["global", "per_request"], default="global")
    ap.add_argument("--min_pixels", type=int, default=MIN_PIXELS)
    ap.add_argument("--max_pixels", type=int, default=MAX_PIXELS)
    # vLLM / 하드웨어
    ap.add_argument("--max_model_len", type=int, default=49152)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    ap.add_argument("--max_num_seqs", type=int, default=64)
    ap.add_argument("--bound_processor_pixels", action="store_true",
                    help="시작 시 OOM 이 나면 사용: vLLM 프로세서에도 같은 min/max_pixels 를 줘서 메모리 프로파일링 크기를 제한"
                         " (이미지는 이미 qwen_vl_utils 로 같은 범위에 맞춰져 있으므로 입력 자체는 바뀌지 않음)")
    ap.add_argument("--use_cot", action="store_true",
                    help="Qwen 공식 run_mmmu.py 의 --use-cot 문구를 프롬프트 끝에 붙임 (공식 기본값은 꺼짐)")
    ap.add_argument("--resume", action="store_true",
                    help="output_file 에 이미 끝난 과목은 건너뛰고 나머지만 생성 (중단 후 재시작용)")
    ap.add_argument("--limit_per_subject", type=int, default=0, help="스모크 테스트용 (0=전체)")
    args = ap.parse_args()

    import torch
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    is_hub_model = not os.path.isdir(args.model_path)
    rev = args.model_revision if is_hub_model else None

    # ① 데이터 로딩 + ② 프롬프트
    records = []
    for subj in SUBJECTS:
        recs = load_subject(args.data_path, subj, cache_dir=args.cache_dir, revision=args.data_revision)
        if args.limit_per_subject:
            recs = recs[: args.limit_per_subject]
        records.extend(recs)
    print(f"[data] {len(records)} questions from {len(SUBJECTS)} subjects")
    max_imgs = max(len(r["images"]) for r in records)

    processor = AutoProcessor.from_pretrained(args.model_path, revision=rev)
    llm = LLM(model=args.model_path, revision=rev, tokenizer_revision=rev,
              dtype="bfloat16", seed=args.seed,
              max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization,
              max_num_seqs=args.max_num_seqs,
              limit_mm_per_prompt={"image": max_imgs},
              tensor_parallel_size=max(1, torch.cuda.device_count()),
              **({"mm_processor_kwargs": {"min_pixels": args.min_pixels, "max_pixels": args.max_pixels}}
                 if args.bound_processor_pixels else {}))

    def sampling(idx):
        return SamplingParams(
            temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
            repetition_penalty=args.repetition_penalty, presence_penalty=args.presence_penalty,
            max_tokens=args.max_new_tokens,
            seed=(args.seed * 100000 + idx) if args.seed_mode == "per_request" else None)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    expected = {s: sum(1 for r in records if r["subject"] == s) for s in SUBJECTS}
    done, keep, timing = (load_resume_state(args.output_file, expected) if args.resume else (set(), [], {}))
    if args.resume:
        print(f"[resume] 완료된 과목 {len(done)}개 ({sum(expected[s] for s in done)} q) 건너뜀: {sorted(done)}")
    t_all = time.time()
    with open(args.output_file, "w") as fout:
        fout.writelines(keep)   # resume: 완료된 과목만 남기고 (중간에 끊긴 과목은 버리고) 다시 씀
        fout.flush()
        # 과목별 소요시간을 재기 위해 과목 단위로 배치 생성 (과목 내부는 vLLM 이 자동 배치)
        for subj in SUBJECTS:
            if subj in done:
                continue
            recs = [(i, r) for i, r in enumerate(records) if r["subject"] == subj]
            msgs = [build_messages(r, args.min_pixels, args.max_pixels, args.use_cot) for _, r in recs]
            inputs = [prepare_inputs_for_vllm(m, processor) for m in msgs]
            params = [sampling(i) for i, _ in recs]
            t0 = time.time()
            outs = llm.generate(inputs, sampling_params=params)
            timing[subj] = time.time() - t0
            for (i, r), m, o in zip(recs, msgs, outs):
                c = o.outputs[0]
                prompt_text = m[0]["content"][-1]["text"]
                fout.write(json.dumps({
                    "idx": i, "id": r["id"], "subject": subj,
                    "question_type": r["question_type"], "question": r["question"],
                    "choices": r["choices"], "answer": r["answer"],
                    "num_images": len(r["images"]), "prompt_text": prompt_text,
                    "response": c.text,
                    "finish_reason": c.finish_reason,          # 'length' 이면 max_new_tokens 에 잘린 것
                    "num_output_tokens": len(c.token_ids),
                    "num_prompt_tokens": len(o.prompt_token_ids or []),
                }, ensure_ascii=False) + "\n")
            fout.flush()   # 과목마다 디스크에 기록 -> 중단돼도 끝난 과목은 보존
            print(f"[infer] {subj:40s} {len(recs):3d} q  {timing[subj]:7.1f}s")
    total = sum(timing.values()) if done else time.time() - t_all

    meta = {"args": vars(args), "timing_sec_by_subject": timing, "total_sec": total,
            "resumed_subjects": sorted(done),
            "num_questions": len(records), "max_images_per_question": max_imgs}
    with open(args.output_file.replace(".jsonl", "_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[infer] done: {len(records)} q in {total/60:.1f} min -> {args.output_file}")


if __name__ == "__main__":
    main()