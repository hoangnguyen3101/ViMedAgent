"""
Lượng tử hoá W4A16 (GPTQ, int4 theo nhóm 128) cho MedGemma-4B — lặp lại ĐÚNG recipe mà RedHatAI đã
dùng để tạo RedHatAI/gemma-3-4b-it-quantized.w4a16 (bản Gemma3 4-bit mà benchmark dùng).

Vì sao phải đúng recipe đó: hiệu số MedGemma − Gemma3 chỉ đo được tác dụng của tinh chỉnh y khoa khi
HAI model đi qua CÙNG một quy trình lượng tử hoá. Trên Hugging Face không có bản MedGemma 4-bit nào làm
theo recipe của RedHatAI, nên tự làm lại. Mọi tham số dưới đây sao chép từ model card + recipe.yaml của
RedHatAI/gemma-3-4b-it-quantized.w4a16 (sha 9288be4) — đừng sửa.

Script cũng chạy được cho google/gemma-3-4b-it, để kiểm chứng: bản tự lượng tử hoá phải cho kết quả
pilot gần bản RedHatAI (vd agreement zero_shot ≥ ~90%) thì mới tin là recipe đã lặp lại đúng.

Yêu cầu: GPU ≥ 16 GB, ~10–30 phút. Môi trường riêng (đừng cài vào env của server SGLang):
    pip install "llmcompressor>=0.6" datasets "transformers>=4.50" accelerate

Usage:
    python scripts/quantize_w4a16.py --model google/medgemma-4b-it --out <dir>/medgemma-4b-it-quantized.w4a16
"""
import argparse
import json
import os
import platform
from datetime import datetime, timezone

# ── Sao chép nguyên văn từ RedHatAI/gemma-3-4b-it-quantized.w4a16 ──────────────
RECIPE_YAML = """\
quant_stage:
  quant_modifiers:
    GPTQModifier:
      ignore: ['re:.*lm_head.*', 're:.*embed_tokens.*', 're:vision_tower.*', 're:multi_modal_projector.*']
      sequential_targets: [Gemma3DecoderLayer]
      dampening_frac: 0.05
      config_groups:
        group0:
          targets: [Linear]
          weights: {num_bits: 4, type: int, strategy: group, group_size: 128, symmetric: true,
            actorder: weight, observer: mse}
"""
CALIB_DATASET = "neuralmagic/calibration"
CALIB_CONFIG = "LLM"
CALIB_SPLIT = "train[:512]"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048
SHUFFLE_SEED = 42
REFERENCE = "RedHatAI/gemma-3-4b-it-quantized.w4a16@9288be4"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="google/medgemma-4b-it (hoặc google/gemma-3-4b-it để kiểm chứng)")
    ap.add_argument("--out", required=True, help="thư mục xuất (sẽ tạo mới)")
    args = ap.parse_args()

    if os.path.exists(args.out) and os.listdir(args.out):
        raise SystemExit(f"{args.out} đã tồn tại và không rỗng — xoá hoặc chọn thư mục khác")
    os.makedirs(args.out, exist_ok=True)

    import torch
    import transformers
    from datasets import load_dataset
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration
    import llmcompressor
    try:
        from llmcompressor import oneshot
    except ImportError:                      # llmcompressor < 0.5
        from llmcompressor.transformers import oneshot

    print(f"[quant] nạp {args.model}")
    model = Gemma3ForConditionalGeneration.from_pretrained(args.model, device_map="auto", torch_dtype="auto")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    tokenizer = processor.tokenizer

    print(f"[quant] calibration: {CALIB_DATASET}/{CALIB_CONFIG} {CALIB_SPLIT}, shuffle seed {SHUFFLE_SEED}")
    ds = load_dataset(CALIB_DATASET, CALIB_CONFIG, split=CALIB_SPLIT).shuffle(seed=SHUFFLE_SEED)
    # Cột "text" đã có sẵn prompt dạng chữ. Tokenize tường minh (llm-compressor bản cũ tự làm việc
    # này bên trong; làm tay để không phụ thuộc phiên bản). Mặc định tokenizer thêm <bos>.
    ds = ds.map(lambda ex: tokenizer(ex["text"], padding=False, truncation=True, max_length=MAX_SEQUENCE_LENGTH),
                remove_columns=ds.column_names)

    recipe_path = os.path.join(args.out, "recipe_input.yaml")
    with open(recipe_path, "w", encoding="utf-8") as f:
        f.write(RECIPE_YAML)

    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe_path,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    )

    print(f"[quant] lưu {args.out}")
    model.save_pretrained(args.out, save_compressed=True)
    processor.save_pretrained(args.out)

    meta = {
        "source_model": args.model,
        "replicates": REFERENCE,
        "scheme": "W4A16 GPTQ, int4 weight, group 128, symmetric, actorder=weight, observer=mse, dampening 0.05",
        "kept_unquantized": ["lm_head", "embed_tokens", "vision_tower", "multi_modal_projector"],
        "calibration": {"dataset": f"{CALIB_DATASET}/{CALIB_CONFIG}", "split": CALIB_SPLIT,
                        "num_samples": NUM_CALIBRATION_SAMPLES, "max_seq_length": MAX_SEQUENCE_LENGTH,
                        "shuffle_seed": SHUFFLE_SEED},
        "versions": {"llmcompressor": llmcompressor.__version__, "torch": torch.__version__,
                     "transformers": transformers.__version__, "python": platform.python_version()},
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(os.path.join(args.out, "vimed_quantization.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
