#!/usr/bin/env python3
"""
E1 LocateAnything offline inference.

Run with:
  CUDA_VISIBLE_DEVICES=2 /home/medteam/Zhrch/.venvs/locany311/bin/python scripts/e1_locate_infer_locany.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CFG = REPO_ROOT / "configs" / "1232_final_v8_2_mged_final_gpu6.yaml"
os.environ.setdefault("CFG_FILE", str(DEFAULT_CFG))
sys.path.insert(0, str(REPO_ROOT / "Eagle" / "Embodied"))


BOX_RE = re.compile(
    r"<ref>([^<]+)</ref>\s*<box>\s*"
    r"<\s*([0-9]+(?:\.[0-9]+)?)\s*>\s*"
    r"<\s*([0-9]+(?:\.[0-9]+)?)\s*>\s*"
    r"<\s*([0-9]+(?:\.[0-9]+)?)\s*>\s*"
    r"<\s*([0-9]+(?:\.[0-9]+)?)\s*>\s*</box>"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default=str(REPO_ROOT / "Eagle" / "Embodied" / "work_dirs" / "1232_final_locany_full_more10000" / "checkpoint-3000"),
    )
    parser.add_argument("--data-root", default="/home/medteam/Zhrch/Datasets/1232_final")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--jsonl",
        default=str(REPO_ROOT / "Eagle" / "Embodied" / "locany_recipe" / "1232_final" / "test.jsonl"),
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "data" / "eagle_teacher" / "1232_final_test_locateanything_ckpt3000.json"),
    )
    parser.add_argument("--generation-mode", default="hybrid", choices=["fast", "slow", "hybrid"])
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0, help="0 means full split")
    parser.add_argument("--resume", action="store_true", help="reuse existing records in --output")
    parser.add_argument(
        "--from-predictions-jsonl",
        default="",
        help="Normalize an existing Eagle/Embodied predictions.jsonl into the teacher cache without loading the model.",
    )
    return parser.parse_args()


def class_name(class_id: int) -> str:
    return f"class_{class_id:02d}"


def load_jsonl_entries(jsonl_path: Path, data_root: Path, split: str) -> list[dict]:
    if jsonl_path.exists():
        entries = []
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
        return entries

    split_dir = data_root / split
    entries = []
    mask_re = re.compile(r"^(?P<stem>.+)_mask_(?P<class_id>\d+)\.png$")
    for image_path in sorted(split_dir.glob("*_image.png"), key=lambda p: p.name):
        stem = image_path.name[: -len("_image.png")]
        class_ids = set()
        for mask_path in split_dir.glob(f"{stem}_mask_*.png"):
            match = mask_re.match(mask_path.name)
            if match:
                class_ids.add(int(match.group("class_id")))
        if not class_ids:
            continue
        categories = [class_name(cid) for cid in sorted(class_ids)]
        question = "Locate all the instances that matches the following description: " + "</c>".join(categories) + "."
        entries.append({"image": str(Path(split) / image_path.name), "conversations": [{"from": "human", "value": question}]})
    return entries


def extract_question(entry: dict) -> str:
    conversations = entry.get("conversations") or []
    if conversations and isinstance(conversations[0], dict):
        return str(conversations[0].get("value") or "")
    question = str(entry.get("question") or "").strip()
    if question:
        return question
    raise ValueError(f"Entry has no LocateAnything question: {entry}")


def label_to_id(label: str) -> int | None:
    match = re.search(r"(\d+)", label or "")
    if not match:
        return None
    return int(match.group(1))


def parse_boxes(text: str, image_w: int, image_h: int) -> list[dict]:
    instances = []
    for match in BOX_RE.finditer(text or ""):
        label = match.group(1).strip()
        coords = [float(match.group(i)) for i in range(2, 6)]
        x1, y1, x2, y2 = coords
        x1, x2 = sorted((max(0.0, min(1000.0, x1)), max(0.0, min(1000.0, x2))))
        y1, y2 = sorted((max(0.0, min(1000.0, y1)), max(0.0, min(1000.0, y2))))
        if x2 <= x1 or y2 <= y1:
            continue
        box = [
            x1 / 1000.0 * image_w,
            y1 / 1000.0 * image_h,
            x2 / 1000.0 * image_w,
            y2 / 1000.0 * image_h,
        ]
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        label_id = label_to_id(label)
        instances.append(
            {
                "label": label,
                "label_id": label_id,
                "cls_id": label_id,
                "category_id": label_id,
                "score": 1.0,
                "confidence": 1.0,
                "bbox": box,
                "extreme_points": [[cx, box[1]], [box[0], cy], [cx, box[3]], [box[2], cy]],
                "source": "LocateAnything",
            }
        )
    return instances


def box_to_instance(label: str, box_1000: list[float], image_w: int, image_h: int) -> dict | None:
    x1, y1, x2, y2 = [float(v) for v in box_1000[:4]]
    x1, x2 = sorted((max(0.0, min(1000.0, x1)), max(0.0, min(1000.0, x2))))
    y1, y2 = sorted((max(0.0, min(1000.0, y1)), max(0.0, min(1000.0, y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    box = [
        x1 / 1000.0 * image_w,
        y1 / 1000.0 * image_h,
        x2 / 1000.0 * image_w,
        y2 / 1000.0 * image_h,
    ]
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    label_id = label_to_id(label)
    return {
        "label": label,
        "label_id": label_id,
        "cls_id": label_id,
        "category_id": label_id,
        "score": 1.0,
        "confidence": 1.0,
        "bbox": box,
        "extreme_points": [[cx, box[1]], [box[0], cy], [cx, box[3]], [box[2], cy]],
        "source": "LocateAnything",
    }


def normalize_existing_predictions(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred_path = Path(args.from_predictions_jsonl).resolve()
    samples = []

    with pred_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            image_rel = str(row["image"])
            image_path = data_root / image_rel
            image = Image.open(image_path).convert("RGB")
            image_w, image_h = image.size
            instances = []
            for label, boxes in sorted((row.get("pred") or {}).items()):
                for box in boxes:
                    inst = box_to_instance(str(label), box, image_w, image_h)
                    if inst is not None:
                        instances.append(inst)
            samples.append(
                {
                    "id": Path(image_rel).stem.replace("_image", ""),
                    "img_path": str(image_path),
                    "image_path": str(image_path),
                    "image_rel": image_rel,
                    "width": image_w,
                    "height": image_h,
                    "question": str(row.get("question") or ""),
                    "raw_response": str(row.get("raw_response") or ""),
                    "instances": instances,
                }
            )
            if args.limit and args.limit > 0 and len(samples) >= args.limit:
                break

    payload = {
        "format": "diffusionsnake_eagle_teacher_v1",
        "task": "E1 LocateAnything detection cache",
        "model_path": str(Path(args.model_path).resolve()),
        "data_root": str(data_root),
        "split": args.split,
        "coordinate_space": "original_image_pixels",
        "score_policy": "LocateAnything output has no calibrated confidence; score=confidence=1.0 for every parsed box.",
        "source_predictions_jsonl": str(pred_path),
        "samples": samples,
        "summary": summarize(samples),
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))
    print(f"Saved LocateAnything cache: {output_path}")


def apply_chat_template(processor, messages: list[dict]) -> str:
    if hasattr(processor, "apply_chat_template"):
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if hasattr(processor, "py_apply_chat_template"):
        return processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    raise AttributeError("Processor does not provide a chat template API")


def process_vision_info(processor, messages: list[dict]):
    if hasattr(processor, "process_vision_info"):
        return processor.process_vision_info(messages)
    raise AttributeError("Processor does not provide process_vision_info")


def prepare_inputs(processor_inputs, device: str) -> dict:
    prepared = {}
    for key in ("input_ids", "attention_mask", "pixel_values", "image_grid_hws"):
        val = processor_inputs.get(key)
        prepared[key] = val.to(device) if torch.is_tensor(val) else val
    return prepared


def decode_output(raw_output, input_ids, processor) -> str:
    if isinstance(raw_output, tuple):
        raw_output = raw_output[0]
    if isinstance(raw_output, str):
        return raw_output
    if isinstance(raw_output, list) and raw_output and isinstance(raw_output[0], str):
        return raw_output[0]
    if torch.is_tensor(raw_output):
        generated_ids = raw_output
        if generated_ids.ndim == 2 and generated_ids.shape[1] >= input_ids.shape[1]:
            generated_ids = generated_ids[:, input_ids.shape[1] :]
        generated_ids = generated_ids.detach().cpu()
        if hasattr(processor, "post_process_image_text_to_text"):
            decoded = processor.post_process_image_text_to_text(
                generated_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            return decoded[0] if isinstance(decoded, list) else str(decoded)
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None and hasattr(processor, "batch_decode"):
            tokenizer = processor
        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        return decoded[0] if isinstance(decoded, list) else str(decoded)
    return str(raw_output)


@torch.inference_mode()
def generate(model, processor, image: Image.Image, question: str, args: argparse.Namespace) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]
    text = apply_chat_template(processor, messages)
    images, videos = process_vision_info(processor, messages)
    processor_inputs = processor(text=[text], images=images, videos=videos, return_tensors="pt", padding=True)
    prepared = prepare_inputs(processor_inputs, args.device)
    tokenizer = getattr(processor, "tokenizer", None)
    kwargs = {
        "pixel_values": prepared["pixel_values"],
        "input_ids": prepared["input_ids"],
        "attention_mask": prepared["attention_mask"],
        "image_grid_hws": prepared["image_grid_hws"],
        "tokenizer": tokenizer,
        "max_new_tokens": args.max_new_tokens,
        "use_cache": True,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "generation_mode": args.generation_mode,
    }
    if args.generation_mode in ("fast", "hybrid"):
        kwargs["n_future_tokens"] = 6
    if tokenizer is not None and getattr(tokenizer, "eos_token_id", None) is not None:
        kwargs["eos_token_id"] = tokenizer.eos_token_id
    raw_output = model.generate(**kwargs)
    return decode_output(raw_output, prepared["input_ids"], processor)


def load_existing(output_path: Path) -> dict[str, dict]:
    if not output_path.exists():
        return {}
    with output_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    records = obj.get("samples", []) if isinstance(obj, dict) else []
    return {str(rec.get("image_rel") or rec.get("img_path")): rec for rec in records if isinstance(rec, dict)}


def summarize(samples: list[dict]) -> dict:
    per_class = defaultdict(int)
    num_instances = 0
    for rec in samples:
        instances = rec.get("instances") or []
        num_instances += len(instances)
        for inst in instances:
            label = inst.get("label") or f"class_{int(inst.get('label_id') or 0):02d}"
            per_class[str(label)] += 1
    return {
        "samples": len(samples),
        "instances": num_instances,
        "per_class_instances": dict(sorted(per_class.items())),
    }


def main() -> None:
    args = parse_args()
    if args.from_predictions_jsonl:
        normalize_existing_predictions(args)
        return

    data_root = Path(args.data_root).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    entries = load_jsonl_entries(Path(args.jsonl), data_root, args.split)
    if args.limit and args.limit > 0:
        entries = entries[: args.limit]

    existing = load_existing(output_path) if args.resume else {}
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, use_fast=True)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
    model = model.to(args.device).eval()

    samples = []
    for entry in tqdm(entries, desc="LocateAnything E1 inference"):
        image_rel = str(entry["image"])
        if image_rel in existing:
            samples.append(existing[image_rel])
            continue
        image_path = data_root / image_rel
        question = extract_question(entry)
        image = Image.open(image_path).convert("RGB")
        image_w, image_h = image.size
        answer = generate(model, processor, image, question, args)
        instances = parse_boxes(answer, image_w, image_h)
        samples.append(
            {
                "id": Path(image_rel).stem.replace("_image", ""),
                "img_path": str(image_path),
                "image_path": str(image_path),
                "image_rel": image_rel,
                "width": image_w,
                "height": image_h,
                "question": question,
                "raw_response": answer,
                "instances": instances,
            }
        )
        payload = {
            "format": "diffusionsnake_eagle_teacher_v1",
            "task": "E1 LocateAnything detection cache",
            "model_path": str(Path(args.model_path).resolve()),
            "data_root": str(data_root),
            "split": args.split,
            "coordinate_space": "original_image_pixels",
            "score_policy": "LocateAnything output has no calibrated confidence; score=confidence=1.0 for every parsed box.",
            "samples": samples,
            "summary": summarize(samples),
        }
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write("\n")

    payload = {
        "format": "diffusionsnake_eagle_teacher_v1",
        "task": "E1 LocateAnything detection cache",
        "model_path": str(Path(args.model_path).resolve()),
        "data_root": str(data_root),
        "split": args.split,
        "coordinate_space": "original_image_pixels",
        "score_policy": "LocateAnything output has no calibrated confidence; score=confidence=1.0 for every parsed box.",
        "samples": samples,
        "summary": summarize(samples),
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))
    print(f"Saved LocateAnything cache: {output_path}")


if __name__ == "__main__":
    main()
