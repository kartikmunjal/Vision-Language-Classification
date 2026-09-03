#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from vision_language_classification.labelers import llm_prompt, parse_structured_labels
from vision_language_classification.schema import read_jsonl


def existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {row["example_id"] for row in read_jsonl(path)}


def extract_json(text: str) -> str:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("model response contains no JSON object")
    return text[start : end + 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("output")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if args.device.startswith("cuda") else torch.float32,
    ).to(args.device).eval()
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    done = existing_ids(target)
    failures = 0
    pending = [row for row in read_jsonl(args.manifest) if row["example_id"] not in done]
    with target.open("a", encoding="utf-8") as handle:
        for offset in range(0, len(pending), args.batch_size):
            examples = pending[offset : offset + args.batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": llm_prompt(example["caption"])}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for example in examples
            ]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(args.device)
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            responses = tokenizer.batch_decode(
                generated[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
            )
            for example, response in zip(examples, responses, strict=True):
                try:
                    labels = parse_structured_labels(extract_json(response))
                    row = {
                        "example_id": example["example_id"],
                        "source": "llm",
                        "model_id": args.model,
                        "decoding": "greedy",
                        "integer_normalization": "positive_to_one",
                        "labels": labels,
                    }
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                    handle.flush()
                except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
                    failures += 1
                    print(f"invalid response for {example['example_id']}: {exc}", flush=True)
                    print(response, flush=True)
            processed = min(offset + args.batch_size, len(pending))
            if processed % 100 < args.batch_size:
                print(f"processed {processed}/{len(pending)}; failures={failures}", flush=True)
    if failures:
        raise SystemExit(f"{failures} examples failed strict parsing; rerun after inspection")


if __name__ == "__main__":
    main()
