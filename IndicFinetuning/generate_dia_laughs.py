import argparse
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf


DEFAULT_MODEL = "nari-labs/Dia-1.6B-0626"
DEFAULT_OUTPUT = "./IndicFinetuning/datasets/LaughterCompositeTest/dia_laugh_raw"
DEFAULT_SAMPLE_RATE = 44100


LAUGH_PROMPTS = [
    "[S1] (laughs)",
    "[S1] (chuckle)",
    "[S1] Oh wow. (laughs) I did not expect that.",
    "[S1] Wait, seriously? (laughs) That is too funny.",
    "[S1] I tried to stay serious. (laughs) But I just could not.",
    "[S1] That was unexpected. (chuckle) Okay, give me a second.",
    "[S1] No way. (laughs) That actually happened?",
    "[S1] I know I should not laugh. (laughs) But that was perfect.",
    "[S1] Hold on. (laughs) Let me breathe for a second.",
    "[S1] That timing was ridiculous. (laughs) I cannot believe it.",
]


def import_dia():
    try:
        from dia.model import Dia
    except ImportError as exc:
        raise ImportError(
            "Could not import Dia. Install or clone Dia on the VM first, then run this script again. "
            "Expected import: from dia.model import Dia"
        ) from exc
    return Dia


def as_numpy(audio):
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    audio = np.asarray(audio)
    if audio.ndim > 1:
        audio = np.squeeze(audio)
    return audio.astype(np.float32)


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_prompt(index: int, prompt_file_rows):
    if prompt_file_rows:
        row = prompt_file_rows[index % len(prompt_file_rows)]
        return row["prompt"], row.get("label", f"custom_{index + 1:03d}")
    return LAUGH_PROMPTS[index % len(LAUGH_PROMPTS)], f"laugh_{index + 1:03d}"


def load_prompt_file(path: str | None):
    if not path:
        return []
    rows = []
    prompt_path = Path(path)
    for line_no, line in enumerate(prompt_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        if "prompt" not in item:
            raise ValueError(f"{prompt_path}:{line_no} missing required key 'prompt'")
        rows.append(item)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Generate raw Dia laugh clips for the Malayalam laughter overfit test.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--prompt-file", default=None, help="Optional JSONL with {'label': str, 'prompt': str} rows.")
    parser.add_argument("--clone-from-audio", default=None, help="Optional reference audio for Dia voice cloning.")
    parser.add_argument("--clone-from-text", default=None, help="Transcript matching --clone-from-audio, in Dia [S1]/[S2] format.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--compute-dtype", default="float16")
    parser.add_argument("--use-torch-compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--temperature", type=float, default=1.8)
    parser.add_argument("--top-p", type=float, default=0.90)
    parser.add_argument("--cfg-filter-top-k", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=1200)
    args = parser.parse_args()

    if args.count < 1:
        raise ValueError("--count must be at least 1")
    if bool(args.clone_from_audio) != bool(args.clone_from_text):
        raise ValueError("--clone-from-audio and --clone-from-text must be provided together.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        import torch

        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    except ImportError:
        pass

    Dia = import_dia()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_rows = load_prompt_file(args.prompt_file)

    print(f"Loading Dia model: {args.model}")
    model = Dia.from_pretrained(args.model, compute_dtype=args.compute_dtype)

    manifest_rows = []
    for index in range(args.count):
        prompt, label = make_prompt(index, prompt_rows)
        generation_text = prompt
        if args.clone_from_text:
            generation_text = args.clone_from_text + prompt

        clip_id = f"dia_{index + 1:04d}_{label}"
        output_path = output_dir / f"{clip_id}.wav"
        print(f"[{index + 1}/{args.count}] {clip_id}: {prompt}")

        generate_kwargs = {
            "use_torch_compile": args.use_torch_compile,
            "verbose": True,
            "cfg_scale": args.cfg_scale,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "cfg_filter_top_k": args.cfg_filter_top_k,
            "max_tokens": args.max_tokens,
        }
        if args.clone_from_audio:
            generate_kwargs["audio_prompt"] = args.clone_from_audio

        audio = model.generate(generation_text, **generate_kwargs)
        sf.write(output_path, as_numpy(audio), DEFAULT_SAMPLE_RATE)

        manifest_rows.append(
            {
                "id": clip_id,
                "path": str(output_path),
                "prompt": prompt,
                "generation_text": generation_text,
                "model": args.model,
                "sample_rate": DEFAULT_SAMPLE_RATE,
                "clone_from_audio": args.clone_from_audio,
                "clone_from_text": args.clone_from_text,
                "cfg_scale": args.cfg_scale,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "cfg_filter_top_k": args.cfg_filter_top_k,
                "max_tokens": args.max_tokens,
                "seed": args.seed,
            }
        )

    manifest_path = output_dir / "manifest.jsonl"
    write_jsonl(manifest_path, manifest_rows)
    print(f"Done. Wrote {len(manifest_rows)} clips to: {output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
