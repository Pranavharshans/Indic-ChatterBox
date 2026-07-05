import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


DEFAULT_MODEL = "k2-fsa/OmniVoice"
DEFAULT_OUTPUT = "./IndicFinetuning/datasets/LaughterCompositeTest/omni_malayalam_speech"
DEFAULT_REF_AUDIO = "./Maya.wav"
DEFAULT_REF_TEXT = "i went to the store to buy some fresh fruits and snacks for the evening"
SAMPLE_RATE = 24000

TIME_PHRASES = [
    "ഇന്ന് രാവിലെ",
    "വൈകുന്നേരം വീട്ടിലേക്ക് വരുമ്പോൾ",
    "ഉച്ചയ്ക്ക് ജോലി കഴിഞ്ഞപ്പോൾ",
    "രാത്രി ഭക്ഷണത്തിന് മുമ്പ്",
    "മഴ നിർത്തിയ ശേഷം",
    "ബസ് കാത്തുനിൽക്കുമ്പോൾ",
    "ചായ കുടിക്കാൻ ഇരുന്നപ്പോൾ",
    "ഓഫീസിൽ നിന്ന് ഇറങ്ങുമ്പോൾ",
    "കട അടയ്ക്കുന്നതിന് മുമ്പ്",
    "ഫോൺ വിളി കഴിഞ്ഞപ്പോൾ",
    "പാർക്കിൽ നടക്കുമ്പോൾ",
    "സ്റ്റേഷനിൽ കാത്തുനിൽക്കുമ്പോൾ",
    "വീട്ടിലെ പണി തീർന്നപ്പോൾ",
    "ക്ലാസ് കഴിഞ്ഞ് പുറത്തുവന്നപ്പോൾ",
    "ബാങ്കിൽ ടോക്കൺ എടുത്ത ശേഷം",
    "ജിം കഴിഞ്ഞ് വിശ്രമിക്കുമ്പോൾ",
    "പുസ്തകം തിരികെ നൽകാൻ പോയപ്പോൾ",
    "മാർക്കറ്റിൽ സാധനം വാങ്ങുമ്പോൾ",
    "അടുക്കളയിൽ ചായ വെക്കുമ്പോൾ",
    "കൂട്ടുകാരോട് സംസാരിക്കുമ്പോൾ",
]

SCENES = [
    "കടയിലെ പഴങ്ങളുടെ മണം മനസ്സിന് നല്ല ആശ്വാസം കൊടുത്തു",
    "ചെറിയ മഴത്തുള്ളികൾ റോഡിലെ പൊടിയെ അടക്കി",
    "മുന്നിലെ കുട്ടി ബാഗ് തുറന്ന് പേന തിരഞ്ഞുകൊണ്ടിരുന്നു",
    "ഓട്ടോ ഡ്രൈവർ പഴയ സിനിമകളെക്കുറിച്ച് സംസാരിക്കാൻ തുടങ്ങി",
    "അമ്മ അടുക്കളയിൽ നിന്ന് എന്റെ പേര് വിളിച്ചു",
    "ഫോണിൽ വന്ന സന്ദേശം ആദ്യം കുറച്ച് ആശങ്കയുണ്ടാക്കി",
    "വഴിയരികിലെ ചായക്കടയിൽ ആളുകൾ പതുക്കെ സംസാരിക്കുകയായിരുന്നു",
    "പുതിയ ഷൂ കെട്ടിയിട്ടും നടക്കാൻ അല്പം ബുദ്ധിമുട്ടായി",
    "അയൽവീട്ടിലെ കുഞ്ഞ് വീണ്ടും അതേ പാട്ട് പാടിക്കൊണ്ടിരുന്നു",
    "മീറ്റിംഗിൽ എല്ലാവരും ഒരേ സമയം സംസാരിക്കാൻ തുടങ്ങി",
    "ബേക്കറിയിലെ അവസാന കേക്ക് ആരോ വാങ്ങിപ്പോയി",
    "ബാങ്കിലെ ക്യൂ പതുക്കെ മാത്രമേ മുന്നോട്ട് നീങ്ങിയുള്ളൂ",
    "മഴ കാരണം ബസിന്റെ ജനൽ മുഴുവൻ മങ്ങിപ്പോയി",
    "പഴയ സുഹൃത്ത് അപ്രതീക്ഷിതമായി മുന്നിൽ വന്ന് നിന്നു",
    "ലിഫ്റ്റ് പ്രവർത്തിക്കാത്തതിനാൽ പടികൾ കയറേണ്ടി വന്നു",
    "പാർക്കിലെ ബെഞ്ചിൽ ഒരു പുസ്തകം മറന്നുവച്ചിരുന്നു",
    "ക്ലാസിലെ പഴയ ബെൽ ശബ്ദം വീണ്ടും ഓർമ്മ വന്നു",
    "വീട്ടിലെ വൈദ്യുതി പോയപ്പോൾ എല്ലാവരും പുറത്തേക്ക് വന്നു",
    "കടക്കാരൻ പണം മാറ്റി നൽകാൻ കുറച്ച് സമയം എടുത്തു",
    "അച്ഛൻ പഴയ കഥ പറഞ്ഞുകൊണ്ട് ചെടികൾക്ക് വെള്ളം ഒഴിച്ചു",
]

REACTIONS = [
    "അത് കണ്ടപ്പോൾ ദിവസം അത്ര മോശമല്ലെന്ന് തോന്നി",
    "അതിനാൽ ഞാൻ കുറച്ച് നേരം അവിടെ തന്നെ നിന്നു",
    "അവസാനം കാര്യം വിചാരിച്ചതിനെക്കാൾ എളുപ്പത്തിൽ തീർന്നു",
    "പിന്നെ അതിനെക്കുറിച്ച് അധികം ചിന്തിക്കേണ്ടി വന്നില്ല",
    "അപ്പോൾ ചെറിയ കാര്യങ്ങൾ പോലും മനസ്സിനെ മാറ്റുമെന്ന് മനസ്സിലായി",
    "അതുകൊണ്ട് വീട്ടിലെത്തുമ്പോൾ മുഖത്ത് ഒരു സമാധാനം ഉണ്ടായിരുന്നു",
    "പിന്നീട് ആ ചെറിയ സംഭവമാണ് മുഴുവൻ ദിവസവും ഓർമ്മയിൽ നിന്നത്",
    "അതിൽ ആരും വലിയ പ്രശ്നമൊന്നും കണ്ടില്ല",
    "എങ്കിലും എനിക്ക് അത് കേൾക്കാൻ നല്ല രസമായി തോന്നി",
    "കുറച്ച് കഴിഞ്ഞപ്പോൾ എല്ലാവരും പഴയ പോലെ ശാന്തമായി",
    "അതിനു ശേഷം ചായയുടെ രുചി പോലും കുറച്ച് നല്ലതായി തോന്നി",
    "ഞാൻ കരുതിയതിലുമധികം ആളുകൾ സഹായിക്കാൻ തയ്യാറായിരുന്നു",
    "പിന്നെ സമയം പോയത് തന്നെ അറിയാതെ പോയി",
    "അവസാനം എല്ലാവരും അതിനെ ഒരു സാധാരണ കാര്യമാക്കി വിട്ടു",
    "ആ നിമിഷം മനസ്സിൽ ചെറുതായി സന്തോഷം നിറഞ്ഞു",
    "ഇതുപോലുള്ള ദിവസങ്ങളാണ് പിന്നീട് കൂടുതലായി ഓർമ്മ വരുന്നത്",
    "പിന്നെ ഞാൻ പതുക്കെ എന്റെ വഴിയിലേക്ക് തിരിഞ്ഞു",
    "അത്ര മാത്രം സംഭവമായിരുന്നെങ്കിലും മനസ്സിൽ ഒരു ചൂട് ബാക്കി നിന്നു",
    "അതുകൊണ്ട് ബാക്കി ജോലി ചെയ്യാൻ കുറച്ച് എളുപ്പമായി",
    "അവസാനം ആ ചെറിയ ഇടവേള തന്നെ ദിവസത്തെ നല്ല ഭാഗമാക്കി",
]

CONNECTORS = [
    "എനിക്ക് ആദ്യം കാര്യം സാധാരണ പോലെ തോന്നിയെങ്കിലും",
    "ആരും അതിനെ വലിയ സംഭവമായി എടുത്തില്ലെങ്കിലും",
    "ഞാൻ ഉടനെ ഒന്നും പറയാതെ നിന്നെങ്കിലും",
    "കുറച്ച് നേരം അതിന്റെ അർത്ഥം പിടിക്കാനായില്ലെങ്കിലും",
    "അവിടെ ഉണ്ടായിരുന്നവർ തിരക്കിലായിരുന്നെങ്കിലും",
    "മനസ്സിൽ ചെറിയ ക്ഷീണം ഉണ്ടായിരുന്നെങ്കിലും",
    "പുറത്ത് ശബ്ദം കുറച്ച് കൂടുതലായിരുന്നെങ്കിലും",
    "സമയം അത്ര കൂടുതലില്ലായിരുന്നെങ്കിലും",
    "പ്ലാൻ ചെയ്തതുപോലെ ഒന്നും നടന്നില്ലെങ്കിലും",
    "അവസാന നിമിഷം മാറ്റം വന്നിരുന്നെങ്കിലും",
]


def build_texts(count: int) -> list[dict]:
    rows = []
    seen = set()
    index = 1
    for time_phrase in TIME_PHRASES:
        for scene in SCENES:
            connector = CONNECTORS[(index - 1) % len(CONNECTORS)]
            reaction = REACTIONS[(index - 1) % len(REACTIONS)]
            text = f"{time_phrase} {scene}. {connector}, {reaction}."
            if text in seen:
                continue
            seen.add(text)
            rows.append(
                {
                    "id": f"omni_ml_{index:04d}",
                    "text": text,
                    "language_id": "ml",
                }
            )
            index += 1
            if len(rows) >= count:
                return rows
    raise RuntimeError(f"Could only build {len(rows)} unique texts, requested {count}.")


def load_text_file(path: str, count: int) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            item = json.loads(line)
            text = item["text"]
            row_id = item.get("id", f"omni_ml_{len(rows) + 1:04d}")
        else:
            text = line
            row_id = f"omni_ml_{len(rows) + 1:04d}"
        rows.append({"id": row_id, "text": text, "language_id": "ml"})
        if len(rows) >= count:
            break
    if len(rows) < count:
        raise ValueError(f"Text file only provided {len(rows)} rows, requested {count}.")
    return rows


def as_float32(audio) -> np.ndarray:
    audio = np.asarray(audio)
    if audio.ndim > 1:
        audio = np.squeeze(audio)
    return audio.astype(np.float32)


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate Malayalam speech chunks with OmniVoice voice cloning.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--count", type=int, default=400)
    parser.add_argument("--ref-audio", default=DEFAULT_REF_AUDIO)
    parser.add_argument("--ref-text", default=DEFAULT_REF_TEXT)
    parser.add_argument("--text-file", default=None, help="Optional TXT or JSONL with one Malayalam text per line.")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()

    if args.count < 1:
        raise ValueError("--count must be at least 1")
    ref_audio = Path(args.ref_audio)
    if not ref_audio.exists():
        raise FileNotFoundError(f"Reference audio not found: {ref_audio}")

    try:
        from omnivoice import OmniVoice
    except ImportError as exc:
        raise ImportError("Could not import OmniVoice. Install the OmniVoice package on the VM first.") from exc

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]

    output_dir = Path(args.output)
    wav_dir = output_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    rows = load_text_file(args.text_file, args.count) if args.text_file else build_texts(args.count)
    if len({row["text"] for row in rows}) != len(rows):
        raise ValueError("Generated duplicate Malayalam texts; refusing to continue.")

    print(f"Loading OmniVoice model: {args.model}")
    model = OmniVoice.from_pretrained(args.model, device_map=args.device_map, dtype=dtype)

    manifest_rows = []
    for position, row in enumerate(rows, start=1):
        wav_path = wav_dir / f"{row['id']}.wav"
        if args.skip_existing and wav_path.exists() and wav_path.stat().st_size > 0:
            print(f"[{position}/{len(rows)}] skip existing {row['id']}")
        else:
            print(f"[{position}/{len(rows)}] generating {row['id']}: {row['text']}")
            audio = model.generate(
                text=row["text"],
                ref_audio=str(ref_audio),
                ref_text=args.ref_text,
            )
            if not audio:
                raise RuntimeError(f"OmniVoice returned no audio for {row['id']}")
            sf.write(wav_path, as_float32(audio[0]), SAMPLE_RATE)
            if args.sleep:
                time.sleep(args.sleep)

        manifest_rows.append(
            {
                **row,
                "path": str(wav_path),
                "sample_rate": SAMPLE_RATE,
                "model": args.model,
                "ref_audio": str(ref_audio),
                "ref_text": args.ref_text,
            }
        )

    manifest_path = output_dir / "manifest.jsonl"
    write_jsonl(manifest_path, manifest_rows)
    (output_dir / "texts.jsonl").write_text(
        "\n".join(json.dumps({"id": row["id"], "text": row["text"], "language_id": "ml"}, ensure_ascii=False) for row in rows)
        + "\n",
        encoding="utf-8",
    )
    print(f"Done. WAVs: {wav_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
