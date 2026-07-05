import argparse
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from snac import SNAC
from transformers import AutoModelForCausalLM, AutoTokenizer


CODE_START_TOKEN_ID = 128257
CODE_END_TOKEN_ID = 128258
CODE_TOKEN_OFFSET = 128266
SNAC_MIN_ID = 128266
SNAC_MAX_ID = 156937
SNAC_TOKENS_PER_FRAME = 7

SOH_ID = 128259
EOH_ID = 128260
SOA_ID = 128261
TEXT_EOT_ID = 128009

DEFAULT_MODEL = "maya-research/maya1"
DEFAULT_SNAC_MODEL = "hubertsiuzdak/snac_24khz"
DEFAULT_OUTPUT = "./IndicFinetuning/datasets/LaughterCompositeTest/maya_laugh_raw"
DEFAULT_DESCRIPTION = (
    "Female, Indian, in her 20s, warm natural conversational voice, clear diction, "
    "friendly tone, realistic casual laugh, medium pitch, natural pacing."
)
ROUND2_DESCRIPTION = (
    "Female, in her 30s with an Indian English accent and is an energetic event host, "
    "bright warm timbre, clear diction, playful tone, naturally amused, expressive laugh, medium-high energy."
)
SUPPORTED_TAGS = ("<laugh>", "<giggle>", "<laugh_harder>")

LAUGH_TEXTS = [
    "I tried to stay serious, but <laugh> that answer was too funny to ignore.",
    "Wait, you actually did that <laugh> I was not ready for that story.",
    "I thought the plan was simple, then <laugh> everything changed in one minute.",
    "No, no, say that again <laugh> because I still cannot believe it.",
    "The way he walked into the room <laugh> made everyone lose their focus.",
    "I was about to explain it calmly, but <laugh> the timing was just perfect.",
    "She looked at the broken cup and said nothing <laugh> which made it even funnier.",
    "I know this is not the right moment <laugh> but that was honestly hilarious.",
    "He tried to act confident <laugh> but everyone knew he forgot the whole line.",
    "The phone rang at exactly the wrong time <laugh> and nobody could continue.",
    "I was holding back the whole time <laugh> but that last comment finished me.",
    "Please do not make that face again <laugh> I cannot keep a straight voice.",
    "The meeting was supposed to be formal, then <laugh> someone played the wrong audio.",
    "I opened the message expecting bad news <laugh> but it was just his usual drama.",
    "She said it so seriously <laugh> that it took me a second to understand the joke.",
    "I almost believed the excuse <laugh> until he forgot his own story.",
    "Everything was quiet for a moment <laugh> and then we all started laughing.",
    "I should probably apologize first <laugh> but that mistake was too perfect.",
    "He brought the wrong file again <laugh> and somehow blamed the printer.",
    "I was trying to be polite <laugh> but that joke came out of nowhere.",
]

ROUND2_TEXTS = [
    "Wow, the way the lights came on at that exact moment <laugh> I could not have planned it better.",
    "Everyone was waiting so seriously, and then the music started from the wrong speaker <laugh> that was perfect.",
    "I asked for one simple entrance, but he came in waving both hands <laugh> the whole room lost it.",
    "This celebration already feels magical, and honestly <laugh> I am having way too much fun up here.",
    "The cake almost fell, the camera missed it, and somehow <laugh> that made the moment even better.",
    "I tried to keep the announcement formal, but your reaction <laugh> made that completely impossible.",
    "She said she was not nervous, then walked on stage backwards <laugh> and still made it look graceful.",
    "The countdown reached three, someone shouted surprise too early, and <laugh> now everyone knows.",
    "I have hosted many events, but this entrance <laugh> is definitely going into my favorite memories.",
    "The microphone worked only after I complimented it <laugh> so apparently even the sound system likes praise.",
    "I should not giggle during the opening, but <giggle> that little dance move was honestly adorable.",
    "You all look so proud right now, and <giggle> I can see three people trying not to cry already.",
    "He practiced that serious face all morning, but <giggle> it disappeared the second he saw the crowd.",
    "This surprise was supposed to be secret, but <giggle> half the family has been whispering about it all day.",
    "I promised I would stay professional, then she made that tiny victory pose <giggle> and I lost focus.",
    "The little one just clapped before the song even started <giggle> and somehow that was the cutest cue.",
    "I love how everyone pretended to be calm, while <giggle> the front row was clearly panicking.",
    "That was such a sweet answer, and <giggle> I can tell you rehearsed it in the mirror.",
    "The decorations are beautiful, the crowd is ready, and <giggle> someone just photobombed the camera.",
    "I was about to move to the next segment, but <giggle> that smile in the back row distracted me.",
    "No, wait, he brought the giant ribbon scissors again <laugh_harder> I cannot believe this is happening twice.",
    "The confetti cannon fired at the wrong person <laugh_harder> and somehow she still bowed like a champion.",
    "I said make some noise, not start a full dance battle <laugh_harder> but honestly I respect the energy.",
    "He tried to open the gift calmly, then the box made that sound <laugh_harder> and the whole stage stopped.",
    "The mascot missed the step, recovered with a spin, and <laugh_harder> now it looks like part of the show.",
    "I have no idea who planned that entrance <laugh_harder> but please give them a raise immediately.",
    "The serious award music started playing for the snack table <laugh_harder> and now I cannot unhear it.",
    "She waved to the wrong camera for ten full seconds <laugh_harder> but the confidence was incredible.",
    "The banner opened upside down, everyone cheered anyway, and <laugh_harder> that is why I love this crowd.",
    "I was ready for applause, but not for that dramatic pose <laugh_harder> this event is officially unforgettable.",
]


def build_prompt(tokenizer, description: str, text: str) -> str:
    soh_token = tokenizer.decode([SOH_ID])
    eoh_token = tokenizer.decode([EOH_ID])
    soa_token = tokenizer.decode([SOA_ID])
    sos_token = tokenizer.decode([CODE_START_TOKEN_ID])
    eot_token = tokenizer.decode([TEXT_EOT_ID])
    bos_token = tokenizer.bos_token
    formatted_text = f'<description="{description}"> {text}'
    return soh_token + bos_token + formatted_text + eot_token + eoh_token + soa_token + sos_token


def extract_snac_codes(token_ids: list[int]) -> list[int]:
    try:
        eos_idx = token_ids.index(CODE_END_TOKEN_ID)
    except ValueError:
        eos_idx = len(token_ids)
    return [token_id for token_id in token_ids[:eos_idx] if SNAC_MIN_ID <= token_id <= SNAC_MAX_ID]


def unpack_snac_from_7(snac_tokens: list[int]) -> list[list[int]]:
    frames = len(snac_tokens) // SNAC_TOKENS_PER_FRAME
    snac_tokens = snac_tokens[: frames * SNAC_TOKENS_PER_FRAME]
    if frames == 0:
        return [[], [], []]

    l1, l2, l3 = [], [], []
    for index in range(frames):
        slots = snac_tokens[index * 7 : (index + 1) * 7]
        l1.append((slots[0] - CODE_TOKEN_OFFSET) % 4096)
        l2.extend(
            [
                (slots[1] - CODE_TOKEN_OFFSET) % 4096,
                (slots[4] - CODE_TOKEN_OFFSET) % 4096,
            ]
        )
        l3.extend(
            [
                (slots[2] - CODE_TOKEN_OFFSET) % 4096,
                (slots[3] - CODE_TOKEN_OFFSET) % 4096,
                (slots[5] - CODE_TOKEN_OFFSET) % 4096,
                (slots[6] - CODE_TOKEN_OFFSET) % 4096,
            ]
        )
    return [l1, l2, l3]


def decode_audio(snac_model, snac_tokens: list[int], device: str) -> np.ndarray:
    levels = unpack_snac_from_7(snac_tokens)
    if not levels[0]:
        raise RuntimeError("No complete SNAC frames were generated.")
    codes_tensor = [torch.tensor(level, dtype=torch.long, device=device).unsqueeze(0) for level in levels]
    with torch.inference_mode():
        z_q = snac_model.quantizer.from_codes(codes_tensor)
        audio = snac_model.decoder(z_q)[0, 0].cpu().numpy()
    if len(audio) > 2048:
        audio = audio[2048:]
    return audio.astype(np.float32)


def load_texts(path: str | None, count: int, preset: str) -> list[str]:
    if not path:
        texts = ROUND2_TEXTS if preset == "round2" else LAUGH_TEXTS
    else:
        prompt_path = Path(path)
        texts = []
        for line in prompt_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                item = json.loads(line)
                texts.append(item["text"])
            else:
                texts.append(line)
    if not texts:
        raise ValueError("No generation texts provided.")
    return [texts[index % len(texts)] for index in range(count)]


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate Maya1 sentence-level laugh variants for trimming.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--snac-model", default=DEFAULT_SNAC_MODEL)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--preset", choices=["round1", "round2"], default="round1")
    parser.add_argument("--description", default=None)
    parser.add_argument("--text-file", default=None, help="Optional TXT or JSONL with one text per line. JSONL key: text.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--torch-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--min-new-tokens", type=int, default=28)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    args = parser.parse_args()

    if args.count < 1:
        raise ValueError("--count must be at least 1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.torch_dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    description = args.description or (ROUND2_DESCRIPTION if args.preset == "round2" else DEFAULT_DESCRIPTION)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    texts = load_texts(args.text_file, args.count, args.preset)

    print(f"Loading Maya1 model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print(f"Loading SNAC decoder: {args.snac_model}")
    snac_model = SNAC.from_pretrained(args.snac_model).eval()
    if torch.cuda.is_available():
        snac_model = snac_model.to("cuda")

    manifest_rows = []
    for index, text in enumerate(texts, start=1):
        tags = [tag for tag in SUPPORTED_TAGS if tag in text]
        if len(tags) != 1:
            raise ValueError(f"Text {index} must contain exactly one supported tag {SUPPORTED_TAGS}: {text}")

        clip_id = f"maya_laugh_{index:04d}"
        output_path = output_dir / f"{clip_id}.wav"
        prompt = build_prompt(tokenizer, description, text)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        print(f"[{index}/{len(texts)}] {clip_id}: {text}")
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.min_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                do_sample=True,
                eos_token_id=CODE_END_TOKEN_ID,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_ids = outputs[0, inputs["input_ids"].shape[1] :].tolist()
        snac_tokens = extract_snac_codes(generated_ids)
        if len(snac_tokens) < SNAC_TOKENS_PER_FRAME:
            raise RuntimeError(f"{clip_id}: not enough SNAC tokens generated ({len(snac_tokens)})")

        audio = decode_audio(snac_model, snac_tokens, device)
        sf.write(output_path, audio, 24000)

        manifest_rows.append(
            {
                "id": clip_id,
                "path": str(output_path),
                "description": description,
                "text": text,
                "tag": tags[0],
                "preset": args.preset,
                "model": args.model,
                "snac_model": args.snac_model,
                "sample_rate": 24000,
                "snac_tokens": len(snac_tokens),
                "duration_sec": round(len(audio) / 24000, 3),
                "seed": args.seed,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "repetition_penalty": args.repetition_penalty,
            }
        )

    manifest_path = output_dir / "manifest.jsonl"
    write_jsonl(manifest_path, manifest_rows)
    print(f"Done. Wrote {len(manifest_rows)} clips to: {output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
