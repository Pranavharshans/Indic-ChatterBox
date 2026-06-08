import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys
import time
import wave

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from IndicFinetuning.indic_text import normalize_indic_text


OPENROUTER_SPEECH_URL = "https://openrouter.ai/api/v1/audio/speech"
DEFAULT_MODEL = "google/gemini-3.1-flash-tts-preview"
DEFAULT_VOICES = ["Callirrhoe", "Aoede", "Kore", "Despina", "Laomedeia"]


PILOT_SAMPLES = [
    {
        "category": "conversation",
        "text": "ഇന്ന് ജോലി കഴിഞ്ഞ് വീട്ടിലേക്ക് വരുമ്പോൾ വഴിയരികിലെ കടയിൽ നിന്ന് ചായയുടെ മണം വന്നു. കുറച്ച് നേരം അവിടെ നിന്നു ചായ കുടിച്ചപ്പോൾ മനസ്സിന് നല്ല ആശ്വാസം തോന്നി.",
        "style": "Speak in natural everyday Malayalam, warm and conversational, not like a newsreader.",
    },
    {
        "category": "conversation",
        "text": "രാവിലെ ഓഫീസിലേക്ക് പോകാൻ തയ്യാറാകുമ്പോൾ ചെറിയ മഴ തുടങ്ങി. കുട എടുക്കാൻ മറന്നതുകൊണ്ട് കുറച്ച് നനഞ്ഞെങ്കിലും, ആ തണുത്ത കാറ്റ് ശരിക്കും നല്ലതായിരുന്നു.",
        "style": "Speak casually in Malayalam, like talking to a close friend.",
    },
    {
        "category": "conversation",
        "text": "ഇന്നത്തെ മീറ്റിംഗ് ആദ്യം കുറച്ച് ബോറായിരുന്നു, പക്ഷേ പിന്നെ എല്ലാവരും തുറന്ന് സംസാരിച്ചതോടെ കാര്യങ്ങൾ വളരെ ക്ലിയർ ആയി.",
        "style": "Speak in relaxed conversational Malayalam with natural pauses.",
    },
    {
        "category": "conversation",
        "text": "വീട്ടിൽ എത്തിയപ്പോൾ അമ്മ ചായ ഉണ്ടാക്കി വെച്ചിരുന്നു. അങ്ങനെ സാധാരണയായ ഒരു നിമിഷം പോലും ചിലപ്പോൾ മുഴുവൻ ദിവസവും നല്ലതാക്കും.",
        "style": "Speak softly and naturally in Malayalam, with a homely tone.",
    },
    {
        "category": "conversation",
        "text": "ഇന്ന് ട്രാഫിക് കുറച്ച് കൂടുതലായിരുന്നു, പക്ഷേ പാട്ട് കേട്ട് പതുക്കെ പോയപ്പോൾ യാത്ര അത്ര ബുദ്ധിമുട്ടായി തോന്നിയില്ല.",
        "style": "Speak in natural Malayalam with an easy, everyday rhythm.",
    },
    {
        "category": "conversation",
        "text": "കുറച്ച് ദിവസമായി കാണാത്ത ഒരു പഴയ സുഹൃത്തിനെ ഇന്ന് കണ്ടു. രണ്ട് മിനിറ്റ് സംസാരിച്ചപ്പോൾ തന്നെ പഴയ കാലം വീണ്ടും ഓർമ്മ വന്നു.",
        "style": "Speak warmly in Malayalam, nostalgic but still casual.",
    },
    {
        "category": "conversation",
        "text": "പുതിയ കാര്യം തുടങ്ങുമ്പോൾ അല്പം ഭയം ഉണ്ടാകും. പക്ഷേ പതുക്കെ മുന്നോട്ട് പോയാൽ വഴി തന്നെ തുറന്ന് വരും.",
        "style": "Speak calmly in Malayalam, encouraging and conversational.",
    },
    {
        "category": "conversation",
        "text": "ചിലപ്പോൾ വലിയ പ്ലാനൊന്നും വേണ്ട. നല്ലൊരു ചായയും കുറച്ച് സമയം സംസാരിക്കാൻ ഒരാളും ഉണ്ടെങ്കിൽ മതി.",
        "style": "Speak in everyday Malayalam, friendly and intimate.",
    },
    {
        "category": "neutral_replay",
        "text": "ഇന്ന് വൈകുന്നേരം മഴ പെയ്യാൻ സാധ്യതയുണ്ട്. അതിനാൽ പുറത്തേക്ക് പോകുന്നവർ കുട കൈയിൽ കരുതുന്നത് നല്ലതാണ്.",
        "style": "Speak clearly in neutral Malayalam, natural but not dramatic.",
    },
    {
        "category": "neutral_replay",
        "text": "വീട്ടിൽ നിന്ന് സ്റ്റേഷനിലേക്ക് പോകാൻ സാധാരണയായി പത്ത് മിനിറ്റ് സമയം മതിയാകും. പക്ഷേ തിരക്കുള്ള സമയത്ത് കുറച്ച് നേരത്തെ ഇറങ്ങുന്നത് നല്ലതാണ്.",
        "style": "Speak in plain neutral Malayalam with clear pronunciation.",
    },
    {
        "category": "neutral_replay",
        "text": "അടുത്ത ആഴ്ച മുതൽ ക്ലാസുകൾ രാവിലെ ഒമ്പത് മണിക്ക് ആരംഭിക്കും. വിദ്യാർത്ഥികൾ സമയത്ത് എത്താൻ ശ്രദ്ധിക്കണം.",
        "style": "Speak in neutral Malayalam, steady and understandable.",
    },
    {
        "category": "neutral_replay",
        "text": "ഈ വഴിയിലൂടെ നേരെ പോയാൽ വലത് വശത്ത് ഒരു ചെറിയ കട കാണാം. അതിന്റെ അടുത്താണ് പുതിയ ഓഫീസ്.",
        "style": "Speak in clear everyday Malayalam, neutral tone.",
    },
    {
        "category": "emotion_laughter",
        "text": "[laughter] അത് കേട്ടപ്പോൾ എനിക്ക് ശരിക്കും ചിരി അടക്കാൻ പറ്റിയില്ല. നീ പറഞ്ഞത് ഒട്ടും പ്രതീക്ഷിച്ചില്ല.",
        "style": "Speak Malayalam with a natural laugh at the beginning, then continue cheerfully.",
    },
    {
        "category": "emotion_giggle",
        "text": "[giggle] അങ്ങനെ പറയല്ലേ, കേൾക്കുമ്പോൾ തന്നെ എനിക്ക് ചിരി വരുന്നു.",
        "style": "Speak Malayalam with a small amused giggle, light and friendly.",
    },
    {
        "category": "emotion_sigh_frustration",
        "text": "[sigh] ഇന്ന് എത്ര ശ്രമിച്ചിട്ടും കാര്യം ശരിയായി തീർന്നില്ല. കുറച്ച് ക്ഷീണം തോന്നുന്നുണ്ട്.",
        "style": "Speak Malayalam with a tired frustrated sigh at the start, then continue naturally.",
    },
    {
        "category": "emotion_sigh_nervous",
        "text": "[sigh] എനിക്ക് ഉറപ്പില്ല, പക്ഷേ ഇത് പറയാതെ ഇരിക്കാൻ പറ്റില്ലെന്ന് തോന്നുന്നു.",
        "style": "Speak Malayalam with a nervous uncertain tone and a soft sigh.",
    },
    {
        "category": "emotion_cry",
        "text": "[cry] അവൻ പറഞ്ഞത് കേട്ടപ്പോൾ മനസ്സിന് വളരെ വേദനയായി. കുറച്ച് നേരം ഒന്നും പറയാൻ കഴിഞ്ഞില്ല.",
        "style": "Speak Malayalam with a tearful emotional tone, but keep the words understandable.",
    },
    {
        "category": "emotion_whisper",
        "text": "[whisper] ഇത് ആരോടും പറയരുത്. ഞാൻ നിന്നോട് മാത്രം പറയുന്ന ഒരു കാര്യമാണിത്.",
        "style": "Whisper in Malayalam, quiet and intimate, but still clear.",
    },
    {
        "category": "emotion_positive",
        "text": "ഇത് ശരിക്കും നല്ല വാർത്തയാണ്. ഇത്രയും നാളത്തെ പരിശ്രമത്തിന് ഒടുവിൽ നല്ലൊരു ഫലം കിട്ടിയതിൽ എനിക്ക് വളരെ സന്തോഷമുണ്ട്.",
        "style": "Speak Malayalam with positive excited energy, but do not shout.",
    },
    {
        "category": "emotion_curiosity",
        "text": "അത് എങ്ങനെ സംഭവിച്ചു എന്ന് എനിക്ക് ശരിക്കും അറിയണം. നീ ഒന്ന് വിശദമായി പറഞ്ഞുതരുമോ?",
        "style": "Speak Malayalam with curious conversational interest.",
    },
]


def clean_transcript_for_tts(text: str) -> str:
    return (
        text.replace("[laughter]", "")
        .replace("[giggle]", "")
        .replace("[sigh]", "")
        .replace("[cry]", "")
        .replace("[whisper]", "")
        .replace("[cough]", "")
        .strip()
    )


def make_tts_input(style: str, transcript: str) -> str:
    spoken_text = clean_transcript_for_tts(transcript)
    return (
        f"{style}\n\n"
        "Say exactly the Malayalam text below. Do not speak bracketed control tags aloud. "
        "Keep the delivery natural, clean, single-speaker, and free of background music.\n\n"
        f'Text: "{spoken_text}"'
    )


def write_pcm_wav(path: Path, pcm_bytes: bytes, sample_rate: int = 24000):
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm_bytes)


def maybe_normalize_wav(path: Path):
    tmp_path = path.with_suffix(".tmp.wav")
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-ar",
        "24000",
        "-ac",
        "1",
        str(tmp_path),
    ]
    subprocess.run(command, check=True)
    tmp_path.replace(path)


def request_speech(api_key: str, model: str, voice: str, input_text: str, response_format: str, retries: int = 3) -> bytes:
    payload = {
        "model": model,
        "voice": voice,
        "input": input_text,
        "response_format": response_format,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/Pranavharshans/Indic-ChatterBox",
        "X-Title": "Indic Chatterbox Dataset Pilot",
    }
    last_error = None
    for attempt in range(retries):
        try:
            response = requests.post(OPENROUTER_SPEECH_URL, json=payload, headers=headers, timeout=180)
            response.raise_for_status()
            return response.content
        except Exception as exc:
            last_error = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"OpenRouter TTS request failed after {retries} attempts: {last_error}")


def main():
    parser = argparse.ArgumentParser(description="Generate a small Malayalam synthetic TTS pilot dataset through OpenRouter.")
    parser.add_argument("--output", default="./IndicFinetuning/datasets/OpenRouterPilot")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--voice", action="append", default=[])
    parser.add_argument("--response-format", choices=["pcm", "mp3"], default="pcm")
    parser.add_argument("--limit", type=int, default=len(PILOT_SAMPLES))
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("Set OPENROUTER_API_KEY before running this script.")

    output_dir = Path(args.output)
    wav_dir = output_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.csv"
    voices = args.voice or DEFAULT_VOICES

    selected = PILOT_SAMPLES[: args.limit]
    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="|", lineterminator="\n")
        for index, sample in enumerate(selected):
            file_id = f"openrouter_pilot_{index:04d}_{sample['category']}"
            transcript = normalize_indic_text(sample["text"])
            voice = voices[index % len(voices)]
            tts_input = make_tts_input(sample["style"], transcript)
            print(f"[{index + 1}/{len(selected)}] {file_id} voice={voice}")
            audio_bytes = request_speech(api_key, args.model, voice, tts_input, args.response_format)

            wav_path = wav_dir / f"{file_id}.wav"
            if args.response_format == "pcm":
                write_pcm_wav(wav_path, audio_bytes)
            else:
                mp3_path = wav_path.with_suffix(".mp3")
                mp3_path.write_bytes(audio_bytes)
                subprocess.run(
                    ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(mp3_path), "-ar", "24000", "-ac", "1", str(wav_path)],
                    check=True,
                )
                mp3_path.unlink(missing_ok=True)
            maybe_normalize_wav(wav_path)
            writer.writerow([file_id, transcript, transcript, "ml"])

    print(f"Metadata: {metadata_path}")
    print(f"WAVs: {wav_dir}")
    print(f"Rows: {len(selected)}")


if __name__ == "__main__":
    main()

