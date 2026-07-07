import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from IndicFinetuning.indic_engine import attach_indic_tokenizer, get_engine_class, tokenizer_vocab_size
from IndicFinetuning.indic_text import apply_language_tag, normalize_indic_text
from src.chatterbox_.models.t3.t3 import T3
from src.model import resize_and_load_t3_weights
from src.utils import setup_logger


PROMPTS = {
    "hi": [
        "आज सुबह बाजार जाते समय रास्ते में हल्की बारिश शुरू हो गई.",
        "मुझे लगता है कि यह काम हम आराम से शाम तक पूरा कर सकते हैं.",
        "तुम चाय बनाओ, तब तक मैं मेहमानों के लिए नाश्ता लगा देता हूं.",
        "कल की बैठक में सभी लोगों ने अपनी राय बहुत साफ तरीके से रखी.",
        "यह कहानी सुनकर मुझे अपने बचपन के स्कूल के दिन याद आ गए.",
    ],
    "ta": [
        "இன்று காலை சந்தைக்கு போகும் வழியில் மெதுவாக மழை பெய்ய ஆரம்பித்தது.",
        "இந்த வேலையை நாம் அமைதியாக மாலை வரை முடிக்க முடியும் என்று நினைக்கிறேன்.",
        "நீ தேநீர் தயாரி, அதற்குள் நான் விருந்தினர்களுக்கான சிற்றுண்டியை வைக்கிறேன்.",
        "நேற்றைய கூட்டத்தில் எல்லோரும் தங்கள் கருத்தை தெளிவாக சொன்னார்கள்.",
        "இந்தக் கதையை கேட்டதும் என் பள்ளி நாட்கள் நினைவுக்கு வந்தது.",
    ],
    "te": [
        "ఈ రోజు ఉదయం మార్కెట్ కి వెళ్తుంటే మార్గంలో మెల్లగా వర్షం మొదలైంది.",
        "ఈ పని మనం సాయంత్రం లోపల ప్రశాంతంగా పూర్తి చేయగలమని అనుకుంటున్నాను.",
        "నువ్వు చాయ్ తయారు చేయి, అప్పటికి నేను అతిథుల కోసం తినుబండారాలు పెడతాను.",
        "నిన్నటి సమావేశంలో అందరూ తమ అభిప్రాయాన్ని చాలా స్పష్టంగా చెప్పారు.",
        "ఈ కథ విన్న వెంటనే నాకు నా పాఠశాల రోజుల జ్ఞాపకాలు వచ్చాయి.",
    ],
    "ml": [
        "ഇന്ന് രാവിലെ മാർക്കറ്റിലേക്ക് പോകുമ്പോൾ വഴിയിൽ ചെറിയ മഴ തുടങ്ങി.",
        "ഈ ജോലി നമുക്ക് ശാന്തമായി വൈകുന്നേരത്തിനുള്ളിൽ തീർക്കാം എന്ന് തോന്നുന്നു.",
        "നീ ചായ ഉണ്ടാക്കൂ, അതിനുള്ളിൽ ഞാൻ അതിഥികൾക്കുള്ള പലഹാരം വയ്ക്കാം.",
        "ഇന്നലത്തെ യോഗത്തിൽ എല്ലാവരും അവരുടെ അഭിപ്രായം വളരെ വ്യക്തമായി പറഞ്ഞു.",
        "ഈ കഥ കേട്ടപ്പോൾ എനിക്ക് എന്റെ സ്കൂൾ കാലത്തെ ദിവസങ്ങൾ ഓർമ്മ വന്നു.",
    ],
    "kn": [
        "ಇಂದು ಬೆಳಿಗ್ಗೆ ಮಾರುಕಟ್ಟೆಗೆ ಹೋಗುವಾಗ ದಾರಿಯಲ್ಲಿ ಸಣ್ಣ ಮಳೆ ಶುರುವಾಯಿತು.",
        "ಈ ಕೆಲಸವನ್ನು ನಾವು ಸಂಜೆ ಒಳಗೆ ಶಾಂತವಾಗಿ ಮುಗಿಸಬಹುದು ಎಂದು ನನಗೆ ಅನಿಸುತ್ತದೆ.",
        "ನೀನು ಚಹಾ ಮಾಡು, ಅಷ್ಟರಲ್ಲಿ ನಾನು ಅತಿಥಿಗಳಿಗಾಗಿ ತಿಂಡಿಯನ್ನು ಇಡುತ್ತೇನೆ.",
        "ನಿನ್ನೆ ನಡೆದ ಸಭೆಯಲ್ಲಿ ಎಲ್ಲರೂ ತಮ್ಮ ಅಭಿಪ್ರಾಯವನ್ನು ಸ್ಪಷ್ಟವಾಗಿ ಹೇಳಿದರು.",
        "ಈ ಕಥೆ ಕೇಳಿದಾಗ ನನಗೆ ನನ್ನ ಶಾಲೆಯ ದಿನಗಳು ನೆನಪಿಗೆ ಬಂದವು.",
    ],
    "bn": [
        "আজ সকালে বাজারে যাওয়ার পথে হালকা বৃষ্টি শুরু হয়ে গেল.",
        "আমার মনে হয় আমরা এই কাজটা শান্তভাবে সন্ধ্যার মধ্যে শেষ করতে পারব.",
        "তুমি চা বানাও, ততক্ষণে আমি অতিথিদের জন্য নাশতা সাজিয়ে দিচ্ছি.",
        "গতকালের মিটিংয়ে সবাই খুব পরিষ্কারভাবে নিজের মতামত বলেছে.",
        "এই গল্পটা শুনে আমার ছোটবেলার স্কুলের দিনগুলোর কথা মনে পড়ে গেল.",
    ],
    "mr": [
        "आज सकाळी बाजारात जाताना रस्त्यात हलका पाऊस सुरू झाला.",
        "मला वाटते हे काम आपण शांतपणे संध्याकाळपर्यंत पूर्ण करू शकतो.",
        "तू चहा कर, तोपर्यंत मी पाहुण्यांसाठी नाश्ता लावतो.",
        "कालच्या बैठकीत सगळ्यांनी आपले मत अगदी स्पष्टपणे मांडले.",
        "ही गोष्ट ऐकून मला माझ्या शाळेतील लहानपणीचे दिवस आठवले.",
    ],
    "gu": [
        "આજે સવારે બજારમાં જતા રસ્તામાં હળવો વરસાદ શરૂ થઈ ગયો.",
        "મને લાગે છે કે આ કામ આપણે શાંતિથી સાંજ સુધી પૂરું કરી શકીશું.",
        "તું ચા બનાવ, ત્યાં સુધી હું મહેમાનો માટે નાસ્તો મૂકી દઉં છું.",
        "ગઈકાલની બેઠકમાં બધાએ પોતાનો મત ખૂબ સ્પષ્ટ રીતે કહ્યું.",
        "આ વાર્તા સાંભળીને મને મારા શાળાના બાળપણના દિવસો યાદ આવી ગયા.",
    ],
    "pa": [
        "ਅੱਜ ਸਵੇਰੇ ਬਾਜ਼ਾਰ ਜਾਂਦੇ ਸਮੇਂ ਰਸਤੇ ਵਿੱਚ ਹੌਲੀ ਬਾਰਿਸ਼ ਸ਼ੁਰੂ ਹੋ ਗਈ.",
        "ਮੈਨੂੰ ਲੱਗਦਾ ਹੈ ਕਿ ਇਹ ਕੰਮ ਅਸੀਂ ਸ਼ਾਮ ਤੱਕ ਆਰਾਮ ਨਾਲ ਮੁਕੰਮਲ ਕਰ ਸਕਦੇ ਹਾਂ.",
        "ਤੂੰ ਚਾਹ ਬਣਾ, ਉਦੋਂ ਤੱਕ ਮੈਂ ਮਹਿਮਾਨਾਂ ਲਈ ਨਾਸ਼ਤਾ ਰੱਖ ਦਿੰਦਾ ਹਾਂ.",
        "ਕੱਲ੍ਹ ਦੀ ਮੀਟਿੰਗ ਵਿੱਚ ਸਭ ਨੇ ਆਪਣੀ ਰਾਏ ਬਹੁਤ ਸਾਫ਼ ਤਰੀਕੇ ਨਾਲ ਦਿੱਤੀ.",
        "ਇਹ ਕਹਾਣੀ ਸੁਣ ਕੇ ਮੈਨੂੰ ਆਪਣੇ ਸਕੂਲ ਦੇ ਬਚਪਨ ਵਾਲੇ ਦਿਨ ਯਾਦ ਆ ਗਏ.",
    ],
    "ur": [
        "آج صبح بازار جاتے ہوئے راستے میں ہلکی بارش شروع ہو گئی۔",
        "مجھے لگتا ہے کہ یہ کام ہم آرام سے شام تک مکمل کر سکتے ہیں۔",
        "تم چائے بنا لو، اتنی دیر میں میں مہمانوں کے لیے ناشتہ رکھ دیتا ہوں۔",
        "کل کی میٹنگ میں سب نے اپنی رائے بہت صاف انداز میں بیان کی۔",
        "یہ کہانی سن کر مجھے اپنے بچپن کے اسکول کے دن یاد آ گئے۔",
    ],
    "or": [
        "ଆଜି ସକାଳେ ବଜାରକୁ ଯାଉଥିବା ବେଳେ ରାସ୍ତାରେ ହାଲୁକା ବର୍ଷା ଆରମ୍ଭ ହେଲା.",
        "ମୋତେ ଲାଗୁଛି ଏହି କାମଟି ଆମେ ସନ୍ଧ୍ୟା ପର୍ଯ୍ୟନ୍ତ ଶାନ୍ତିରେ ସରିପାରିବୁ.",
        "ତୁମେ ଚା ବନାଅ, ସେତେବେଳେ ମୁଁ ଅତିଥିମାନଙ୍କ ପାଇଁ ଖାଦ୍ୟ ରଖୁଛି.",
        "ଗତକାଲିର ବୈଠକରେ ସମସ୍ତେ ନିଜ ମତ ସ୍ପଷ୍ଟ ଭାବରେ କହିଥିଲେ.",
        "ଏହି କାହାଣୀ ଶୁଣି ମୋତେ ମୋର ସ୍କୁଲ ଦିନଗୁଡ଼ିକ ମନେ ପଡ଼ିଲା.",
    ],
    "as": [
        "আজি ৰাতিপুৱা বজাৰলৈ যাওঁতে বাটত পাতল বৰষুণ আৰম্ভ হ'ল.",
        "মোৰ মনে হয় এই কামটো আমি সন্ধিয়ালৈকে শান্তভাৱে শেষ কৰিব পাৰিম.",
        "তুমি চাহ বনোৱা, তেতিয়ালৈ মই অতিথিসকলৰ বাবে জলপান সাজু কৰোঁ.",
        "কালিৰ সভাত সকলোৱে নিজৰ মতামত বহুত স্পষ্টকৈ ক'লে.",
        "এই গল্পটো শুনি মোৰ সৰুবেলাৰ স্কুলৰ দিনবোৰ মনত পৰিল.",
    ],
}


def load_config(config_file: str, class_name: str):
    path = Path(config_file).resolve()
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load config file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_text(config, text: str, language_id: str) -> str:
    text = normalize_indic_text(text, config.normalize_unicode)
    return apply_language_tag(text, language_id, config.add_language_tag)


def load_lora_engine(config, adapter_path: str, device: str):
    from peft import PeftModel

    if not config.is_turbo:
        config.new_vocab_size = tokenizer_vocab_size(config)

    engine_class = get_engine_class(config.is_turbo)
    temp_engine = engine_class.from_local(config.model_dir, device="cpu")
    temp_engine = attach_indic_tokenizer(temp_engine, config)
    pretrained_state = temp_engine.t3.state_dict()
    original_config = temp_engine.t3.hp
    original_config.text_tokens_dict_size = config.new_vocab_size
    setattr(original_config, "use_cache", False)

    new_t3 = T3(hp=original_config)
    new_t3 = resize_and_load_t3_weights(new_t3, pretrained_state)
    if config.is_turbo and hasattr(new_t3.tfmr, "wte"):
        del new_t3.tfmr.wte

    del temp_engine
    del pretrained_state

    engine = engine_class.from_local(config.model_dir, device="cpu")
    engine = attach_indic_tokenizer(engine, config)
    engine.t3 = PeftModel.from_pretrained(new_t3, adapter_path, is_trainable=False)
    engine.t3.to(device).eval()
    engine.s3gen.to(device).eval()
    engine.ve.to(device).eval()
    engine.device = device
    return engine


def main():
    parser = argparse.ArgumentParser(description="Generate fixed multilingual eval samples for an Indic adapter.")
    parser.add_argument("--config-file", default="./IndicFinetuning/multilingual_indic/first_real_run/config.py")
    parser.add_argument("--config-class", default="MultilingualIndicConfig")
    parser.add_argument("--adapter-path", default="./IndicFinetuning/outputs/multilingual_indic_first_real_run/indic_adapter")
    parser.add_argument("--output-dir", default="./IndicFinetuning/outputs/multilingual_indic_eval_samples")
    parser.add_argument("--prompt-wav", required=True, help="Reference speaker wav used for all samples.")
    parser.add_argument("--languages", nargs="+", default=list(PROMPTS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    args = parser.parse_args()

    logger = setup_logger("Multilingual-Indic-Eval")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = load_config(args.config_file, args.config_class)
    adapter_path = Path(args.adapter_path)
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")
    prompt_wav = Path(args.prompt_wav)
    if not prompt_wav.exists():
        raise FileNotFoundError(f"Prompt wav not found: {prompt_wav}")

    set_seed(args.seed)
    logger.info(f"Loading adapter: {adapter_path}")
    engine = load_lora_engine(config, str(adapter_path), device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for language_id in args.languages:
        if language_id not in PROMPTS:
            raise ValueError(f"No prompts configured for language: {language_id}")
        language_dir = output_dir / language_id
        language_dir.mkdir(parents=True, exist_ok=True)
        prompt_lines = []

        for index, prompt in enumerate(PROMPTS[language_id], start=1):
            formatted_text = format_text(config, prompt, language_id)
            wav_tensor = engine.generate(
                text=formatted_text,
                audio_prompt_path=str(prompt_wav),
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
            )
            if isinstance(wav_tensor, tuple):
                wav_tensor = wav_tensor[0]
            wav_np = wav_tensor.squeeze().detach().cpu().numpy()
            output_file = language_dir / f"{language_id}_{index:02d}.wav"
            sf.write(str(output_file), wav_np, engine.sr)
            prompt_lines.append(f"{index}. {prompt}")
            manifest.append(
                {
                    "language": language_id,
                    "index": index,
                    "text": prompt,
                    "audio": str(output_file),
                }
            )
            logger.info(f"Wrote {output_file}")

        (language_dir / "prompts.txt").write_text("\n".join(prompt_lines) + "\n", encoding="utf-8")

    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Eval samples saved to: {output_dir}")


if __name__ == "__main__":
    main()
