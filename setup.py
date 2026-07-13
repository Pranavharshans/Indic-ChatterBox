import argparse
import os
import requests
import sys
import json
from tqdm import tqdm
from transformers import AutoTokenizer
from safetensors import safe_open
from src.config import TrainConfig


DEST_DIR = "pretrained_models"
 
CHATTERBOX_TURBO_FILES = {
    "ve.safetensors": "https://huggingface.co/ResembleAI/chatterbox-turbo/resolve/main/ve.safetensors?download=true",
    "t3_turbo_v1.safetensors": "https://huggingface.co/ResembleAI/chatterbox-turbo/resolve/main/t3_turbo_v1.safetensors?download=true",
    "s3gen_meanflow.safetensors": "https://huggingface.co/ResembleAI/chatterbox-turbo/resolve/main/s3gen_meanflow.safetensors?download=true",
    "conds.pt": "https://huggingface.co/ResembleAI/chatterbox-turbo/resolve/main/conds.pt?download=true",
    "vocab.json": "https://huggingface.co/ResembleAI/chatterbox-turbo/resolve/main/vocab.json?download=true",
    "added_tokens.json": "https://huggingface.co/ResembleAI/chatterbox-turbo/resolve/main/added_tokens.json?download=true",
    "special_tokens_map.json": "https://huggingface.co/ResembleAI/chatterbox-turbo/resolve/main/special_tokens_map.json?download=true",
    "tokenizer_config.json": "https://huggingface.co/ResembleAI/chatterbox-turbo/resolve/main/tokenizer_config.json?download=true",
    "merges.txt": "https://huggingface.co/ResembleAI/chatterbox-turbo/resolve/main/merges.txt?download=true",
    "grapheme_mtl_merged_expanded_v1.json": "https://huggingface.co/ResembleAI/chatterbox/resolve/main/grapheme_mtl_merged_expanded_v1.json?download=true"
}


CHATTERBOX_FILES = {
    "ve.safetensors": "https://huggingface.co/ResembleAI/chatterbox/resolve/main/ve.safetensors?download=true",
    "t3_cfg.safetensors": "https://huggingface.co/ResembleAI/chatterbox/resolve/main/t3_cfg.safetensors?download=true",
    "s3gen.safetensors": "https://huggingface.co/ResembleAI/chatterbox/resolve/main/s3gen.safetensors?download=true",
    "conds.pt": "https://huggingface.co/ResembleAI/chatterbox/resolve/main/conds.pt?download=true",
    "tokenizer.json": "https://huggingface.co/ResembleAI/chatterbox/resolve/main/grapheme_mtl_merged_expanded_v1.json?download=true"
}

def download_file(url, dest_path):
    """Downloads a file from a URL to a specific destination with a progress bar."""
    
    if os.path.exists(dest_path):
        if is_existing_file_valid(dest_path):
            print(f"File already exists: {dest_path}")
            return
        print(f"Existing file is invalid, redownloading: {dest_path}")
        os.remove(dest_path)

    print(f"Downloading: {os.path.basename(dest_path)}...")
    temp_path = f"{dest_path}.part"
    
    try:
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        
        with open(temp_path, 'wb') as file, tqdm(
            desc=os.path.basename(dest_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            
            for data in response.iter_content(block_size):
                
                size = file.write(data)
                bar.update(size)

        os.replace(temp_path, dest_path)
                
        print(f"Download complete: {dest_path}\n")
        
        
    except requests.exceptions.RequestException as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        print(f"Error downloading {url}: {e}")
        sys.exit(1)


def is_existing_file_valid(path):
    if os.path.getsize(path) == 0:
        return False
    if path.endswith(".safetensors"):
        try:
            with safe_open(path, framework="pt", device="cpu") as handle:
                handle.keys()
            return True
        except Exception:
            return False
    return True



def merge_and_save_turbo_tokenizer(languages=None):
    """
    It combines the downloaded original GPT-2 tokenizer with our custom vocab
    and overwrites the original files.
    """
    print("\n--- Turbo Vocab Merging Begins ---")
    
    try:
        # Preserve the exact token IDs used by the official Turbo checkpoint,
        # then append only the requested language tokens.
        base_tokenizer = AutoTokenizer.from_pretrained(DEST_DIR, local_files_only=True)
    except Exception as e:
        print(f"ERROR: The original tokenizer could not be loaded. Did you download the files correctly? -> {e}")
        return 0
        
        
    initial_len = len(base_tokenizer)
    print(f"   Original Size: {initial_len}")


    if languages:
        from IndicFinetuning.indic_languages import get_graphemes, get_language_tags

        unique_tokens_to_add = get_language_tags(languages) + get_graphemes(languages)
        print(f"Extending Turbo tokenizer for: {', '.join(languages)}")
    else:
        custom_vocab_path = os.path.join(DEST_DIR, "grapheme_mtl_merged_expanded_v1.json")
        print(f"Loading: Custom Vocab ({custom_vocab_path})")
        with open(custom_vocab_path, 'r', encoding='utf-8') as f:
            custom_data = json.load(f)
        if "model" not in custom_data or "vocab" not in custom_data["model"]:
            print("Warning: The custom VOCAB format may differ from what is expected.")
            return 0
        unique_tokens_to_add = list(custom_data["model"]["vocab"].keys())

    added_count = base_tokenizer.add_tokens(unique_tokens_to_add)
    final_len = len(base_tokenizer)

    print(f"Merging: {added_count} new token added.")
    print(f"   New Dimension: {final_len}")


    print(f"Saving: Writing the combined tokenizer to the '{DEST_DIR}' folder...")
    base_tokenizer.save_pretrained(DEST_DIR)
    
    print("MERGER SUCCESSFUL!")
    
    return final_len



def test_merge_tokenizer_process(tokenizer_path):
    
    try:

        tok = AutoTokenizer.from_pretrained(tokenizer_path)
        
        print(f"--- RESULTS ---")
        print(f"Folder: {tokenizer_path}")
        print(f"Actual Vocab Size (len): {len(tok)}")

        test_token = "[ta]"
        test_id = tok.encode(test_token, add_special_tokens=False)
        
        print(f"Test Token '{test_token}' ID: {test_id}")
        
        if len(tok) > 50276:
            print("SUCCESS! New tokens have been added.")
            
        else:
            print("ERROR: The size still appears old.")


    except Exception as e:
        print(f"Error: {e}")




def main():
    parser = argparse.ArgumentParser(description="Download and prepare Chatterbox model files.")
    parser.add_argument("--mode", choices=["standard", "turbo"], default=None)
    parser.add_argument("--dest-dir", default="pretrained_models")
    parser.add_argument("--languages", nargs="+", default=None, help="Turbo-only language tokenizer expansion.")
    args = parser.parse_args()

    global DEST_DIR
    DEST_DIR = args.dest_dir

    print("--- Chatterbox Pretrained Model Setup ---\n")
    
    # 1. Create the directory if it doesn't exist
    if not os.path.exists(DEST_DIR):
        
        print(f"Creating directory: {DEST_DIR}")
        os.makedirs(DEST_DIR, exist_ok=True)
        
    else:
        print(f"Directory found: {DEST_DIR}")
        

    cfg = TrainConfig()

    is_turbo = cfg.is_turbo if args.mode is None else args.mode == "turbo"

    if is_turbo:
        print(f"Mode: CHATTERBOX-TURBO (Checking {len(CHATTERBOX_TURBO_FILES)} files)")
        FILES_TO_DOWNLOAD = CHATTERBOX_TURBO_FILES
        
    else:
        print(f"Mode: CHATTERBOX-TTS (Checking {len(CHATTERBOX_FILES)} files)")
        FILES_TO_DOWNLOAD = CHATTERBOX_FILES

    # 2. Download files
    for filename, url in FILES_TO_DOWNLOAD.items():
        dest_path = os.path.join(DEST_DIR, filename)
        download_file(url, dest_path)

    if is_turbo:
        new_vocab_size = merge_and_save_turbo_tokenizer(args.languages)
        if new_vocab_size > 0:
            
            #test_merge_tokenizer_process(DEST_DIR)

            print("\n" + "="*60)
            print("INSTALLATION COMPLETE (CHATTERBOX-TURBO MODE)")
            print(f"All models are set up in '{DEST_DIR}/' folder.")
            print(f"Final Turbo tokenizer vocabulary size: {new_vocab_size}")
            print("="*60 + "\n")

    else:
        print("\nINSTALLATION COMPLETE (CHATTERBOX-TTS MOD)")
        print(f"All models are set up in '{DEST_DIR}/' folder.")
        print(f"Note: 'grapheme_mtl_merged_expanded_v1.json' was saved as 'tokenizer.json' for the new vocabulary.")



if __name__ == "__main__":
    main()
