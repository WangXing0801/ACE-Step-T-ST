#!/usr/bin/env python3

import argparse
import os

import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

from generate_prompts_lyrics import inference, parse_prompt_lyrics


def load_model(model_path: str):
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.disable_talker()
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    return model, processor


def do_files(data_dir, overwrite, do_lyrics):
    model, processor = load_model("Qwen/Qwen2.5-Omni-3B")

    # Formats supported by torchaudio
    extensions = {
        ".aac",
        ".flac",
        ".m4a",
        ".mp3",
        ".ogg",
        ".wav",
    }

    for file in sorted(os.listdir(data_dir)):
        stem, ext = os.path.splitext(file)
        if ext.lower() not in extensions:
            continue

        file_path = os.path.join(data_dir, file)
        stem_path = os.path.join(data_dir, stem)
        prompt_path = stem_path + "_prompt.txt"
        lyrics_path = stem_path + "_lyrics.txt"

        need_prompt = overwrite or (not os.path.exists(prompt_path))
        need_lyrics = do_lyrics and (overwrite or (not os.path.exists(lyrics_path)))

        if not (need_prompt or need_lyrics):
            continue

        print(file)
        content = inference(file_path, model, processor, do_lyrics)
        prompt, lyrics = parse_prompt_lyrics(content)

        if need_prompt:
            with open(prompt_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(prompt)
        if need_lyrics:
            with open(lyrics_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(lyrics)


@torch.inference_mode
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=r"C:\data\audio")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--lyrics", action="store_true")
    args = parser.parse_args()

    do_files(
        data_dir=args.data_dir,
        overwrite=args.overwrite,
        do_lyrics=args.lyrics,
    )


if __name__ == "__main__":
    main()
