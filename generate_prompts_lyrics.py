#!/usr/bin/env python3
import os
# 🔧 禁用 PyTorch 动态编译和 Triton 相关功能
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCHINDUCTOR_DISABLE'] = '1'

import argparse
import json

import torch
import torchaudio
from gptqmodel import GPTQModel
from gptqmodel.models.auto import MODEL_MAP, SUPPORTED_MODELS
from gptqmodel.models.base import BaseGPTQModel
from huggingface_hub import snapshot_download
from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniProcessor

from modeling_qwen2_5_omni_low_VRAM_mode import Qwen2_5OmniForConditionalGeneration

# 🔧 禁用 PyTorch 编译功能
torch._dynamo.config.suppress_errors = True
torch.compiler.disable()

QWEN_SAMPLE_RATE = 16000

QWEN_SYSTEM_PROMPT = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."

# Roughly based on the official prompt https://github.com/ace-step/ACE-Step/blob/main/TRAIN_INSTRUCTION.md
PROMPT = r"""Analyze the input audio:
1. `genre`: Most representative genres of the audio.
2. `subgenre`: Three or more tags of specific sub-genres and techniques.
3. `instrument`: All audibly present instruments in the audio, except vocal.
4. `tempo`: Tags describing the tempo of the audio. Do not use number or BPM.
5. `mood`: Tags describing the mood of the audio.
6. `has_vocal`: Whether there is any vocal in the audio.
7. `vocal`: If there is any vocal, then output a list of tags describing the vocal timbre. Otherwise, output an empty list.

Output format:
```json
{
  "genre": <str list>,
  "subgenre": <str list>,
  "instrument": <str list>,
  "tempo": <str list>,
  "mood": <str list>,
  "has_vocal": <bool>,
  "vocal": <str list>
}
```"""

PROMPT_LYRICS = r"""Analyze the input audio:
1. `genre`: Most representative genres of the audio.
2. `subgenre`: Three or more tags of specific sub-genres and techniques.
3. `instrument`: All audibly present instruments in the audio, except vocal.
4. `tempo`: Tags describing the tempo of the audio. Do not use number or BPM.
5. `mood`: Tags describing the mood of the audio.
6. `has_vocal`: Whether there is any vocal in the audio.
7. `vocal`: If there is any vocal, then output a list of tags describing the vocal timbre. Otherwise, output an empty list.
8. `lyrics`: If there is any vocal, then transcribe the lyrics and output at most 1000 characters. Otherwise, output an empty string. Use \n after each sentence.

Output format:
```json
{
  "genre": <str list>,
  "subgenre": <str list>,
  "instrument": <str list>,
  "tempo": <str list>,
  "mood": <str list>,
  "has_vocal": <bool>,
  "vocal": <str list>,
  "lyrics": <str>
}
```"""


@classmethod
def patched_from_config(cls, config, *args, **kwargs):
    kwargs.pop("trust_remote_code", None)
    model = cls._from_config(config, **kwargs)
    return model


Qwen2_5OmniForConditionalGeneration.from_config = patched_from_config


class Qwen2_5OmniThinkerGPTQ(BaseGPTQModel):
    loader = Qwen2_5OmniForConditionalGeneration
    base_modules = [
        "thinker.model.embed_tokens",
        "thinker.model.norm",
        "thinker.audio_tower",
        "thinker.model.rotary_emb",
    ]
    pre_lm_head_norm_module = "thinker.model.norm"
    require_monkeypatch = False
    layers_node = "thinker.model.layers"
    layer_type = "Qwen2_5OmniDecoderLayer"
    layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"],
    ]

    def pre_quantize_generate_hook_start(self):
        self.thinker.audio_tower = self.thinker.audio_tower.to(
            self.quantize_config.device
        )

    def pre_quantize_generate_hook_end(self):
        self.thinker.audio_tower = self.thinker.audio_tower.to("cpu")

    def preprocess_dataset(self, sample):
        return sample


MODEL_MAP["qwen2_5_omni"] = Qwen2_5OmniThinkerGPTQ
SUPPORTED_MODELS.extend(["qwen2_5_omni"])


def load_model(model_path: str):
    if not os.path.exists(model_path):
        model_path = snapshot_download(repo_id=model_path)

    device_map = {
        "thinker.model": "cuda",
        "thinker.lm_head": "cuda",
        # "thinker.visual": "cpu",
        "thinker.audio_tower": "cpu",
        "talker": "cpu",
        "token2wav": "cpu",
    }

    model = GPTQModel.load(
        model_path,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",  # 🔧 改为 eager 避免 flash_attention 问题
    )
    model.disable_talker()

    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    return model, processor


def read_audio(file_path):
    audio, sr = torchaudio.load(file_path)
    audio = audio[:, : sr * 360]
    if sr != QWEN_SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, sr, QWEN_SAMPLE_RATE)
        sr = QWEN_SAMPLE_RATE
    audio = audio.mean(dim=0, keepdim=True)
    return audio, sr


def inference(file_path, model, processor, do_lyrics):
    audio, _ = read_audio(file_path)
    audio = audio.numpy().squeeze(axis=0)

    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": PROMPT_LYRICS if do_lyrics else PROMPT},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio},
            ],
        },
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    )

    # Copy tensors to GPU and match dtypes
    ks = list(inputs.keys())
    for k in ks:
        if hasattr(inputs[k], "to"):
            inputs[k] = inputs[k].to("cuda")
            if inputs[k].dtype.is_floating_point:
                inputs[k] = inputs[k].to(model.dtype)

    # 🔧 添加错误处理
    try:
        output_ids = model.thinker.generate(
            **inputs,
            max_new_tokens=1000,
            use_audio_in_video=False,
        )
    except Exception as e:
        print(f"Warning: Generate failed with error: {e}")
        # 返回空的输出
        return "{}"

    generate_ids = output_ids[:, inputs.input_ids.shape[1] :]
    response = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return response


def parse_prompt_lyrics(content):
    content = content.replace("```json", "")
    content = content.replace("```", "")
    content = content.strip()

    prompt = ""
    lyrics = ""
    try:
        tags = []
        data = json.loads(content)
        tags += data["genre"]
        tags += data["subgenre"]
        tags += data["instrument"]
        tags += data["tempo"]
        tags += data["mood"]
        tags += [x.strip() + " vocal" for x in data["vocal"] if x != "vocal"]

        tags = [x.strip().lower() for x in tags]
        # The order of tags does not matter, so we sort them here
        # Tags will be shuffled in training
        tags = sorted(set(tags))
        prompt = ", ".join(tags)

        lyrics = data.get("lyrics")
        if not lyrics:
            lyrics = "[instrumental]"
    except Exception as e:
        print(f"Failed to parse content: {e}")
        print(content)
        prompt = "audio"
        lyrics = "[instrumental]"

    return prompt, lyrics


def do_files(data_dir, overwrite, do_lyrics):
    # 🔧 添加错误处理
    try:
        model, processor = load_model("Qwen/Qwen2.5-Omni-7B-GPTQ-Int4")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Formats supported by torchaudio
    extensions = {
        ".aac",
        ".flac",
        ".m4a",
        ".mp3",
        ".ogg",
        ".wav",
    }

    files_processed = 0
    files_error = 0

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

        print(f"Processing: {file}")
        try:
            content = inference(file_path, model, processor, do_lyrics)
            prompt, lyrics = parse_prompt_lyrics(content)

            if need_prompt:
                with open(prompt_path, "w", encoding="utf-8", newline="\n") as f:
                    f.write(prompt)
            if need_lyrics:
                with open(lyrics_path, "w", encoding="utf-8", newline="\n") as f:
                    f.write(lyrics)
            
            files_processed += 1
            print(f"✓ Success: {file}")
            
        except Exception as e:
            files_error += 1
            print(f"✗ Error processing {file}: {e}")
            # 创建默认文件避免重复处理
            if need_prompt:
                with open(prompt_path, "w", encoding="utf-8", newline="\n") as f:
                    f.write("audio")
            if need_lyrics:
                with open(lyrics_path, "w", encoding="utf-8", newline="\n") as f:
                    f.write("[instrumental]")
            continue
    
    print(f"Processing completed. Success: {files_processed}, Errors: {files_error}")


@torch.inference_mode()
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
