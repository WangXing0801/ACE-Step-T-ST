

# ACE-Step 模型训练操作流程文档

## 📋 训练准备阶段

### 1. 准备音频数据 (gui要求，用户可以上传音频文件，如果音频文件是中文名自动转成拼音，字符不要太长，做到避免文件名重复就行。如果同一个文件夹下已经存在音频文件，则给出提示换一个文件夹或者清空当前文件夹，音频文件放在下面示例文件夹里，下面的xxx都代表音频文件的确定后的名字，确定存放目录后，用户是可以手动修改的)
- 收集训练用的音频文件
- 将音频文件存放至指定目录（示例：`D:\AIJOB\ACE-Step-T\ACE-Step\Taudio\xxx`）

### 2. 生成提示词 (gui要求，这里是按钮，点击生成提示词，等待模型运行完可以有个文本框显示提示词进行浏览和更改)
```powershell
python generate_prompts_lyrics.py --data_dir D:\AIJOB\ACE-Step-T\ACE-Step\Taudio\xxx
```

### 3. 生成歌词 (gui要求，这里是按钮，点击生成歌词，等待模型运行完可以有个文本框显示提示词进行浏览和更改)
```powershell
python generate_prompts_lyrics.py --data_dir D:\AIJOB\ACE-Step-T\ACE-Step\Taudio\xxx --lyrics
```
### 4. 创建文件名数据集 (gui要求，这里也是按钮，创建数据集)
```powershell
python convert2hf_dataset_new.py --data_dir D:\AIJOB\ACE-Step-T\ACE-Step\Taudio\xxx --output_name D:\AIJOB\ACE-Step-T\ACE-Step\Taudio\xxx_filenames
```

### 5. 音频预处理 (gui要求，这里也是按钮，音频预处理)
```powershell
python preprocess_dataset_new.py --input_name D:\AIJOB\ACE-Step-T\ACE-Step\Taudio\xxx_filenames --output_dir D:\AIJOB\ACE-Step-T\ACE-Step\Taudio\xxx_prep
```

## 🏋️ 训练执行阶段 

### 6. 开始训练 (gui要求，这里也是按钮，开始训练)
```powershell
python trainer_new.py --dataset_path D:\AIJOB\ACE-Step-T\ACE-Step\Taudio\音频文件名文件夹_prep
```

gui要求底部要有日志框，以上步骤按序号顺序执行，没有执行完上一步骤不可以进行下一步。
增加启动 TensorBoardLogger 功能按钮，启动后使用默认浏览器自动打开对应网站。
增加一个重置按钮，可以恢复所有默认设置。
以上提到的py文件和这个gui文件在同一个目录平级。
增加训练模型参数设置
--batch_size	1	批大小
--num_workers	0	DataLoader 使用的线程数
--tag_dropout	0.5	文本标签的 dropout 概率
--learning_rate	1e-4	学习率
--max_steps	2000	最大训练步数
--precision	"bf16-mixed"	混合精度训练设置
--save_every_n_train_steps	100	每多少步保存一次检查点
以上内容参照我下面发给你的模型训练文件一起完成gui文件
















###!!!!!!注意！！！####
如果是conda环境，最后一步报错的话，使用下面方法
临时修改源码
找到文件：

TEXT
C:\ProgramData\anaconda3\envs\ace_step_t\lib\site-packages\lightning_fabric\accelerators\cuda.py
将 _check_cuda_matmul_precision 函数修改为：

PYTHON
def _check_cuda_matmul_precision(device: torch.device) -> None:
    # 临时绕过检查
    return



**重要注意事项：**
- 训练前需清空 `checkpoints` 目录
- LoRA 权重将保存在 `checkpoints` 目录中
- 默认使用 Wandb 日志记录（可移除 `WandbLogger`）

## 🔧 训练后处理

### 6. LoRA 权重优化
训练完成后，需要调整 LoRA 强度参数：

**手动设置方式：**
- rsLoRA 模式：强度 = `alpha / sqrt(rank)`
- 非 rsLoRA 模式：强度 = `alpha / rank`

**自动处理方式：**
```powershell
python add_alpha_in_lora.py --input_name checkpoints/epoch=0-step=100_lora/pytorch_lora_weights.safetensors --output_name out.safetensors --lora_config_path config/lora_config_transformer_only.json
```
处理后的 LoRA 文件可在 ComfyUI 中直接使用，强度设置为 1。

## 🎯 优化建议

### 训练技巧
1. **初学者建议**：先用单个音频过拟合测试训练流程
2. **模块选择**：可冻结歌词解码器，仅训练 transformer
3. **优化器设置**：
   - Adam 类优化器需注意 `1 - beta2` 与 `1 / max_steps` 的关系
   - 使用 Prodigy 优化器时确保参数 d 能增长到较大值

### 性能优化
- 启用梯度检查点以节省显存
- 全部使用 bf16 精度训练
- 支持在单张 RTX 3080（<10GB 显存）上运行























# [ACE-Step](https://github.com/ace-step/ACE-Step) fork

## Progress

* Separate data preprocessing (music and text encoding) and training
* Enable gradient checkpointing
* Cast everything to bf16

Now I can run the training on a single RTX 3080 with < 10 GB VRAM and 0.3 it/s speed, using music duration < 360 seconds and LoRA rank = 64.

I've trained some LoRAs at https://huggingface.co/woctordho/ACE-Step-v1-LoRA-collection

## Usage

1. Collect some audios, for example, in the directory `C:\data\audio`.

2. Generate prompts using Qwen2.5-Omni-7B:
    ```pwsh
    python generate_prompts_lyrics.py --data_dir C:\data\audio
    ```
    Each prompt is a list of tags separated by comma space `, ` without EOL. The order of tags will be randomly shuffled in the training. (TODO: Check how natural language prompts affect the performance.)

    **(Experimental)** The above script uses gptqmodel. Alternatively, you can use llama.cpp:
    <details>
    <summary>Expand</summary>

    Start llama-server (by default it listens host 127.0.0.1, port 8080)
    ```pwsh
    llama-server -m Qwen2.5-Omni-7B-Q8_0.gguf --mmproj mmproj-Qwen2.5-Omni-7B-Q8_0.gguf -c 32768 -fa -ngl 999 --cache-reuse 256
    ```
    Then run
    ```pwsh
    python generate_prompts_lyrics_llamacpp.py --data_dir C:\data\audio
    ```
    After this step, you can shut down llama-server to save VRAM.

    Unfortunately, for now llama.cpp did not reproduce the original model with enough accuracy, so tags may not be accurate and lyrics almost does not work at all.
    </details>

    **(Experimental)** You can also generate lyrics:
    <details>
    <summary>Expand</summary>

    ```pwsh
    python generate_prompts_lyrics.py --data_dir C:\data\audio --lyrics
    ```
    It seems Qwen2.5-Omni-7B works well for Chinese lyrics, but not so well for English and other languages.
    </details>

    Besides using an AI model to transcribe lyrics, you can also extract lyrics embedded in the audio file, or query online databases such as [163MusicLyrics](https://github.com/jitwxs/163MusicLyrics), [LyricsGenius](https://github.com/johnwmillr/LyricsGenius), [LyricWiki](https://archive.org/details/lyricsfandomcom-20200216-patched.7z). You may try [ace-data_tool](https://github.com/methmx83/ace-data_tool).

    For music without vocal, just use `[instrumental]` for the lyrics.

    At this point, the directory `C:\data\audio` should be like:
    ```
    audio1.wav
    audio1_lyrics.txt
    audio1_prompt.txt
    audio2.mp3
    audio2_lyrics.txt
    audio2_prompt.txt
    ...
    ```

4. Create a dataset that only contains the filenames, not the audio data:
    ```pwsh
    python convert2hf_dataset_new.py --data_dir C:\data\audio --output_name C:\data\audio_filenames
    ```

5. Load the audios, do the preprocessing, save to a new dataset:
    ```pwsh
    python preprocess_dataset_new.py --input_name C:\data\audio_filenames --output_dir C:\data\audio_prep
    ```
    The preprocessed dataset takes ~0.2 MB for every second of input audio.

    TODO: If you have a lot of training data and want to reduce disk space requirement, we can add a switch to move MERT and mHuBERT from preprocessing to training.

7. Do the training:
    ```pwsh
    python trainer_new.py --dataset_path C:\data\audio_prep
    ```
    The LoRA will be saved to the directory `checkpoints`. Make sure to clear this directory before training, otherwise the LoRA may not be correctly saved.

    If you have a lot of VRAM, you can remove `self.transformer.enable_gradient_checkpointing()` for faster training speed.

    My script uses Wandb rather than TensorBoard. If you don't need it, you can remove the `WandbLogger`.

9. LoRA strength:

    At this point, when loading the LoRA in ComfyUI, you need to set the LoRA strength to `alpha / sqrt(rank)` (for rsLoRA) or `alpha / rank` (for non-rsLoRA). For example, if rank = 64, alpha = 1, rsLoRA is enabled, then the LoRA strength should be `1 / sqrt(64) = 0.125`.

    To avoid manually setting this, you can run:
    ```pwsh
    python add_alpha_in_lora.py --input_name checkpoints/epoch=0-step=100_lora/pytorch_lora_weights.safetensors --output_name out.safetensors --lora_config_path config/lora_config_transformer_only.json
    ```
    Then load `out.safetensors` in ComfyUI and set the LoRA strength to 1.

## Tips

* If you don't have experience, you can first try to train with a single audio and make sure that it can be overfitted. This is a sanity check of the training pipeline
* You can freeze the lyrics decoder and only train the transformer using `config/lora_config_transformer_only.json`. I think training the lyrics decoder is needed only when adding a new language
* In the LoRA config, you can add
    ```
    "projectors.0.0",
    "projectors.0.2",
    "projectors.0.4",
    "projectors.1.0",
    "projectors.1.2",
    "projectors.1.4",
    ```
    to `target_modules`. This may help the model learn the music style
* When using an Adam-like optimizer (including AdamW and Prodigy), you should not let `1 - beta2` be much smaller than `1 / max_steps`
* When using Prodigy optimizer, make sure that `d` rises to a large value (such as 1e-4, should be much larger than the initial 1e-6) after `1 / (1 - beta2)` steps
* After training, you can prune the LoRA using SVD, such as [`resize_lora.py`](https://github.com/kohya-ss/sd-scripts/blob/main/networks/resize_lora.py) in Kohya's sd-scripts. If the dynamic pruning tells you that the LoRA rank can be much smaller without changing the output quality, then next time you can train the LoRA using a smaller rank

## TODO

* Support batch size > 1, maybe bucketing samples with similar lengths
* How to normalize the audio loudness before preprocessing? It seems the audios generated by ACE-Step usually have loudness in -16 .. -12 LUFS, and they don't follow prompts like 'loud' and 'quiet'
* To generate the tags, maybe a specialized tagger can perform better than Qwen2.5-Omni-7B, such as [OpenJMLA](https://huggingface.co/UniMus/OpenJMLA), [GLAP](https://github.com/xiaomi-research/dasheng-glap), [MuFun](https://github.com/laitselec/MuFun)
    * The statistics of the tags used to train the base model is shared on [Discord](https://discord.com/channels/1369256267645849741/1372633881215500429/1374037211145830442)
* When an audio is cropped because it's too long, also crop the lyrics
* I would not include BPM in the AI-generated tags, because it's much more accurate to detect BPM using traditional methods than AI. Also, to control the BPM of the generated audio, I guess it's more adhesive to use a control net than the prompt, similar to the Canny control net for images.
* Use [prodigy-plus-schedule-free](https://github.com/LoganBooker/prodigy-plus-schedule-free)








