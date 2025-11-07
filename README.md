这个版本来自于 https://github.com/woct0rdho/ACE-Step/ 这位大神的优化版本。
本人机器是win10系统，使用conda虚拟机。3080改20G显卡。
本人只是做了个gui界面，针对自己的机器环境做了些小调整能跑起来，很流畅的训练。 具体用法请看完下面流程
### 性能优化
- 启用梯度检查点以节省显存
- 全部使用 bf16 精度训练
- 支持在单张 RTX 3080（<10GB 显存）上运行



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

注意：当前gui有bug，从第3步开始不知道为什么会有重复训练的问题，目前没解决。3，4，5步骤不影响什么，但是第6步开始训练影响严重。可以暂时先用下面指令使用训练！我是懒得改了，哪位大神改好后麻烦告诉小弟一下~~~

第6步开始训练不适用gui界面，可以避免重复问题
python trainer_new.py --dataset_path D:\AIJOB\ACE-Step-T\ACE-Step\Taudio\1_prep --batch_size 4 --num_workers 0 --tag_dropout 0.5 --learning_rate 0.0001 --max_steps 5000 --precision bf16-mixed --save_every_n_train_steps 100


训练结束后模型会以文件夹的形势保存至项目根目录的 checkpoints，至于生成几个文件夹看你设置的多少步保存一次了。
原生的 ACE-Step 项目使用的lora模型是写死的用 huggingface 的模型，我改成了动态读取 checkpoints 下的模型了，你把你训练好的模型扔进文件夹去就行了。
替换 原生的 ACE-Step 项目的ui文件我也已经放到了项目里了。

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



