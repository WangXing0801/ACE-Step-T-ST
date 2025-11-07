<img width="1889" height="880" alt="image" src="https://github.com/user-attachments/assets/24383ea6-27d0-4ca6-b259-55d3745add78" />这个版本来自于 https://github.com/woct0rdho/ACE-Step/ 这位大神的优化版本。
本人机器是win10系统，使用conda虚拟机。3080改20G显卡。
本人只是做了个gui界面，针对自己的机器环境做了些小调整能跑起来，很流畅的训练。 具体用法请看完下面流程
### 性能优化
- 启用梯度检查点以节省显存
- 全部使用 bf16 精度训练
- 支持在单张 RTX 3080（<10GB 显存）上运行



# ACE-Step 模型训练操作流程文档

## 📋 使用方法：

### 1. 上传音频文件

### 2. 生成提示词 

### 3. 生成歌词

### 4. 创建文件名数据集 

### 5. 音频预处理 

### 6. 开始训练 

注意：当前gui有bug，从第3步开始不知道为什么会有重复训练的问题，目前没解决。3，4，5步骤不影响什么，但是第6步开始训练影响严重。可以暂时先用下面指令使用训练！我是懒得改了，哪位大神改好后麻烦告诉小弟一下~~~

第6步开始训练不适用gui界面，可以避免重复问题
python trainer_new.py --dataset_path D:\AIJOB\ACE-Step-T\ACE-Step\Taudio\1_prep --batch_size 4 --num_workers 0 --tag_dropout 0.5 --learning_rate 0.0001 --max_steps 5000 --precision bf16-mixed --save_every_n_train_steps 100


训练结束后模型会以文件夹的形势保存至项目根目录的 checkpoints，至于生成几个文件夹看你设置的多少步保存一次了。
原生的 ACE-Step 项目使用的lora模型是写死的用 huggingface 的模型，我改成了动态读取 checkpoints 下的模型了，你把你训练好的模型扔进文件夹去就行了。
替换 原生的 ACE-Step 项目的ui文件我也已经放到了项目里了。


！！！请注意使时权重调整，如果默认用我的方式训练，使用时权重请设置在0.125，>=0.2会噪音满满！！！
<img width="1889" height="880" alt="image" src="https://github.com/user-attachments/assets/e08b1744-2c7e-4cc9-ae7a-7a50d57e66b8" />
训练好的lora文件夹名称可以更改
<img width="904" height="407" alt="image" src="https://github.com/user-attachments/assets/44fc710e-8d06-4dab-86b8-8aa23970e89d" />
<img width="1813" height="815" alt="image" src="https://github.com/user-attachments/assets/6828eb20-d221-4d0d-8864-12f1cf159f69" />




训练模型参数设置
--batch_size	1	批大小
--num_workers	0	DataLoader 使用的线程数
--tag_dropout	0.5	文本标签的 dropout 概率
--learning_rate	1e-4	学习率
--max_steps	2000	最大训练步数
--precision	"bf16-mixed"	混合精度训练设置
--save_every_n_train_steps	100	每多少步保存一次检查点



