@echo off
echo 设置环境变量...
set TORCHDYNAMO_DISABLE=1
set TORCH_COMPILE_DISABLE=1
set TORCHINDUCTOR_DISABLE=1
set PYTORCH_ALLOC_CONF=expandable_segments:True
set CUDA_DEVICE_ORDER=PCI_BUS_ID

echo 清除可能的旧精度设置...
set TORCH_ALLOW_TF32_CUBLAS=
set TORCH_ALLOW_TF32_CUDNN=

echo 激活conda环境...
call conda activate ace_step_t

echo 开始训练...
python trainer_new.py --dataset_path C:\data\audio_prep

echo 训练完成
pause
