#!/bin/bash

echo "=== 安装训练所需的Python包 ==="

# 激活环境
echo "激活myenv环境..."
source /opt/conda/bin/activate myenv

# 更新pip
echo "更新pip..."
pip install --upgrade pip

# 安装PyTorch (根据CUDA 12.8版本)
echo "安装PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装Transformers相关
echo "安装Transformers相关包..."
pip install transformers
pip install peft
pip install accelerate
pip install safetensors

# 安装图像处理
echo "安装图像处理包..."
pip install Pillow

# 安装数据处理
echo "安装数据处理包..."
pip install numpy
pip install scikit-learn
pip install pandas

# 安装监控工具
echo "安装监控工具..."
pip install tensorboard
pip install matplotlib
pip install seaborn

# 安装其他工具
echo "安装其他工具..."
pip install datasets
pip install tokenizers
pip install wandb  # 可选：实验跟踪
pip install tqdm   # 进度条

# 兼容性修复
echo "安装兼容性包..."
pip install "numpy<2.0"  # 修复TensorBoard兼容性

echo "=== 安装完成 ==="
echo "请运行以下命令验证安装："
echo "conda activate myenv"
echo "python -c \"import torch; import transformers; import peft; print('所有包安装成功!')\"" 