"""模型定义与LoRA配置

本模块提供：
- Qwen2.5-VL-7B模型的LoRA微调封装
- 支持多设备部署和量化
- 与MyStrategy的兼容接口
"""

import torch
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import warnings
import os


class QwenVLWithLoRA(nn.Module):
    """Qwen2.5-VL-7B的LoRA微调封装
    
    特点：
    - 支持LoRA高效微调
    - 兼容多设备部署
    - 支持量化模型
    - 与MyStrategy无缝集成
    """
    
    def __init__(
        self,
        model_name_or_path="Qwen/Qwen2.5-VL-7B-Instruct",
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=None,
        device_map="auto",
        torch_dtype=torch.float16,
        load_in_8bit=False,
        load_in_4bit=False,
        **kwargs
    ):
        super().__init__()
        
        self.model_name = model_name_or_path
        self.lora_config = None
        self.is_lora_applied = False
        
        # 设置LoRA目标模块（如果未指定，使用默认配置）
        if target_modules is None:
            target_modules = [
                # 语言模型部分
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
                
                # 视觉transformer部分
                "qkv", "proj",
                
                # 多模态融合部分
                "mlp.0", "mlp.2"
            ]
        
        # 检测是否是本地路径
        is_local_path = os.path.exists(model_name_or_path)
        
        if is_local_path:
            print(f"检测到本地模型路径: {model_name_or_path}")
            # 对于本地路径，强制使用local_files_only=True
            load_kwargs = {
                "device_map": device_map,
                "torch_dtype": torch_dtype,
                "load_in_8bit": load_in_8bit,
                "load_in_4bit": load_in_4bit,
                "local_files_only": True,  # 关键：强制使用本地文件
                **kwargs
            }
        else:
            print(f"使用Hugging Face Hub模型: {model_name_or_path}")
            load_kwargs = {
                "device_map": device_map,
                "torch_dtype": torch_dtype,
                "load_in_8bit": load_in_8bit,
                "load_in_4bit": load_in_4bit,
                **kwargs
            }
        
        # 配置量化（使用新的BitsAndBytesConfig方式）
        from transformers import BitsAndBytesConfig
        
        if load_in_8bit or load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
                llm_int8_enable_fp32_cpu_offload=True,
            )
            load_kwargs['quantization_config'] = quantization_config
            # 移除旧的参数
            load_kwargs.pop('load_in_8bit', None)
            load_kwargs.pop('load_in_4bit', None)
        
        # 加载基础模型
        self.backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            **load_kwargs
        )
        
        # 检查是否使用了多设备映射
        self.uses_device_map = hasattr(self.backbone, 'hf_device_map')
        
        # 配置LoRA
        self._setup_lora(lora_rank, lora_alpha, lora_dropout, target_modules)
        
        # 启用梯度检查点以节省显存
        if hasattr(self.backbone, 'gradient_checkpointing_enable'):
            self.backbone.gradient_checkpointing_enable()
            print("已启用梯度检查点")
        
        # 设置模型为训练模式
        self.backbone.train()
        
        print(f"模型加载完成，设备映射: {self.uses_device_map}")
        print(f"LoRA已应用: {self.is_lora_applied}")
    
    def _setup_lora(self, lora_rank, lora_alpha, lora_dropout, target_modules):
        """配置并应用LoRA"""
        try:
            # 创建LoRA配置
            self.lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            # 准备模型进行k-bit训练
            self.backbone = prepare_model_for_kbit_training(self.backbone)
            
            # 应用LoRA
            self.backbone = get_peft_model(self.backbone, self.lora_config)
            self.is_lora_applied = True
            
            print(f"LoRA配置成功: rank={lora_rank}, alpha={lora_alpha}")
            
        except Exception as e:
            print(f"LoRA配置失败: {e}")
            print("将使用原始模型进行训练")
            self.is_lora_applied = False
    
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        """前向传播
        
        参数：
            input_ids: 输入token ID
            attention_mask: 注意力掩码
            labels: 标签（用于计算损失）
            **kwargs: 其他参数（如图像输入）
        
        返回：
            transformers.modeling_outputs.CausalLMOutputWithPast
        """
        # 调用backbone的前向传播
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
        
        return outputs
    
    def generate(self, **kwargs):
        """生成文本
        
        参数：
            **kwargs: 生成参数（如max_new_tokens, temperature等）
        
        返回：
            生成的token序列
        """
        return self.backbone.generate(**kwargs)
    
    def parameters(self):
        """返回模型参数（用于优化器）"""
        return self.backbone.parameters()
    
    def named_parameters(self):
        """返回命名的模型参数"""
        return self.backbone.named_parameters()
    
    def state_dict(self):
        """返回模型状态字典"""
        return self.backbone.state_dict()
    
    def load_state_dict(self, state_dict, strict=True):
        """加载模型状态字典"""
        return self.backbone.load_state_dict(state_dict, strict=strict)
    
    def train(self, mode=True):
        """设置训练模式"""
        return self.backbone.train(mode)
    
    def eval(self):
        """设置评估模式"""
        return self.backbone.eval()
    
    def to(self, device):
        """移动模型到指定设备（仅在不使用device_map时）"""
        if not self.uses_device_map:
            return self.backbone.to(device)
        else:
            print("模型使用device_map，无需手动移动")
            return self
    
    def print_trainable_parameters(self):
        """打印可训练参数信息"""
        if self.is_lora_applied:
            self.backbone.print_trainable_parameters()
        else:
            total_params = sum(p.numel() for p in self.backbone.parameters())
            trainable_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
            print(f"总参数: {total_params:,}")
            print(f"可训练参数: {trainable_params:,}")
            print(f"可训练比例: {100 * trainable_params / total_params:.2f}%")
    
    def save_pretrained(self, save_directory, **kwargs):
        """保存模型"""
        if self.is_lora_applied:
            # 保存LoRA权重
            self.backbone.save_pretrained(save_directory, **kwargs)
        else:
            # 保存完整模型
            self.backbone.save_pretrained(save_directory, **kwargs)
    
    def get_device(self):
        """获取模型当前设备"""
        if self.uses_device_map:
            # 对于多设备映射，返回第一个参数所在的设备
            for param in self.backbone.parameters():
                if param.device.type != 'meta':
                    return param.device
            return torch.device('cpu')
        else:
            return next(self.backbone.parameters()).device


class QwenVLProcessor:
    """Qwen2.5-VL处理器封装
    
    简化处理器的使用，提供统一的接口
    """
    
    def __init__(self, model_name_or_path="Qwen/Qwen2.5-VL-7B-Instruct"):
        from transformers import AutoProcessor
        
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        self.tokenizer = self.processor.tokenizer
    
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        """应用聊天模板"""
        return self.processor.apply_chat_template(
            messages, 
            tokenize=tokenize, 
            add_generation_prompt=add_generation_prompt
        )
    
    def __call__(self, text=None, image_inputs=None, images=None, **kwargs):
        """处理输入"""
        # 兼容旧参数名 image_inputs -> images
        if images is None and image_inputs is not None:
            images = image_inputs
        return self.processor(
            text=text,
            images=images,
            **kwargs
        )
    
    def decode(self, token_ids, **kwargs):
        """解码token ID为文本"""
        return self.tokenizer.decode(token_ids, **kwargs)
    
    def encode(self, text, **kwargs):
        """编码文本为token ID"""
        return self.tokenizer.encode(text, **kwargs)


def create_model_and_processor(
    model_name="Qwen/Qwen2.5-VL-7B-Instruct",
    lora_rank=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=None,
    device_map="cuda:0",  # 改为具体设备，不要用 "auto"
    torch_dtype=torch.float16,
    **kwargs
):
    """创建模型和处理器的便捷函数
    
    参数：
        model_name: 预训练模型名称或路径
        lora_rank: LoRA秩
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        target_modules: LoRA目标模块
        device_map: 设备映射策略
        torch_dtype: 数据类型
        **kwargs: 其他模型参数
    
    返回：
        (model, processor): 模型和处理器元组
    """
    
    # 创建模型
    model = QwenVLWithLoRA(
        model_name_or_path=model_name,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        device_map=device_map,
        torch_dtype=torch_dtype,
        **kwargs
    )
    
    # 创建处理器
    processor = QwenVLProcessor(model_name)
    
    return model, processor


# 兼容性别名
QwenVLModel = QwenVLWithLoRA


if __name__ == "__main__":
    # 测试代码
    print("测试模型创建...")
    
    try:
        # 创建模型和处理器
        model, processor = create_model_and_processor(
            lora_rank=8,
            lora_alpha=16,
            device_map="cuda:0"
        )
        
        print("模型创建成功!")
        print(f"模型类型: {type(model)}")
        print(f"处理器类型: {type(processor)}")
        
        # 打印可训练参数
        model.print_trainable_parameters()
        
    except Exception as e:
        print(f"模型创建失败: {e}")
        print("请检查模型路径和依赖是否正确安装")
