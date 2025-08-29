import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import json
import time
from PIL import Image
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset, DataLoader
from mystrategy import MyStrategy, BaselineStrategy
from mystrategy import SlidingWindowMeanStrategy
from torch.utils.tensorboard import SummaryWriter
from model import create_model_and_processor
import numpy as np
from sklearn.model_selection import train_test_split
import random

class ImageCaptionDataset(Dataset):
    """图片描述数据集（适配你的JSON格式）"""
    
    @staticmethod
    def convert_json_format(input_json_path, output_json_path):
        """转换JSON格式以适配当前代码（静态方法，避免重复代码）"""
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        converted_data = []
        
        for item in data:
            # 使用公共的静态方法提取数据
            image_path = ImageCaptionDataset._extract_image_path_static(item)
            caption = ImageCaptionDataset._extract_caption_static(item)
            
            if image_path and caption:
                # 使用公共的静态方法处理路径分隔符
                image_path = ImageCaptionDataset._process_path_separators(image_path)
                
                converted_data.append({
                    'image_path': image_path,
                    'caption': caption
                })
        
        # 保存转换后的数据
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, ensure_ascii=False, indent=2)
        
        print(f"转换完成，共 {len(converted_data)} 个样本")
        return converted_data
    
    @staticmethod
    def _extract_image_path_static(item):
        """从数据项中提取图片路径（静态方法）"""
        try:
            for message in item.get('messages', []):
                if message.get('role') == 'user':
                    for content in message.get('content', []):
                        if content.get('type') == 'image':
                            return content.get('image')
            return None
        except:
            return None
    
    @staticmethod
    def _extract_caption_static(item):
        """从数据项中提取描述（静态方法）"""
        try:
            return item.get('target', {}).get('description', '').strip()
        except:
            return ''
    
    @staticmethod
    def _process_path_separators(image_path):
        """处理路径分隔符（静态方法）"""
        return image_path.replace('\\', '/')
    
    def __init__(self, data_path, processor, max_length=512, split='train', train_indices=None, test_indices=None):
        self.processor = processor
        self.max_length = max_length
        self.split = split
        self.samples = []
        
        # 加载数据集
        if os.path.isdir(data_path):
            self._load_from_directory(data_path)
        else:
            self._load_from_json(data_path)
        
        # 根据split和indices过滤样本
        if split == 'train' and train_indices is not None:
            self.samples = [self.samples[i] for i in train_indices]
        elif split == 'test' and test_indices is not None:
            self.samples = [self.samples[i] for i in test_indices]
        
        print(f"{split.capitalize()}数据集加载完成，共 {len(self.samples)} 个样本")
        print("使用空间感知专家提示词模板")
    
    def _load_from_json(self, json_path):
        """从JSON文件加载数据（适配你的数据格式）"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                raise ValueError("JSON数据应该是列表格式")
            
            json_dir = os.path.dirname(os.path.abspath(json_path))
            valid_samples = 0
            
            for i, item in enumerate(data):
                try:
                    # 使用公共的静态方法提取数据
                    image_path = ImageCaptionDataset._extract_image_path_static(item)
                    if not image_path:
                        continue
                    
                    caption = ImageCaptionDataset._extract_caption_static(item)
                    if not caption:
                        continue
                    
                    # 处理路径
                    image_path = self._process_image_path(image_path, json_dir)
                    if not image_path:
                        continue
                    
                    # 验证和预处理
                    if self._validate_sample(image_path, caption):
                        self.samples.append({
                            'image_path': image_path,
                            'caption': caption
                        })
                        valid_samples += 1
                        
                except Exception as e:
                    print(f"处理第{i}项时出错: {e}")
                    continue
            
            print(f"成功加载 {valid_samples}/{len(data)} 个有效样本")
            
            if valid_samples == 0:
                raise ValueError("没有找到有效的训练样本")
                
        except Exception as e:
            print(f"加载JSON数据失败: {e}")
            raise
    
    def _load_from_directory(self, data_dir):
        """从目录加载数据（支持多种格式）"""
        try:
            # 支持的图片格式
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            
            # 查找所有图片文件
            image_files = []
            for root, dirs, files in os.walk(data_dir):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in image_extensions):
                        image_files.append(os.path.join(root, file))
            
            if not image_files:
                raise ValueError(f"在目录 {data_dir} 中没有找到图片文件")
            
            print(f"找到 {len(image_files)} 个图片文件")
            
            # 为每个图片生成默认描述（用于测试）
            for image_path in image_files:
                # 生成简单的文件名作为描述
                filename = os.path.splitext(os.path.basename(image_path))[0]
                caption = f"这是一张关于{filename}的图片"
                
                if self._validate_sample(image_path, caption):
                    self.samples.append({
                        'image_path': image_path,
                        'caption': caption
                    })
            
            print(f"成功加载 {len(self.samples)} 个样本")
            
        except Exception as e:
            print(f"从目录加载数据失败: {e}")
            raise
    
    def _extract_image_path(self, item):
        """从数据项中提取图片路径（实例方法，调用静态方法）"""
        return ImageCaptionDataset._extract_image_path_static(item)
    
    def _extract_caption(self, item):
        """从数据项中提取描述（实例方法，调用静态方法）"""
        return ImageCaptionDataset._extract_caption_static(item)
    
    def _process_image_path(self, image_path, json_dir):
        """处理图片路径"""
        try:
            # 使用公共的静态方法处理路径分隔符
            image_path = ImageCaptionDataset._process_path_separators(image_path)
            
            # 处理相对路径
            if not os.path.isabs(image_path):
                image_path = os.path.join(json_dir, image_path)
            
            return image_path
        except:
            return None
    
    def _validate_sample(self, image_path, caption):
        """验证样本有效性"""
        # 检查文件存在性
        if not os.path.exists(image_path):
            print(f"警告: 图片文件不存在: {image_path}")
            return False
        
        # 检查文件格式
        if not image_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            print(f"警告: 不支持的图片格式: {image_path}")
            return False
        
        # 检查描述长度
        if len(caption) < 5:
            print(f"警告: 描述过短: {caption[:20]}...")
            return False
        
        if len(caption) > 500:
            print(f"警告: 描述过长，截断到500字符")
            caption = caption[:500]
        
        return True
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载图片
        image = Image.open(sample['image_path']).convert('RGB')
        caption = sample['caption']
        
        # 从图片路径中提取任务名称
        task_name = self._extract_task_name_from_path(sample['image_path'])
        
        # 构建空间感知专家提示词
        spatial_prompt = f"""你是一个空间感知专家。请你观察图片中的内容，已知图中机械臂在完成{task_name}任务，请描述图片内容。"""
        
        # 构建多模态输入
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": spatial_prompt}
            ]
        }]
        
        # 应用聊天模板
        chat_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # 拼接完整文本
        full_text = chat_prompt + caption
        
        # 处理输入
        inputs = self.processor(
            text=[full_text],
            images=[image],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )

        # Debug (disabled): print raw (no truncation) vs used (max_length) lengths
        # try:
        #     raw_inputs = self.processor(
        #         text=[full_text],
        #         images=[image],
        #         return_tensors="pt",
        #         padding="longest",
        #         truncation=False
        #     )
        #     raw_len = int(raw_inputs.input_ids.size(-1))
        # except Exception:
        #     raw_len = -1
        # used_len = int(inputs.input_ids.size(-1))
        # is_truncated = (raw_len != -1 and raw_len > self.max_length)
        # if is_truncated or (idx % 200 == 0):
        #     print(f"[SeqLen] idx={idx} raw_len={raw_len} used_len={used_len} max_len={self.max_length} truncated={is_truncated} path={sample['image_path']}")
        
        # 仅在 caption 段计算损失：屏蔽 PAD 与前缀为 -100
        labels = inputs.input_ids.squeeze(0).clone()
        attention = inputs.attention_mask.squeeze(0)
        # 屏蔽 PAD
        labels[attention == 0] = -100
        # 计算前缀长度（chat_prompt + "\nassistant\n"），并屏蔽前缀部分
        prefix_text = chat_prompt + "\nassistant\n"
        prefix_ids = self.processor(
            text=[prefix_text],
            images=[image],
            return_tensors="pt",
            padding="longest",   # 仅用于估计前缀长度，避免整段 max_length
            truncation=True,
            max_length=self.max_length
        ).input_ids.squeeze(0)
        prefix_len = min(prefix_ids.size(0), labels.size(0))
        labels[:prefix_len] = -100
        
        # 提取可选的视觉辅助键（兼容不同返回类型）
        def _opt(key):
            try:
                v = getattr(inputs, key) if hasattr(inputs, key) else None
            except Exception:
                v = None
            if v is None:
                try:
                    v = inputs[key] if (hasattr(inputs, '__contains__') and key in inputs) else None
                except Exception:
                    v = None
            return v
        image_grid_thw_val = _opt('image_grid_thw')
        image_sizes_val = _opt('image_sizes')
        image_input_size_val = _opt('image_input_size')
        
        # 转为Tensor（如可能）
        def to_tensor_safe(v):
            if v is None:
                return None
            import torch as _torch
            if isinstance(v, _torch.Tensor):
                return v
            try:
                return _torch.as_tensor(v)
            except Exception:
                return None
        image_grid_thw_val = to_tensor_safe(image_grid_thw_val)
        image_sizes_val = to_tensor_safe(image_sizes_val)
        image_input_size_val = to_tensor_safe(image_input_size_val)
        # 去掉前导的 batch 维，保证形状为 (3,) / (2,)
        for name, val in (('image_grid_thw', image_grid_thw_val), ('image_sizes', image_sizes_val), ('image_input_size', image_input_size_val)):
            if isinstance(val, torch.Tensor) and val.dim() >= 2 and val.size(0) == 1:
                if name == 'image_grid_thw':
                    image_grid_thw_val = val.squeeze(0)
                elif name == 'image_sizes':
                    image_sizes_val = val.squeeze(0)
                else:
                    image_input_size_val = val.squeeze(0)
        
        result = {
            'input_ids': inputs.input_ids.squeeze(0),
            'attention_mask': inputs.attention_mask.squeeze(0),
            'labels': labels,
            'pixel_values': (inputs.pixel_values.squeeze(0) if hasattr(inputs, 'pixel_values') else inputs["pixel_values"].squeeze(0)),
            'image_path': sample['image_path']
        }
        if image_grid_thw_val is not None:
            result['image_grid_thw'] = image_grid_thw_val
        if image_sizes_val is not None:
            result['image_sizes'] = image_sizes_val
        if image_input_size_val is not None:
            result['image_input_size'] = image_input_size_val
        
        return result
    
    def _extract_task_name_from_path(self, image_path):
        """从图片路径中提取任务名称"""
        try:
            # 获取图片所在的目录名（通常是任务名称）
            # 例如：/root/autodl-tmp/testset/327/650403/Place the held carambola into the shopping cart's plastic bag_/00085.jpg
            # 任务名称是：Place the held carambola into the shopping cart's plastic bag
            
            # 分割路径
            path_parts = image_path.split('/')
            
            # 查找包含任务名称的目录
            # 通常任务名称在倒数第二个目录中
            if len(path_parts) >= 2:
                # 尝试倒数第二个目录
                potential_task_name = path_parts[-2]
                
                # 如果倒数第二个目录看起来像任务名称（包含空格和描述性词汇）
                if ' ' in potential_task_name and len(potential_task_name) > 10:
                    return potential_task_name
                
                # 如果倒数第二个目录不是，尝试倒数第三个
                if len(path_parts) >= 3:
                    potential_task_name = path_parts[-3]
                    if ' ' in potential_task_name and len(potential_task_name) > 10:
                        return potential_task_name
            
            # 如果都没找到，返回默认任务名称
            return "机械臂操作任务"
            
        except Exception as e:
            print(f"提取任务名称时出错: {e}")
            return "机械臂操作任务"

def create_train_test_datasets(data_path, processor, max_length=512, test_size=0.2, random_state=42):
    """创建训练集和测试集"""
    print("创建训练集和测试集...")
    
    # 首先加载所有数据
    temp_dataset = ImageCaptionDataset(data_path, processor, max_length)
    all_samples = temp_dataset.samples
    
    # 设置随机种子确保可重复性
    random.seed(random_state)
    np.random.seed(random_state)
    
    # 分割数据
    indices = list(range(len(all_samples)))
    train_indices, test_indices = train_test_split(
        indices, 
        test_size=test_size, 
        random_state=random_state,
        shuffle=True
    )
    
    print(f"数据集分割完成:")
    print(f"  总样本数: {len(all_samples)}")
    print(f"  训练集: {len(train_indices)} 样本")
    print(f"  测试集: {len(test_indices)} 样本")
    print(f"  分割比例: 训练集 {1-test_size:.1%}, 测试集 {test_size:.1%}")
    
    # 创建训练集和测试集
    train_dataset = ImageCaptionDataset(data_path, processor, max_length, 'train', train_indices, None)
    test_dataset = ImageCaptionDataset(data_path, processor, max_length, 'test', None, test_indices)
    
    return train_dataset, test_dataset

def create_datasets_with_val_from_test(data_path, processor, max_length=512, test_size=0.2, val_ratio=0.5, random_state=42):
    """创建训练集、验证集和测试集（从测试集中分割验证集）
    
    Args:
        data_path: 数据路径
        processor: 处理器
        max_length: 最大序列长度
        test_size: 原始测试集比例
        val_ratio: 从测试集中分割验证集的比例
        random_state: 随机种子
    
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    print("创建训练集、验证集和测试集（从测试集中分割验证集）...")
    
    # 首先加载所有数据
    temp_dataset = ImageCaptionDataset(data_path, processor, max_length)
    all_samples = temp_dataset.samples
    
    # 设置随机种子确保可重复性
    random.seed(random_state)
    np.random.seed(random_state)
    
    # 首先按照原有的方式分割训练集和测试集
    indices = list(range(len(all_samples)))
    train_indices, test_indices = train_test_split(
        indices, 
        test_size=test_size, 
        random_state=random_state,
        shuffle=True
    )
    
    # 从测试集中分割出验证集
    val_size_from_test = int(len(test_indices) * val_ratio)
    val_indices, final_test_indices = train_test_split(
        test_indices,
        test_size=val_size_from_test,
        random_state=random_state,
        shuffle=True
    )
    
    print(f"数据集分割完成:")
    print(f"  总样本数: {len(all_samples)}")
    print(f"  训练集: {len(train_indices)} 样本 ({len(train_indices)/len(all_samples):.1%})")
    print(f"  验证集: {len(val_indices)} 样本 ({len(val_indices)/len(all_samples):.1%})")
    print(f"  测试集: {len(final_test_indices)} 样本 ({len(final_test_indices)/len(all_samples):.1%})")
    print(f"  分割方式: 训练集 {1-test_size:.1%}, 原测试集 {test_size:.1%}")
    print(f"  验证集占比: 原测试集的 {val_ratio:.1%}")
    
    # 创建三个数据集
    train_dataset = ImageCaptionDataset(data_path, processor, max_length, 'train', train_indices, None)
    val_dataset = ImageCaptionDataset(data_path, processor, max_length, 'val', None, val_indices)
    test_dataset = ImageCaptionDataset(data_path, processor, max_length, 'test', None, final_test_indices)
    
    return train_dataset, val_dataset, test_dataset

def create_datasets(data_path, processor, max_length=512, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    """创建训练集、验证集和测试集"""
    print("创建训练集、验证集和测试集...")
    
    # 首先加载所有数据
    temp_dataset = ImageCaptionDataset(data_path, processor, max_length)
    all_samples = temp_dataset.samples
    
    # 设置随机种子确保可重复性
    random.seed(random_state)
    np.random.seed(random_state)
    
    # 分割数据
    indices = list(range(len(all_samples)))
    train_indices, test_indices = train_test_split(
        indices, 
        test_size=test_size, 
        random_state=random_state,
        shuffle=True
    )
    
    # 从训练集中分割出验证集
    val_size_from_train = int(len(train_indices) * val_size)
    val_indices, final_train_indices = train_test_split(
        train_indices,
        test_size=val_size_from_train,
        random_state=random_state,
        shuffle=True
    )
    
    print(f"数据集分割完成:")
    print(f"  总样本数: {len(all_samples)}")
    print(f"  训练集: {len(final_train_indices)} 样本 ({len(final_train_indices)/len(all_samples):.1%})")
    print(f"  验证集: {len(val_indices)} 样本 ({len(val_indices)/len(all_samples):.1%})")
    print(f"  测试集: {len(test_indices)} 样本 ({len(test_indices)/len(all_samples):.1%})")
    print(f"  分割方式: 训练集 {1-test_size:.1%}, 验证集 {val_size:.1%}, 测试集 {test_size:.1%}")
    
    # 创建三个数据集
    train_dataset = ImageCaptionDataset(data_path, processor, max_length, 'train', final_train_indices, None)
    val_dataset = ImageCaptionDataset(data_path, processor, max_length, 'val', None, val_indices)
    test_dataset = ImageCaptionDataset(data_path, processor, max_length, 'test', None, test_indices)
    
    return train_dataset, val_dataset, test_dataset

class MyStrategyTrainer:
    """使用MyStrategy的训练器"""
    
    def __init__(self, model, processor, args):
        self.model = model
        self.processor = processor
        self.args = args
        self.device = next(model.parameters()).device
        
        # 创建损失函数和优化器
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # 创建MyStrategy
        self.strategy = MyStrategy(
            model=model,
            criterion=self.criterion,
            optimizer=self.optimizer,
            window_size=args.window_size,
            loss_threshold=args.loss_threshold,
            trend_threshold=args.trend_threshold,
            vol_threshold=args.vol_threshold,
            outlier_threshold=args.outlier_threshold,
            save_outlier_log=True,
            outlier_log_path=os.path.join(args.output_dir, 'outlier_log.json')
        )
        
        # 创建TensorBoard writer - 使用更专业的目录结构
        tensorboard_dir = os.path.join(args.output_dir, 'tensorboard')
        os.makedirs(tensorboard_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=tensorboard_dir)
        
        # 添加累积步数计数器
        self.cumulative_training_steps = 0
        
        # 添加数组存储
        self.epoch_losses = []
        self.epoch_discard_rates = []
        self.epoch_test_losses = []  # 新增：测试集损失
        self.epoch_test_accuracies = []  # 新增：测试集准确率
        self.batch_losses = []
        self.batch_discard_rates = []
        
        # 新增：智能记录控制
        self.total_epochs = args.epochs
        self.record_frequency = self._calculate_record_frequency()
        self.last_recorded_step = {}
        
        # 新增：时间跟踪
        self.training_start_time = time.time()
        self.last_time_record = self.training_start_time
    
    def _calculate_record_frequency(self):
        """根据epoch数量智能计算记录频率"""
        if self.total_epochs <= 10:
            # 10个epoch以下：每个epoch记录
            return {'epoch': 1, 'batch': 1}
        elif self.total_epochs <= 50:
            # 50个epoch以下：每2个epoch记录一次，batch每5步记录
            return {'epoch': 2, 'batch': 5}
        elif self.total_epochs <= 100:
            # 100个epoch以下：每5个epoch记录一次，batch每10步记录
            return {'epoch': 5, 'batch': 10}
        else:
            # 100个epoch以上：每10个epoch记录一次，batch每20步记录
            return {'epoch': 10, 'batch': 20}
    
    def _should_record(self, metric_type, current_value):
        """判断是否应该记录当前指标"""
        if metric_type == 'epoch':
            return current_value % self.record_frequency['epoch'] == 0
        elif metric_type == 'batch':
            return current_value % self.record_frequency['batch'] == 0
        return True
    
    def train_epoch(self, train_loader, epoch, test_loader=None):
        """训练一个epoch"""
        self.model.train()
        self.strategy.set_epoch(epoch)
        
        total_loss = 0
        total_batches = 0
        discarded_batches = 0
        
        # 添加有效训练步数计数器
        self.effective_training_steps = getattr(self, 'effective_training_steps', 0)
        
        for batch_idx, batch in enumerate(train_loader):
            # 将数据移到设备
            inputs = {
                'input_ids': batch['input_ids'].to(self.device),
                'attention_mask': batch['attention_mask'].to(self.device),
                'labels': batch['labels'].to(self.device),
                'pixel_values': batch['pixel_values'].to(self.device)
            }
            # 兼容可选图像辅助键
            for opt_key in ('image_grid_thw','image_sizes','image_input_size'):
                if opt_key in batch and batch[opt_key] is not None:
                    val = batch[opt_key]
                    inputs[opt_key] = val.to(self.device) if hasattr(val, 'to') else val
            image_path = batch.get('image_path') if isinstance(batch, dict) else None
            
            # 使用MyStrategy训练
            loss, repeat_count, is_discarded = self.strategy.train_batch(
                inputs,
                inputs['labels'],
                image_path=image_path
            )
            
            if is_discarded:
                discarded_batches += 1
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}: 丢弃（异常检测）")
                
                # 记录丢弃决策到TensorBoard（使用总步数）
                self.writer.add_scalar('TrainingStep/Strategy_Decision/策略决策行为', 0, self.cumulative_training_steps)
                
                # 只记录丢弃标记
                self.batch_discard_rates.append(1.0)
            else:
                total_loss += loss
                total_batches += 1
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}: 损失={loss:.4f}, 训练次数={repeat_count}")
                
                # 记录有效训练的loss（使用有效步数）
                self.writer.add_scalar('TrainingStep/Train_Loss/训练过程监控', loss, self.effective_training_steps)
                # 额外记录：以总步数为横轴的loss（便于与Baseline按总步数对齐比较）
                self.writer.add_scalar('TrainingStep_Total/Train_Loss_总步数坐标', loss, self.cumulative_training_steps)
                
                # 新增：记录时间维度的loss
                current_time = time.time()
                elapsed_time = current_time - self.training_start_time
                self.writer.add_scalar('TimeBased/Train_Loss_时间坐标', loss, elapsed_time)
                
                # 记录策略决策行为（使用总步数）
                self.writer.add_scalar('TrainingStep/Strategy_Decision/策略决策行为', 1, self.cumulative_training_steps)
                
                # 记录重复训练次数（使用有效步数）
                self.writer.add_scalar('TrainingStep/Repeat_Count/ABR策略决策行为', repeat_count, self.effective_training_steps)
                
                # 更新有效训练步数
                self.effective_training_steps += 1
                
                # 记录批次数组
                self.batch_losses.append(loss)
                self.batch_discard_rates.append(0.0)
            
            # 更新总步数
            self.cumulative_training_steps += 1
            
            # 记录epoch信息
            self.writer.add_scalar('Progress/Epoch', epoch, self.cumulative_training_steps)
            
            # 记录epoch边界
            if batch_idx == 0:
                self.writer.add_scalar('Progress/Epoch_Start', epoch, self.cumulative_training_steps)
            
            # 限制训练批次数（用于快速测试）
            if self.args.max_train_batches > 0 and batch_idx >= self.args.max_train_batches - 1:
                break
        
        # 计算平均损失
        avg_loss = total_loss / max(1, total_batches)
        discard_rate = discarded_batches / max(1, len(train_loader))
        
        # 在测试集上评估
        test_loss = None
        test_accuracy = None
        if test_loader is not None:
            test_loss, test_accuracy = self.evaluate(test_loader)
            self.epoch_test_losses.append(test_loss)
            self.epoch_test_accuracies.append(test_accuracy)
            print(f"Epoch {epoch+1} 测试集损失: {test_loss:.4f}, 准确率: {test_accuracy:.4f}")
            
            # 记录测试准确率到TensorBoard - 使用类似run_parallel_grid_search.py的方式
            self.writer.add_scalar('Epoch/Test_Accuracy/最终主模型性能', test_accuracy, epoch)
            self.writer.add_scalar('Epoch/Test_Loss/判断模型过拟合风险', test_loss, epoch)
            
            # 新增：记录时间维度的测试loss
            current_time = time.time()
            elapsed_time = current_time - self.training_start_time
            self.writer.add_scalar('TimeBased/Test_Loss_时间坐标', test_loss, elapsed_time)
            self.writer.add_scalar('TimeBased/Test_Accuracy_时间坐标', test_accuracy, elapsed_time)
        
        # 记录epoch级别的指标到TensorBoard
        self.writer.add_scalar('Epoch/Train_Loss/训练损失监控', avg_loss, epoch)
        self.writer.add_scalar('Epoch/Discard_Rate/策略丢弃率', discard_rate, epoch)
        self.writer.add_scalar('Epoch/Cumulative_Steps/累计算力消耗对比', self.cumulative_training_steps, epoch)
        self.writer.add_scalar('Progress/Epoch_Complete', 1.0, epoch)
        self.writer.add_scalar('Progress/Epoch_Boundary', epoch + 1, epoch)
        
        # 记录进度百分比
        progress = (epoch + 1) / self.total_epochs
        self.writer.add_scalar('Progress/Epoch_Progress', progress, epoch)
        
        # 记录相对进度（0-1之间）
        relative_progress = epoch / max(1, self.total_epochs - 1)
        self.writer.add_scalar('Progress/Relative_Progress', relative_progress, epoch)
        
        # 记录epoch级别的指标
        self.epoch_losses.append(avg_loss)
        self.epoch_discard_rates.append(discard_rate)
        
        # 打印统计信息
        self.strategy.print_statistics()
        
        return avg_loss, discard_rate, test_loss, test_accuracy
    
    def evaluate(self, eval_loader):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        total_batches = 0
        total_correct = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in eval_loader:
                inputs = {
                    'input_ids': batch['input_ids'].to(self.device),
                    'attention_mask': batch['attention_mask'].to(self.device),
                    'labels': batch['labels'].to(self.device),
                    'pixel_values': batch['pixel_values'].to(self.device)
                }
                # 兼容可选图像辅助键
                for opt_key in ('image_grid_thw','image_sizes','image_input_size'):
                    if opt_key in batch and batch[opt_key] is not None:
                        val = batch[opt_key]
                        inputs[opt_key] = val.to(self.device) if hasattr(val, 'to') else val
                
                outputs = self.model(**inputs)
                loss = outputs.logits
                
                # 计算损失
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
                loss_per_token = loss_fct(loss.view(-1, loss.size(-1)), inputs['labels'].view(-1))
                total_loss += loss_per_token.sum().item()
                total_batches += 1
                
                # 计算准确率
                predictions = loss.argmax(dim=-1)
                labels = inputs['labels']
                
                # 只计算非忽略的token
                non_ignore_mask = labels != -100
                correct_predictions = (predictions == labels) & non_ignore_mask
                total_correct += correct_predictions.sum().item()
                total_tokens += non_ignore_mask.sum().item()
        
        avg_loss = total_loss / max(1, total_tokens)
        accuracy = total_correct / max(1, total_tokens)
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch, loss):
        """保存检查点"""
        checkpoint_path = os.path.join(self.args.output_dir, f'checkpoint_epoch_{epoch+1}.pt')
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'strategy_state': {
                'loss_window': list(self.strategy.loss_window),
                'batch_history': self.strategy.batch_history,
                'outlier_log': self.strategy.outlier_log
            }
        }, checkpoint_path)
        
        print(f"检查点已保存到: {checkpoint_path}")
    
    def close(self):
        """关闭资源"""
        self.writer.close()
        self.strategy.save_final_logs()
        
        # 保存NumPy结果
        self.save_numpy_results()

    def save_numpy_results(self):
        """保存NumPy结果用于对比分析"""
        results_dir = os.path.join(self.args.output_dir, 'mystrategy_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # 保存epoch级别的指标
        np.save(os.path.join(results_dir, 'epoch_losses.npy'), np.array(self.epoch_losses))
        np.save(os.path.join(results_dir, 'epoch_discard_rates.npy'), np.array(self.epoch_discard_rates))
        np.save(os.path.join(results_dir, 'epoch_test_losses.npy'), np.array(self.epoch_test_losses))
        np.save(os.path.join(results_dir, 'epoch_test_accuracies.npy'), np.array(self.epoch_test_accuracies))
        
        # 保存batch级别的指标
        np.save(os.path.join(results_dir, 'batch_losses.npy'), np.array(self.batch_losses))
        np.save(os.path.join(results_dir, 'batch_discard_rates.npy'), np.array(self.batch_discard_rates))
        
        # 保存策略的NumPy数据
        if hasattr(self.strategy, 'save_numpy_data'):
            self.strategy.save_numpy_data(self.args.output_dir)
        
        print(f"MyStrategy NumPy结果已保存到: {results_dir}")

class BaselineStrategyTrainer:
    """使用BaselineStrategy的训练器"""
    
    def __init__(self, model, processor, args):
        self.model = model
        self.processor = processor
        self.args = args
        self.device = next(model.parameters()).device
        
        # 创建损失函数和优化器
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # 创建BaselineStrategy
        from mystrategy import BaselineStrategy
        self.strategy = BaselineStrategy(
            model=model,
            criterion=self.criterion,
            optimizer=self.optimizer
        )
        
        # 创建TensorBoard writer - 使用更专业的目录结构
        tensorboard_dir = os.path.join(args.output_dir, 'tensorboard')
        os.makedirs(tensorboard_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=tensorboard_dir)
        
        # 添加累积步数计数器
        self.cumulative_training_steps = 0
        
        # 添加数组存储
        self.epoch_losses = []
        self.epoch_val_losses = []  # 新增：验证集损失
        self.epoch_val_accuracies = []  # 新增：验证集准确率
        self.epoch_test_losses = []  # 新增：测试集损失
        self.epoch_test_accuracies = []  # 新增：测试集准确率
        self.batch_losses = []
        
        # 新增：智能记录控制（与MyStrategy保持一致）
        self.total_epochs = args.epochs
        self.record_frequency = self._calculate_record_frequency()
        self.last_recorded_step = {}
        
        # 新增：时间跟踪
        self.training_start_time = time.time()
        self.last_time_record = self.training_start_time
    
    def _calculate_record_frequency(self):
        """根据epoch数量智能计算记录频率（与MyStrategy保持一致）"""
        if self.total_epochs <= 10:
            # 10个epoch以下：每个epoch记录
            return {'epoch': 1, 'batch': 1}
        elif self.total_epochs <= 50:
            # 50个epoch以下：每2个epoch记录一次，batch每5步记录
            return {'epoch': 2, 'batch': 5}
        elif self.total_epochs <= 100:
            # 100个epoch以下：每5个epoch记录一次，batch每10步记录
            return {'epoch': 5, 'batch': 10}
        else:
            # 100个epoch以上：每10个epoch记录一次，batch每20步记录
            return {'epoch': 10, 'batch': 20}
    
    def _should_record(self, metric_type, current_value):
        """判断是否应该记录当前指标（与MyStrategy保持一致）"""
        if metric_type == 'epoch':
            return current_value % self.record_frequency['epoch'] == 0
        elif metric_type == 'batch':
            return current_value % self.record_frequency['batch'] == 0
        return True
    
    def train_epoch(self, train_loader, epoch, val_loader=None, test_loader=None):
        """训练一个epoch"""
        self.model.train()
        self.strategy.set_epoch(epoch)
        
        total_loss = 0
        total_batches = 0
        
        # 添加有效训练步数计数器（Baseline总是有效训练）
        self.effective_training_steps = getattr(self, 'effective_training_steps', 0)
        
        for batch_idx, batch in enumerate(train_loader):
            # 将数据移到设备
            inputs = {
                'input_ids': batch['input_ids'].to(self.device),
                'attention_mask': batch['attention_mask'].to(self.device),
                'labels': batch['labels'].to(self.device),
                'pixel_values': batch['pixel_values'].to(self.device)
            }
            # 兼容可选图像辅助键
            for opt_key in ('image_grid_thw','image_sizes','image_input_size'):
                if opt_key in batch and batch[opt_key] is not None:
                    val = batch[opt_key]
                    inputs[opt_key] = val.to(self.device) if hasattr(val, 'to') else val
            image_path = batch.get('image_path') if isinstance(batch, dict) else None
            
            # 使用BaselineStrategy训练
            loss, repeat_count, is_discarded = self.strategy.train_batch(
                inputs,
                inputs['labels'],
                image_path=image_path
            )
            
            total_loss += loss
            total_batches += 1
            print(f"Epoch {epoch+1}, Batch {batch_idx+1}: 损失={loss:.4f}")
            
            # 记录训练损失到TensorBoard - 使用类似run_parallel_grid_search.py的方式
            self.writer.add_scalar('TrainingStep/Train_Loss/训练过程监控', loss, self.effective_training_steps)
            # 额外记录：以总步数为横轴的loss
            self.writer.add_scalar('TrainingStep_Total/Train_Loss_总步数坐标', loss, self.cumulative_training_steps)
            
            # 新增：记录时间维度的loss
            current_time = time.time()
            elapsed_time = current_time - self.training_start_time
            self.writer.add_scalar('TimeBased/Train_Loss_时间坐标', loss, elapsed_time)
            
            # 记录策略决策行为（Baseline总是训练）
            self.writer.add_scalar('TrainingStep/Strategy_Decision/策略决策行为', 1, self.cumulative_training_steps)
            
            # 记录重复训练次数（Baseline总是1）
            self.writer.add_scalar('TrainingStep/Repeat_Count/ABR策略决策行为', repeat_count, self.effective_training_steps)
            
            # 更新有效训练步数（Baseline总是有效训练）
            self.effective_training_steps += 1
            
            # 记录批次数组
            self.batch_losses.append(loss)
            
            # 更新累积步数
            self.cumulative_training_steps += 1
            
            # 记录epoch信息
            self.writer.add_scalar('Progress/Epoch', epoch, self.cumulative_training_steps)
            
            # 记录epoch边界
            if batch_idx == 0:
                self.writer.add_scalar('Progress/Epoch_Start', epoch, self.cumulative_training_steps)
            
            # 限制训练批次数（用于快速测试）
            if self.args.max_train_batches > 0 and batch_idx >= self.args.max_train_batches - 1:
                break
        
        # 计算平均损失
        avg_loss = total_loss / max(1, total_batches)
        
        # 在验证集上评估
        val_loss = None
        val_accuracy = None
        if val_loader is not None:
            val_loss, val_accuracy = self.evaluate(val_loader)
            self.epoch_val_losses.append(val_loss)
            self.epoch_val_accuracies.append(val_accuracy)
            print(f"Epoch {epoch+1} 验证集损失: {val_loss:.4f}, 准确率: {val_accuracy:.4f}")
            
            # 记录验证准确率到TensorBoard
            self.writer.add_scalar('Epoch/Val_Accuracy/验证集性能', val_accuracy, epoch)
            self.writer.add_scalar('Epoch/Val_Loss/验证集损失', val_loss, epoch)
            
            # 新增：记录时间维度的验证loss
            current_time = time.time()
            elapsed_time = current_time - self.training_start_time
            self.writer.add_scalar('TimeBased/Val_Loss_时间坐标', val_loss, elapsed_time)
            self.writer.add_scalar('TimeBased/Val_Accuracy_时间坐标', val_accuracy, elapsed_time)
        
        # 在测试集上评估
        test_loss = None
        test_accuracy = None
        if test_loader is not None:
            test_loss, test_accuracy = self.evaluate(test_loader)
            self.epoch_test_losses.append(test_loss)
            self.epoch_test_accuracies.append(test_accuracy)
            print(f"Epoch {epoch+1} 测试集损失: {test_loss:.4f}, 准确率: {test_accuracy:.4f}")
            
            # 记录测试准确率到TensorBoard
            self.writer.add_scalar('Epoch/Test_Accuracy/最终主模型性能', test_accuracy, epoch)
            self.writer.add_scalar('Epoch/Test_Loss/判断模型过拟合风险', test_loss, epoch)
            
            # 新增：记录时间维度的测试loss
            current_time = time.time()
            elapsed_time = current_time - self.training_start_time
            self.writer.add_scalar('TimeBased/Test_Loss_时间坐标', test_loss, elapsed_time)
            self.writer.add_scalar('TimeBased/Test_Accuracy_时间坐标', test_accuracy, elapsed_time)
        
        # 记录epoch级别的指标到TensorBoard
        self.writer.add_scalar('Epoch/Train_Loss/训练损失监控', avg_loss, epoch)
        self.writer.add_scalar('Epoch/Cumulative_Steps/累计算力消耗对比', self.cumulative_training_steps, epoch)
        self.writer.add_scalar('Progress/Epoch_Complete', 1.0, epoch)
        self.writer.add_scalar('Progress/Epoch_Boundary', epoch + 1, epoch)
        
        # 记录进度百分比
        progress = (epoch + 1) / self.total_epochs
        self.writer.add_scalar('Progress/Epoch_Progress', progress, epoch)
        
        # 记录相对进度（0-1之间）
        relative_progress = epoch / max(1, self.total_epochs - 1)
        self.writer.add_scalar('Progress/Relative_Progress', relative_progress, epoch)
        
        # 记录epoch级别的指标
        self.epoch_losses.append(avg_loss)
        
        # 打印统计信息
        self.strategy.print_statistics()
        
        return avg_loss, 0.0, val_loss, val_accuracy, test_loss, test_accuracy
    
    def evaluate(self, eval_loader):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        total_batches = 0
        total_correct = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in eval_loader:
                inputs = {
                    'input_ids': batch['input_ids'].to(self.device),
                    'attention_mask': batch['attention_mask'].to(self.device),
                    'labels': batch['labels'].to(self.device),
                    'pixel_values': batch['pixel_values'].to(self.device)
                }
                # 兼容可选图像辅助键
                for opt_key in ('image_grid_thw','image_sizes','image_input_size'):
                    if opt_key in batch and batch[opt_key] is not None:
                        val = batch[opt_key]
                        inputs[opt_key] = val.to(self.device) if hasattr(val, 'to') else val
                
                outputs = self.model(**inputs)
                loss = outputs.logits
                
                # 计算损失
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
                loss_per_token = loss_fct(loss.view(-1, loss.size(-1)), inputs['labels'].view(-1))
                total_loss += loss_per_token.sum().item()
                total_batches += 1
                
                # 计算准确率
                predictions = loss.argmax(dim=-1)
                labels = inputs['labels']
                
                # 只计算非忽略的token
                non_ignore_mask = labels != -100
                correct_predictions = (predictions == labels) & non_ignore_mask
                total_correct += correct_predictions.sum().item()
                total_tokens += non_ignore_mask.sum().item()
        
        avg_loss = total_loss / max(1, total_tokens)
        accuracy = total_correct / max(1, total_tokens)
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch, loss):
        """保存检查点"""
        checkpoint_path = os.path.join(self.args.output_dir, f'checkpoint_epoch_{epoch+1}.pt')
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'strategy_state': {
                'batch_history': self.strategy.batch_history,
            }
        }, checkpoint_path)
        
        print(f"检查点已保存到: {checkpoint_path}")
    
    def close(self):
        """关闭资源"""
        self.writer.close()
        self.strategy.save_final_logs()
        
        # 保存NumPy结果
        self.save_numpy_results()

    def save_numpy_results(self):
        """保存NumPy结果用于对比分析"""
        results_dir = os.path.join(self.args.output_dir, 'baseline_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # 保存epoch级别的指标
        np.save(os.path.join(results_dir, 'epoch_losses.npy'), np.array(self.epoch_losses))
        np.save(os.path.join(results_dir, 'epoch_val_losses.npy'), np.array(self.epoch_val_losses))
        np.save(os.path.join(results_dir, 'epoch_val_accuracies.npy'), np.array(self.epoch_val_accuracies))
        np.save(os.path.join(results_dir, 'epoch_test_losses.npy'), np.array(self.epoch_test_losses))
        np.save(os.path.join(results_dir, 'epoch_test_accuracies.npy'), np.array(self.epoch_test_accuracies))
        
        # 保存batch级别的指标
        np.save(os.path.join(results_dir, 'batch_losses.npy'), np.array(self.batch_losses))
        
        # 保存策略的NumPy数据
        if hasattr(self.strategy, 'save_numpy_data'):
            self.strategy.save_numpy_data(self.args.output_dir)
        
        print(f"Baseline NumPy结果已保存到: {results_dir}")

class SlidingWindowMeanStrategyTrainer:
    """使用SlidingWindowMeanStrategy的训练器"""
    
    def __init__(self, model, processor, args):
        self.model = model
        self.processor = processor
        self.args = args
        self.device = next(model.parameters()).device
        
        # 创建损失函数和优化器
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # 延迟导入SlidingWindowMeanStrategy
        try:
            from mystrategy import SlidingWindowMeanStrategy
            self.strategy = SlidingWindowMeanStrategy(
                model=model,
                criterion=self.criterion,
                optimizer=self.optimizer,
                window_size=args.window_size,
                threshold=args.threshold,
                save_outlier_log=True,
                outlier_log_path=os.path.join(args.output_dir, 'sliding_window_outlier_log.json')
            )
        except ImportError as e:
            print(f"导入SlidingWindowMeanStrategy失败: {e}")
            print("请检查mystrategy.py文件中的类定义")
            raise
        
        # 创建TensorBoard writer - 使用更专业的目录结构
        tensorboard_dir = os.path.join(args.output_dir, 'tensorboard')
        os.makedirs(tensorboard_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=tensorboard_dir)
        
        # 添加累积步数计数器
        self.cumulative_training_steps = 0
        
        # 添加数组存储
        self.epoch_losses = []
        self.epoch_discard_rates = []
        self.epoch_val_losses = []  # 新增：验证集指标列表，避免在train_epoch追加时报AttributeError
        self.epoch_val_accuracies = []
        self.epoch_test_losses = []  # 新增：测试集损失
        self.epoch_test_accuracies = []  # 新增：测试集准确率
        self.batch_losses = []
        self.batch_discard_rates = []
        
        # 新增：智能记录控制
        self.total_epochs = args.epochs
        self.record_frequency = self._calculate_record_frequency()
        self.last_recorded_step = {}
        
        # 新增：时间跟踪
        self.training_start_time = time.time()
        self.last_time_record = self.training_start_time
    
    def _calculate_record_frequency(self):
        """根据epoch数量智能计算记录频率"""
        if self.total_epochs <= 10:
            # 10个epoch以下：每个epoch记录
            return {'epoch': 1, 'batch': 1}
        elif self.total_epochs <= 50:
            # 50个epoch以下：每2个epoch记录一次，batch每5步记录
            return {'epoch': 2, 'batch': 5}
        elif self.total_epochs <= 100:
            # 100个epoch以下：每5个epoch记录一次，batch每10步记录
            return {'epoch': 5, 'batch': 10}
        else:
            # 100个epoch以上：每10个epoch记录一次，batch每20步记录
            return {'epoch': 10, 'batch': 20}
    
    def _should_record(self, metric_type, current_value):
        """判断是否应该记录当前指标"""
        if metric_type == 'epoch':
            return current_value % self.record_frequency['epoch'] == 0
        elif metric_type == 'batch':
            return current_value % self.record_frequency['batch'] == 0
        return True
    
    def train_epoch(self, train_loader, epoch, val_loader=None, test_loader=None):
        """训练一个epoch"""
        self.model.train()
        self.strategy.set_epoch(epoch)
        
        total_loss = 0
        total_batches = 0
        discarded_batches = 0
        
        # 添加有效训练步数计数器
        self.effective_training_steps = getattr(self, 'effective_training_steps', 0)
        
        for batch_idx, batch in enumerate(train_loader):
            # 将数据移到设备
            inputs = {
                'input_ids': batch['input_ids'].to(self.device),
                'attention_mask': batch['attention_mask'].to(self.device),
                'labels': batch['labels'].to(self.device),
                'pixel_values': batch['pixel_values'].to(self.device)
            }
            # 兼容可选图像辅助键
            for opt_key in ('image_grid_thw','image_sizes','image_input_size'):
                if opt_key in batch and batch[opt_key] is not None:
                    val = batch[opt_key]
                    inputs[opt_key] = val.to(self.device) if hasattr(val, 'to') else val
            image_path = batch.get('image_path') if isinstance(batch, dict) else None
            
            # 使用SlidingWindowMeanStrategy训练
            loss, repeat_count, is_discarded = self.strategy.train_batch(
                inputs,
                inputs['labels'],
                image_path=image_path
            )
            
            if is_discarded:
                discarded_batches += 1
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}: 丢弃（异常检测）")
                
                # 记录丢弃决策到TensorBoard（使用总步数）
                self.writer.add_scalar('TrainingStep/Strategy_Decision/策略决策行为', 0, self.cumulative_training_steps)
                
                # 只记录丢弃标记
                self.batch_discard_rates.append(1.0)
            else:
                total_loss += loss
                total_batches += 1
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}: 损失={loss:.4f}, 训练次数={repeat_count}")
                
                # 只记录有效训练的loss（使用有效步数）
                self.writer.add_scalar('TrainingStep/Train_Loss/训练过程监控', loss, self.effective_training_steps)
                # 额外记录：以总步数为横轴的loss
                self.writer.add_scalar('TrainingStep_Total/Train_Loss_总步数坐标', loss, self.cumulative_training_steps)
                
                # 新增：记录时间维度的loss
                current_time = time.time()
                elapsed_time = current_time - self.training_start_time
                self.writer.add_scalar('TimeBased/Train_Loss_时间坐标', loss, elapsed_time)
                
                # 记录策略决策行为
                self.writer.add_scalar('TrainingStep/Strategy_Decision/策略决策行为', 1, self.cumulative_training_steps)
                
                # 记录重复训练次数
                self.writer.add_scalar('TrainingStep/Repeat_Count/ABR策略决策行为', repeat_count, self.effective_training_steps)
                
                # 更新有效训练步数
                self.effective_training_steps += 1
                
                # 记录批次数组
                self.batch_losses.append(loss)
                self.batch_discard_rates.append(0.0)
            
            # 更新累积步数
            self.cumulative_training_steps += 1
            
            # 记录epoch信息
            self.writer.add_scalar('Progress/Epoch', epoch, self.cumulative_training_steps)
            
            # 记录epoch边界
            if batch_idx == 0:
                self.writer.add_scalar('Progress/Epoch_Start', epoch, self.cumulative_training_steps)
            
            # 限制训练批次数（用于快速测试）
            if self.args.max_train_batches > 0 and batch_idx >= self.args.max_train_batches - 1:
                break
        
        # 计算平均损失
        avg_loss = total_loss / max(1, total_batches)
        discard_rate = discarded_batches / max(1, len(train_loader))
        
        # 在验证集上评估
        val_loss = None
        val_accuracy = None
        if val_loader is not None:
            val_loss, val_accuracy = self.evaluate(val_loader)
            self.epoch_val_losses.append(val_loss)
            self.epoch_val_accuracies.append(val_accuracy)
            print(f"Epoch {epoch+1} 验证集损失: {val_loss:.4f}, 准确率: {val_accuracy:.4f}")
            
            # 记录验证准确率到TensorBoard
            self.writer.add_scalar('Epoch/Val_Accuracy/验证集性能', val_accuracy, epoch)
            self.writer.add_scalar('Epoch/Val_Loss/验证集损失', val_loss, epoch)
            
            # 新增：记录时间维度的验证loss
            current_time = time.time()
            elapsed_time = current_time - self.training_start_time
            self.writer.add_scalar('TimeBased/Val_Loss_时间坐标', val_loss, elapsed_time)
            self.writer.add_scalar('TimeBased/Val_Accuracy_时间坐标', val_accuracy, elapsed_time)
        
        # 在测试集上评估
        test_loss = None
        test_accuracy = None
        if test_loader is not None:
            test_loss, test_accuracy = self.evaluate(test_loader)
            self.epoch_test_losses.append(test_loss)
            self.epoch_test_accuracies.append(test_accuracy)
            print(f"Epoch {epoch+1} 测试集损失: {test_loss:.4f}, 准确率: {test_accuracy:.4f}")
            
            # 记录测试准确率到TensorBoard
            self.writer.add_scalar('Epoch/Test_Accuracy/最终主模型性能', test_accuracy, epoch)
            self.writer.add_scalar('Epoch/Test_Loss/判断模型过拟合风险', test_loss, epoch)
            
            # 新增：记录时间维度的测试loss
            current_time = time.time()
            elapsed_time = current_time - self.training_start_time
            self.writer.add_scalar('TimeBased/Test_Loss_时间坐标', test_loss, elapsed_time)
            self.writer.add_scalar('TimeBased/Test_Accuracy_时间坐标', test_accuracy, elapsed_time)
        
        # 记录epoch级别的指标到TensorBoard
        self.writer.add_scalar('Epoch/Train_Loss/训练损失监控', avg_loss, epoch)
        self.writer.add_scalar('Epoch/Discard_Rate/策略丢弃率', discard_rate, epoch)
        self.writer.add_scalar('Epoch/Cumulative_Steps/累计算力消耗对比', self.cumulative_training_steps, epoch)
        self.writer.add_scalar('Progress/Epoch_Complete', 1.0, epoch)
        self.writer.add_scalar('Progress/Epoch_Boundary', epoch + 1, epoch)
        
        # 记录进度百分比
        progress = (epoch + 1) / self.total_epochs
        self.writer.add_scalar('Progress/Epoch_Progress', progress, epoch)
        
        # 记录相对进度（0-1之间）
        relative_progress = epoch / max(1, self.total_epochs - 1)
        self.writer.add_scalar('Progress/Relative_Progress', relative_progress, epoch)
        
        # 记录epoch级别的指标
        self.epoch_losses.append(avg_loss)
        self.epoch_discard_rates.append(discard_rate)
        
        # 打印统计信息
        self.strategy.print_statistics()
        
        return avg_loss, discard_rate, val_loss, val_accuracy, test_loss, test_accuracy
    
    def evaluate(self, eval_loader):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        total_batches = 0
        total_correct = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in eval_loader:
                inputs = {
                    'input_ids': batch['input_ids'].to(self.device),
                    'attention_mask': batch['attention_mask'].to(self.device),
                    'labels': batch['labels'].to(self.device),
                    'pixel_values': batch['pixel_values'].to(self.device)
                }
                # 兼容可选图像辅助键
                for opt_key in ('image_grid_thw','image_sizes','image_input_size'):
                    if opt_key in batch and batch[opt_key] is not None:
                        val = batch[opt_key]
                        inputs[opt_key] = val.to(self.device) if hasattr(val, 'to') else val
                
                outputs = self.model(**inputs)
                loss = outputs.logits
                
                # 计算损失
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
                loss_per_token = loss_fct(loss.view(-1, loss.size(-1)), inputs['labels'].view(-1))
                total_loss += loss_per_token.sum().item()
                total_batches += 1
                
                # 计算准确率
                predictions = loss.argmax(dim=-1)
                labels = inputs['labels']
                
                # 只计算非忽略的token
                non_ignore_mask = labels != -100
                correct_predictions = (predictions == labels) & non_ignore_mask
                total_correct += correct_predictions.sum().item()
                total_tokens += non_ignore_mask.sum().item()
        
        avg_loss = total_loss / max(1, total_tokens)
        accuracy = total_correct / max(1, total_tokens)
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch, loss):
        """保存检查点"""
        checkpoint_path = os.path.join(self.args.output_dir, f'checkpoint_epoch_{epoch+1}.pt')
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'strategy_state': {
                'loss_window': list(self.strategy.loss_window),
                'batch_history': self.strategy.batch_history,
                'outlier_log': self.strategy.outlier_log
            }
        }, checkpoint_path)
        
        print(f"检查点已保存到: {checkpoint_path}")
    
    def close(self):
        """关闭资源"""
        self.writer.close()
        self.strategy.save_final_logs()
        
        # 保存NumPy结果
        self.save_numpy_results()

    def save_numpy_results(self):
        """保存NumPy结果用于对比分析"""
        results_dir = os.path.join(self.args.output_dir, 'sliding_window_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # 保存epoch级别的指标
        np.save(os.path.join(results_dir, 'epoch_losses.npy'), np.array(self.epoch_losses))
        np.save(os.path.join(results_dir, 'epoch_discard_rates.npy'), np.array(self.epoch_discard_rates))
        np.save(os.path.join(results_dir, 'epoch_val_losses.npy'), np.array(self.epoch_val_losses))
        np.save(os.path.join(results_dir, 'epoch_val_accuracies.npy'), np.array(self.epoch_val_accuracies))
        np.save(os.path.join(results_dir, 'epoch_test_losses.npy'), np.array(self.epoch_test_losses))
        np.save(os.path.join(results_dir, 'epoch_test_accuracies.npy'), np.array(self.epoch_test_accuracies))
        
        # 保存batch级别的指标
        np.save(os.path.join(results_dir, 'batch_losses.npy'), np.array(self.batch_losses))
        np.save(os.path.join(results_dir, 'batch_discard_rates.npy'), np.array(self.batch_discard_rates))
        
        # 保存策略的NumPy数据
        if hasattr(self.strategy, 'save_numpy_data'):
            self.strategy.save_numpy_data(self.args.output_dir)
        
        print(f"SlidingWindowMeanStrategy NumPy结果已保存到: {results_dir}")

class SlidingWindowProportionalStrategyTrainer:
    """使用SlidingWindowProportionalStrategy的训练器"""
    
    def __init__(self, model, processor, args):
        self.model = model
        self.processor = processor
        self.args = args
        self.device = next(model.parameters()).device
        
        # 创建损失函数和优化器
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # 延迟导入SlidingWindowProportionalStrategy
        try:
            from mystrategy import SlidingWindowProportionalStrategy
            self.strategy = SlidingWindowProportionalStrategy(
                model=model,
                criterion=self.criterion,
                optimizer=self.optimizer,
                window_size=args.window_size,
                proportion_factor=args.proportion_factor,
                save_outlier_log=True,
                outlier_log_path=os.path.join(args.output_dir, 'sliding_window_proportional_outlier_log.json')
            )
        except ImportError as e:
            print(f"导入SlidingWindowProportionalStrategy失败: {e}")
            print("请检查mystrategy.py文件中的类定义")
            raise
        
        # 创建TensorBoard writer - 使用更专业的目录结构
        tensorboard_dir = os.path.join(args.output_dir, 'tensorboard')
        os.makedirs(tensorboard_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=tensorboard_dir)
        
        # 添加累积步数计数器
        self.cumulative_training_steps = 0
        
        # 添加数组存储
        self.epoch_losses = []
        self.epoch_discard_rates = []
        self.epoch_val_losses = []  # 新增：验证集指标列表，避免在train_epoch追加时报AttributeError
        self.epoch_val_accuracies = []
        self.epoch_test_losses = []
        self.epoch_test_accuracies = []
        self.batch_losses = []
        self.batch_discard_rates = []
        
        # 智能记录控制
        self.total_epochs = args.epochs
        self.record_frequency = self._calculate_record_frequency()
        self.last_recorded_step = {}
        
        # 新增：时间跟踪
        self.training_start_time = time.time()
        self.last_time_record = self.training_start_time
    
    def _calculate_record_frequency(self):
        """根据epoch数量智能计算记录频率"""
        if self.total_epochs <= 10:
            return {'epoch': 1, 'batch': 1}
        elif self.total_epochs <= 50:
            return {'epoch': 2, 'batch': 5}
        elif self.total_epochs <= 100:
            return {'epoch': 5, 'batch': 10}
        else:
            return {'epoch': 10, 'batch': 20}
    
    def _should_record(self, metric_type, current_value):
        """判断是否应该记录当前指标"""
        if metric_type == 'epoch':
            return current_value % self.record_frequency['epoch'] == 0
        elif metric_type == 'batch':
            return current_value % self.record_frequency['batch'] == 0
        return True
    
    def train_epoch(self, train_loader, epoch, val_loader=None, test_loader=None):
        """训练一个epoch"""
        self.model.train()
        self.strategy.set_epoch(epoch)
        
        total_loss = 0
        total_batches = 0
        discarded_batches = 0
        
        # 添加有效训练步数计数器
        self.effective_training_steps = getattr(self, 'effective_training_steps', 0)
        
        for batch_idx, batch in enumerate(train_loader):
            # 将数据移到设备
            inputs = {
                'input_ids': batch['input_ids'].to(self.device),
                'attention_mask': batch['attention_mask'].to(self.device),
                'labels': batch['labels'].to(self.device),
                'pixel_values': batch['pixel_values'].to(self.device)
            }
            # 兼容可选图像辅助键
            for opt_key in ('image_grid_thw','image_sizes','image_input_size'):
                if opt_key in batch and batch[opt_key] is not None:
                    val = batch[opt_key]
                    inputs[opt_key] = val.to(self.device) if hasattr(val, 'to') else val
            image_path = batch.get('image_path') if isinstance(batch, dict) else None
            
            # 使用SlidingWindowProportionalStrategy训练
            loss, repeat_count, is_discarded = self.strategy.train_batch(
                inputs,
                inputs['labels'],
                image_path=image_path
            )
            
            if is_discarded:
                discarded_batches += 1
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}: 丢弃（比例阈值异常检测）")
                
                # 记录丢弃决策到TensorBoard（使用总步数）
                self.writer.add_scalar('TrainingStep/Strategy_Decision/策略决策行为', 0, self.cumulative_training_steps)
                
                # 只记录丢弃标记
                self.batch_discard_rates.append(1.0)
            else:
                total_loss += loss
                total_batches += 1
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}: 损失={loss:.4f}, 训练次数={repeat_count}")
                
                # 只记录有效训练的loss（使用有效步数）
                self.writer.add_scalar('TrainingStep/Train_Loss/训练过程监控', loss, self.effective_training_steps)
                # 额外记录：以总步数为横轴的loss
                self.writer.add_scalar('TrainingStep_Total/Train_Loss_总步数坐标', loss, self.cumulative_training_steps)
                
                # 新增：记录时间维度的loss
                current_time = time.time()
                elapsed_time = current_time - self.training_start_time
                self.writer.add_scalar('TimeBased/Train_Loss_时间坐标', loss, elapsed_time)
                
                # 记录策略决策行为
                self.writer.add_scalar('TrainingStep/Strategy_Decision/策略决策行为', 1, self.cumulative_training_steps)
                
                # 记录重复训练次数
                self.writer.add_scalar('TrainingStep/Repeat_Count/ABR策略决策行为', repeat_count, self.effective_training_steps)
                
                # 更新有效训练步数
                self.effective_training_steps += 1
                
                # 记录批次数组
                self.batch_losses.append(loss)
                self.batch_discard_rates.append(0.0)
            
            # 更新累积步数
            self.cumulative_training_steps += 1
            
            # 记录epoch信息
            self.writer.add_scalar('Progress/Epoch', epoch, self.cumulative_training_steps)
            
            # 记录epoch边界
            if batch_idx == 0:
                self.writer.add_scalar('Progress/Epoch_Start', epoch, self.cumulative_training_steps)
            
            # 限制训练批次数（用于快速测试）
            if self.args.max_train_batches > 0 and batch_idx >= self.args.max_train_batches - 1:
                break
        
        # 计算平均损失
        avg_loss = total_loss / max(1, total_batches)
        discard_rate = discarded_batches / max(1, len(train_loader))
        
        # 在验证集上评估
        val_loss = None
        val_accuracy = None
        if val_loader is not None:
            val_loss, val_accuracy = self.evaluate(val_loader)
            self.epoch_val_losses.append(val_loss)
            self.epoch_val_accuracies.append(val_accuracy)
            print(f"Epoch {epoch+1} 验证集损失: {val_loss:.4f}, 准确率: {val_accuracy:.4f}")
            
            # 记录验证准确率到TensorBoard
            self.writer.add_scalar('Epoch/Val_Accuracy/验证集性能', val_accuracy, epoch)
            self.writer.add_scalar('Epoch/Val_Loss/验证集损失', val_loss, epoch)
            
            # 新增：记录时间维度的验证loss
            current_time = time.time()
            elapsed_time = current_time - self.training_start_time
            self.writer.add_scalar('TimeBased/Val_Loss_时间坐标', val_loss, elapsed_time)
            self.writer.add_scalar('TimeBased/Val_Accuracy_时间坐标', val_accuracy, elapsed_time)
        
        # 在测试集上评估
        test_loss = None
        test_accuracy = None
        if test_loader is not None:
            test_loss, test_accuracy = self.evaluate(test_loader)
            self.epoch_test_losses.append(test_loss)
            self.epoch_test_accuracies.append(test_accuracy)
            print(f"Epoch {epoch+1} 测试集损失: {test_loss:.4f}, 准确率: {test_accuracy:.4f}")
            
            # 记录测试准确率到TensorBoard
            self.writer.add_scalar('Epoch/Test_Accuracy/最终主模型性能', test_accuracy, epoch)
            self.writer.add_scalar('Epoch/Test_Loss/判断模型过拟合风险', test_loss, epoch)
            
            # 新增：记录时间维度的测试loss
            current_time = time.time()
            elapsed_time = current_time - self.training_start_time
            self.writer.add_scalar('TimeBased/Test_Loss_时间坐标', test_loss, elapsed_time)
            self.writer.add_scalar('TimeBased/Test_Accuracy_时间坐标', test_accuracy, elapsed_time)
        
        # 记录epoch级别的指标到TensorBoard
        self.writer.add_scalar('Epoch/Train_Loss/训练损失监控', avg_loss, epoch)
        self.writer.add_scalar('Epoch/Discard_Rate/策略丢弃率', discard_rate, epoch)
        self.writer.add_scalar('Epoch/Cumulative_Steps/累计算力消耗对比', self.cumulative_training_steps, epoch)
        self.writer.add_scalar('Progress/Epoch_Complete', 1.0, epoch)
        self.writer.add_scalar('Progress/Epoch_Boundary', epoch + 1, epoch)
        
        # 记录进度百分比
        progress = (epoch + 1) / self.total_epochs
        self.writer.add_scalar('Progress/Epoch_Progress', progress, epoch)
        
        # 记录相对进度（0-1之间）
        relative_progress = epoch / max(1, self.total_epochs - 1)
        self.writer.add_scalar('Progress/Relative_Progress', relative_progress, epoch)
        
        # 记录epoch级别的指标
        self.epoch_losses.append(avg_loss)
        self.epoch_discard_rates.append(discard_rate)
        
        # 打印统计信息
        self.strategy.print_statistics()
        
        return avg_loss, discard_rate, val_loss, val_accuracy, test_loss, test_accuracy
    
    def evaluate(self, eval_loader):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        total_batches = 0
        total_correct = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in eval_loader:
                inputs = {
                    'input_ids': batch['input_ids'].to(self.device),
                    'attention_mask': batch['attention_mask'].to(self.device),
                    'labels': batch['labels'].to(self.device),
                    'pixel_values': batch['pixel_values'].to(self.device)
                }
                # 兼容可选图像辅助键
                for opt_key in ('image_grid_thw','image_sizes','image_input_size'):
                    if opt_key in batch and batch[opt_key] is not None:
                        val = batch[opt_key]
                        inputs[opt_key] = val.to(self.device) if hasattr(val, 'to') else val
                
                outputs = self.model(**inputs)
                loss = outputs.logits
                
                # 计算损失
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
                loss_per_token = loss_fct(loss.view(-1, loss.size(-1)), inputs['labels'].view(-1))
                total_loss += loss_per_token.sum().item()
                total_batches += 1
                
                # 计算准确率
                predictions = loss.argmax(dim=-1)
                labels = inputs['labels']
                
                # 只计算非忽略的token
                non_ignore_mask = labels != -100
                correct_predictions = (predictions == labels) & non_ignore_mask
                total_correct += correct_predictions.sum().item()
                total_tokens += non_ignore_mask.sum().item()
        
        avg_loss = total_loss / max(1, total_tokens)
        accuracy = total_correct / max(1, total_tokens)
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch, loss):
        """保存检查点"""
        checkpoint_path = os.path.join(self.args.output_dir, f'checkpoint_epoch_{epoch+1}.pt')
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'strategy_state': {
                'loss_window': list(self.strategy.loss_window),
                'batch_history': self.strategy.batch_history,
                'outlier_log': self.strategy.outlier_log
            }
        }, checkpoint_path)
        
        print(f"检查点已保存到: {checkpoint_path}")
    
    def close(self):
        """关闭资源"""
        self.writer.close()
        self.strategy.save_final_logs()
        
        # 保存NumPy结果
        self.save_numpy_results()

    def save_numpy_results(self):
        """保存NumPy结果用于对比分析"""
        results_dir = os.path.join(self.args.output_dir, 'sliding_window_proportional_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # 保存epoch级别的指标
        np.save(os.path.join(results_dir, 'epoch_losses.npy'), np.array(self.epoch_losses))
        np.save(os.path.join(results_dir, 'epoch_discard_rates.npy'), np.array(self.epoch_discard_rates))
        np.save(os.path.join(results_dir, 'epoch_val_losses.npy'), np.array(self.epoch_val_losses))
        np.save(os.path.join(results_dir, 'epoch_val_accuracies.npy'), np.array(self.epoch_val_accuracies))
        np.save(os.path.join(results_dir, 'epoch_test_losses.npy'), np.array(self.epoch_test_losses))
        np.save(os.path.join(results_dir, 'epoch_test_accuracies.npy'), np.array(self.epoch_test_accuracies))
        
        # 保存batch级别的指标
        np.save(os.path.join(results_dir, 'batch_losses.npy'), np.array(self.batch_losses))
        np.save(os.path.join(results_dir, 'batch_discard_rates.npy'), np.array(self.batch_discard_rates))
        
        # 保存策略的NumPy数据
        if hasattr(self.strategy, 'save_numpy_data'):
            self.strategy.save_numpy_data(self.args.output_dir)
        
        print(f"SlidingWindowProportionalStrategy NumPy结果已保存到: {results_dir}")

def main():
    parser = argparse.ArgumentParser(description='使用MyStrategy微调Qwen2.5-VL-7B')
    
    # 数据参数
    parser.add_argument('--data_path', type=str, required=True, help='数据集路径（目录或JSON文件）')
    parser.add_argument('--output_dir', type=str, default='./output', help='输出目录')
    parser.add_argument('--train_size', type=float, default=0.7, help='训练集比例（0.0-1.0）')
    parser.add_argument('--val_size', type=float, default=0.15, help='验证集比例（0.0-1.0）')
    parser.add_argument('--test_size', type=float, default=0.15, help='测试集比例（0.0-1.0）')
    parser.add_argument('--random_state', type=int, default=42, help='随机种子，确保可重复性')
    parser.add_argument('--use_validation', action='store_true', help='是否使用验证集（三路分割）')
    parser.add_argument('--val_from_test', action='store_true', help='从测试集中分割验证集（保持训练集不变）')
    parser.add_argument('--val_ratio', type=float, default=0.5, help='从测试集中分割验证集的比例（0.0-1.0）')
    
    # 新增：数据格式转换参数
    parser.add_argument('--convert_format', action='store_true', 
                       help='是否转换JSON格式（如果数据格式不匹配）')
    parser.add_argument('--converted_output', type=str, default='./converted_data.json',
                       help='转换后的数据输出路径')
    
    # 模型参数
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-VL-7B-Instruct', help='预训练模型名称')
    parser.add_argument('--max_length', type=int, default=512, help='最大序列长度')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=2, help='批次大小')
    parser.add_argument('--epochs', type=int, default=1, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=5e-6, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--max_train_batches', type=int, default=-1, help='最大训练批次数（-1表示不限制）')
    
    # MyStrategy参数
    parser.add_argument('--window_size', type=int, default=5, help='滑动窗口大小')
    parser.add_argument('--loss_threshold', type=float, default=0.3, help='损失阈值')
    parser.add_argument('--trend_threshold', type=float, default=0.01, help='趋势阈值')
    parser.add_argument('--vol_threshold', type=float, default=0.1, help='波动阈值')
    parser.add_argument('--outlier_threshold', type=float, default=2.0, help='异常检测阈值')
    
    # 新增：滑动窗口均值策略参数
    parser.add_argument('--threshold', type=float, default=1.0, help='滑动窗口均值策略的固定阈值')
    
    # LoRA参数
    parser.add_argument('--lora_rank', type=int, default=16, help='LoRA秩')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout')
    
    # 添加策略选择参数
    parser.add_argument('--strategy', type=str, default='mystrategy', 
                       choices=['mystrategy', 'baseline', 'sliding_window_mean', 'sliding_window_proportional'], 
                       help='训练策略选择: mystrategy, baseline, sliding_window_mean, 或 sliding_window_proportional')
    
    # 新增：策略比较参数
    parser.add_argument('--compare_strategies', action='store_true', 
                       help='是否比较两个策略（需要重新训练）')
    parser.add_argument('--comparison_output_dir', type=str, default='./comparison_results',
                       help='策略比较结果输出目录')
    
    # 新增：滑动窗口比例策略参数
    parser.add_argument('--proportion_factor', type=float, default=1.2, 
                       help='滑动窗口比例策略的比例系数（如1.2表示阈值=平均值*1.2）')
    
    args = parser.parse_args()
    
    # 如果需要转换数据格式
    if args.convert_format:
        print("=== 数据格式转换模式 ===")
        print(f"输入文件: {args.data_path}")
        print(f"输出文件: {args.converted_output}")
        
        # 使用ImageCaptionDataset的静态方法转换格式
        converted_data = ImageCaptionDataset.convert_json_format(args.data_path, args.converted_output)
        print(f"转换完成！共转换 {len(converted_data)} 个样本")
        print(f"转换后的数据已保存到: {args.converted_output}")
        print("请使用转换后的数据文件重新运行训练命令")
        return
    
    # 创建输出目录 - 使用专业结构
    timestamp = int(time.time())
    base_output_dir = f'results/strategy_training/tensorboard/california_{timestamp}'
    
    if args.strategy == 'baseline':
        strategy_output_dir = os.path.join(base_output_dir, 'Baseline')
    elif args.strategy == 'sliding_window_mean':
        strategy_output_dir = os.path.join(base_output_dir, f'SlidingWindow_Thres{args.threshold}')
    elif args.strategy == 'sliding_window_proportional':
        strategy_output_dir = os.path.join(base_output_dir, f'SlidingWindowProportional_Factor{args.proportion_factor}')
    elif args.strategy == 'mystrategy':
        strategy_output_dir = os.path.join(base_output_dir, f'MyStrategy_Thres{args.outlier_threshold}')
    else:
        strategy_output_dir = os.path.join(base_output_dir, args.strategy)
    
    os.makedirs(strategy_output_dir, exist_ok=True)
    args.output_dir = strategy_output_dir  # 更新args中的output_dir
    
    if args.compare_strategies:
        # 策略比较模式
        print("=== 策略比较模式 ===")
        os.makedirs(args.comparison_output_dir, exist_ok=True)
        
        # 创建模型和处理器
        print("创建模型和处理器...")
        model, processor = create_model_and_processor(
            model_name=args.model_name,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            device_map="auto",
            torch_dtype=torch.float32  # 改为float32
        )
        
        print("可训练参数:")
        model.print_trainable_parameters()
        
        # 创建数据集
        print("创建数据集...")
        if args.val_from_test:
            # 从测试集中分割验证集（保持训练集不变）
            train_dataset, val_dataset, test_dataset = create_datasets_with_val_from_test(
                args.data_path, processor, args.max_length, 
                test_size=args.test_size, val_ratio=args.val_ratio, 
                random_state=args.random_state
            )
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        elif args.use_validation:
            # 三路分割（训练集、验证集、测试集）
            train_dataset, val_dataset, test_dataset = create_datasets(
                args.data_path, processor, args.max_length, 
                train_size=args.train_size, val_size=args.val_size, test_size=args.test_size, 
                random_state=args.random_state
            )
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        else:
            # 原有的两路分割（训练集、测试集）
            train_dataset, test_dataset = create_train_test_datasets(
                args.data_path, processor, args.max_length, 
                test_size=args.test_size, random_state=args.random_state
            )
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            val_loader = None
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        # 训练MyStrategy
        print("\n=== 训练MyStrategy ===")
        mystrategy_output_dir = os.path.join(args.comparison_output_dir, 'mystrategy')
        os.makedirs(mystrategy_output_dir, exist_ok=True)
        
        mystrategy_args = type('Args', (), {
            'output_dir': mystrategy_output_dir,
            'window_size': args.window_size,
            'loss_threshold': args.loss_threshold,
            'trend_threshold': args.trend_threshold,
            'vol_threshold': args.vol_threshold,
            'outlier_threshold': args.outlier_threshold,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'epochs': args.epochs,
            'max_train_batches': args.max_train_batches
        })()
        
        mystrategy_trainer = MyStrategyTrainer(model, processor, mystrategy_args)
        
        mystrategy_results = []
        for epoch in range(args.epochs):
            print(f"\n--- MyStrategy Epoch {epoch+1}/{args.epochs} ---")
            if args.val_from_test or args.use_validation:
                train_loss, discard_rate, val_loss, val_accuracy, test_loss, test_accuracy = mystrategy_trainer.train_epoch(train_loader, epoch, val_loader, test_loader)
            else:
                train_loss, discard_rate, val_loss, val_accuracy, test_loss, test_accuracy = mystrategy_trainer.train_epoch(train_loader, epoch, None, test_loader)
            mystrategy_results.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'test_loss': test_loss,
                'test_accuracy': test_accuracy,
                'discard_rate': discard_rate
            })
            
            if (epoch + 1) % 2 == 0:
                mystrategy_trainer.save_checkpoint(epoch, train_loss)
        
        mystrategy_trainer.close()
        
        # 重新创建模型和处理器（避免权重污染）
        print("\n重新创建模型和处理器...")
        del model, processor, mystrategy_trainer
        torch.cuda.empty_cache()
        
        model, processor = create_model_and_processor(
            model_name=args.model_name,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            device_map="auto",
            torch_dtype=torch.float32  # 改为float32
        )
        
        # 训练BaselineStrategy
        print("\n=== 训练BaselineStrategy ===")
        baseline_output_dir = os.path.join(args.comparison_output_dir, 'baseline')
        os.makedirs(baseline_output_dir, exist_ok=True)
        
        baseline_args = type('Args', (), {
            'output_dir': baseline_output_dir,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'epochs': args.epochs,
            'max_train_batches': args.max_train_batches
        })()
        
        baseline_trainer = BaselineStrategyTrainer(model, processor, baseline_args)
        
        baseline_results = []
        for epoch in range(args.epochs):
            print(f"\n--- Baseline Epoch {epoch+1}/{args.epochs} ---")
            if args.val_from_test or args.use_validation:
                train_loss, discard_rate, val_loss, val_accuracy, test_loss, test_accuracy = baseline_trainer.train_epoch(train_loader, epoch, val_loader, test_loader)
            else:
                train_loss, discard_rate, val_loss, val_accuracy, test_loss, test_accuracy = baseline_trainer.train_epoch(train_loader, epoch, None, test_loader)
            baseline_results.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'test_loss': test_loss,
                'test_accuracy': test_accuracy,
                'discard_rate': discard_rate
            })
            
            if (epoch + 1) % 2 == 0:
                baseline_trainer.save_checkpoint(epoch, train_loss)
        
        baseline_trainer.close()
        
        # 训练SlidingWindowMeanStrategy
        print("\n=== 训练SlidingWindowMeanStrategy ===")
        sliding_window_output_dir = os.path.join(args.comparison_output_dir, 'sliding_window_mean')
        os.makedirs(sliding_window_output_dir, exist_ok=True)
        
        sliding_window_args = type('Args', (), {
            'output_dir': sliding_window_output_dir,
            'window_size': args.window_size,
            'threshold': args.threshold,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'epochs': args.epochs,
            'max_train_batches': args.max_train_batches
        })()
        
        sliding_window_trainer = SlidingWindowMeanStrategyTrainer(model, processor, sliding_window_args)
        
        sliding_window_results = []
        for epoch in range(args.epochs):
            print(f"\n--- SlidingWindowMeanStrategy Epoch {epoch+1}/{args.epochs} ---")
            if args.val_from_test or args.use_validation:
                train_loss, discard_rate, val_loss, val_accuracy, test_loss, test_accuracy = sliding_window_trainer.train_epoch(train_loader, epoch, val_loader, test_loader)
            else:
                train_loss, discard_rate, val_loss, val_accuracy, test_loss, test_accuracy = sliding_window_trainer.train_epoch(train_loader, epoch, None, test_loader)
            sliding_window_results.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'test_loss': test_loss,
                'test_accuracy': test_accuracy,
                'discard_rate': discard_rate
            })
            
            if (epoch + 1) % 2 == 0:
                sliding_window_trainer.save_checkpoint(epoch, train_loss)
        
        sliding_window_trainer.close()
        
        # 训练SlidingWindowProportionalStrategy
        print("\n=== 训练SlidingWindowProportionalStrategy ===")
        sliding_proportional_output_dir = os.path.join(args.comparison_output_dir, 'sliding_window_proportional')
        os.makedirs(sliding_proportional_output_dir, exist_ok=True)
        
        sliding_proportional_args = type('Args', (), {
            'output_dir': sliding_proportional_output_dir,
            'window_size': args.window_size,
            'proportion_factor': args.proportion_factor,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'epochs': args.epochs,
            'max_train_batches': args.max_train_batches
        })()
        
        sliding_proportional_trainer = SlidingWindowProportionalStrategyTrainer(model, processor, sliding_proportional_args)
        
        sliding_proportional_results = []
        for epoch in range(args.epochs):
            print(f"\n--- SlidingWindowProportionalStrategy Epoch {epoch+1}/{args.epochs} ---")
            if args.val_from_test or args.use_validation:
                train_loss, discard_rate, val_loss, val_accuracy, test_loss, test_accuracy = sliding_proportional_trainer.train_epoch(train_loader, epoch, val_loader, test_loader)
            else:
                train_loss, discard_rate, val_loss, val_accuracy, test_loss, test_accuracy = sliding_proportional_trainer.train_epoch(train_loader, epoch, None, test_loader)
            sliding_proportional_results.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'test_loss': test_loss,
                'test_accuracy': test_accuracy,
                'discard_rate': discard_rate
            })
            
            if (epoch + 1) % 2 == 0:
                sliding_proportional_trainer.save_checkpoint(epoch, train_loss)
        
        sliding_proportional_trainer.close()
        
        # 生成比较报告
        print("\n=== 生成策略比较报告 ===")
        # 生成两两比较报告
        generate_comparison_report(mystrategy_results, baseline_results, args.comparison_output_dir, "MyStrategy", "Baseline")
        generate_comparison_report(mystrategy_results, sliding_window_results, args.comparison_output_dir, "MyStrategy", "SlidingWindowMeanStrategy")
        generate_comparison_report(baseline_results, sliding_window_results, args.comparison_output_dir, "Baseline", "SlidingWindowMeanStrategy")
        generate_comparison_report(mystrategy_results, sliding_proportional_results, args.comparison_output_dir, "MyStrategy", "SlidingWindowProportionalStrategy")
        generate_comparison_report(baseline_results, sliding_proportional_results, args.comparison_output_dir, "Baseline", "SlidingWindowProportionalStrategy")
        generate_comparison_report(sliding_window_results, sliding_proportional_results, args.comparison_output_dir, "SlidingWindowMeanStrategy", "SlidingWindowProportionalStrategy")
        
        # 生成三策略综合比较报告
        generate_three_way_comparison_report(mystrategy_results, baseline_results, sliding_window_results, args.comparison_output_dir)
        generate_three_way_comparison_report(mystrategy_results, baseline_results, sliding_proportional_results, args.comparison_output_dir)
        generate_three_way_comparison_report(baseline_results, sliding_window_results, sliding_proportional_results, args.comparison_output_dir)
        
        print(f"\n策略比较完成！结果保存在: {args.comparison_output_dir}")
        
    else:
        # 单策略训练模式
        print(f"=== 单策略训练模式: {args.strategy} ===")
        
        # 创建模型和处理器
        print("创建模型和处理器...")
        model, processor = create_model_and_processor(
            model_name=args.model_name,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            device_map="auto",
            torch_dtype=torch.float16,  # 使用float16减少显存
            load_in_8bit=True  # 启用8bit量化
        )
        
        print("可训练参数:")
        model.print_trainable_parameters()
        
        # 创建数据集
        print("创建数据集...")
        if args.val_from_test:
            # 从测试集中分割验证集（保持训练集不变）
            train_dataset, val_dataset, test_dataset = create_datasets_with_val_from_test(
                args.data_path, processor, args.max_length, 
                test_size=args.test_size, val_ratio=args.val_ratio, 
                random_state=args.random_state
            )
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        elif args.use_validation:
            # 三路分割（训练集、验证集、测试集）
            train_dataset, val_dataset, test_dataset = create_datasets(
                args.data_path, processor, args.max_length, 
                train_size=args.train_size, val_size=args.val_size, test_size=args.test_size, 
                random_state=args.random_state
            )
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        else:
            # 原有的两路分割（训练集、测试集）
            train_dataset, test_dataset = create_train_test_datasets(
                args.data_path, processor, args.max_length, 
                test_size=args.test_size, random_state=args.random_state
            )
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            val_loader = None
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        # 创建训练器（根据策略选择）
        if args.strategy == 'baseline':
            trainer = BaselineStrategyTrainer(model, processor, args)
        elif args.strategy == 'sliding_window_mean':
            # 创建SlidingWindowMeanStrategyTrainer
            trainer = SlidingWindowMeanStrategyTrainer(model, processor, args)
        elif args.strategy == 'sliding_window_proportional':
            # 创建SlidingWindowProportionalStrategyTrainer
            trainer = SlidingWindowProportionalStrategyTrainer(model, processor, args)
        else: # mystrategy
            trainer = MyStrategyTrainer(model, processor, args)
        
        # 开始训练
        print("开始训练...")
        start_time = time.time()
        
        for epoch in range(args.epochs):
            print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
            
            # 训练
            if args.val_from_test or args.use_validation:
                train_loss, discard_rate, val_loss, val_accuracy, test_loss, test_accuracy = trainer.train_epoch(train_loader, epoch, val_loader, test_loader)
            else:
                train_loss, discard_rate, val_loss, val_accuracy, test_loss, test_accuracy = trainer.train_epoch(train_loader, epoch, None, test_loader)
            
            # 移除复杂的记录，只保留简单的打印
            print(f"Epoch {epoch+1} 完成:")
            print(f"  训练损失: {train_loss:.4f}")
            print(f"  丢弃率: {discard_rate:.2%}")
            if val_loss is not None:
                print(f"  验证集损失: {val_loss:.4f}, 准确率: {val_accuracy:.4f}")
            if test_loss is not None:
                print(f"  测试集损失: {test_loss:.4f}, 准确率: {test_accuracy:.4f}")
            
            # 保存检查点
            if (epoch + 1) % 2 == 0:  # 每2个epoch保存一次
                trainer.save_checkpoint(epoch, train_loss)
        
        total_time = time.time() - start_time
        print(f"\n训练完成! 总时间: {total_time:.2f}秒")
        
        # 保存最终模型
        final_model_path = os.path.join(args.output_dir, 'final_model')
        model.save_pretrained(final_model_path)
        try:
            # 尝试保存处理器
            if hasattr(processor, 'save_pretrained'):
                processor.save_pretrained(final_model_path)
            else:
                # 对于不支持 save_pretrained 的处理器，保存配置
                processor_config = {
                    'processor_type': type(processor).__name__,
                    'config': getattr(processor, 'config', {}),
                    'tokenizer_config': getattr(processor, 'tokenizer_config', {})
                }
                
                config_path = os.path.join(final_model_path, 'processor_config.json')
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(processor_config, f, indent=2, ensure_ascii=False)
                
                print(f"处理器配置已保存到: {config_path}")
                
        except Exception as e:
            print(f"保存处理器时出错: {e}")
            print("继续保存其他组件...")
        print(f"最终模型已保存到: {final_model_path}")
        
        # 关闭资源
        trainer.close()

def generate_comparison_report(strategy1_results, strategy2_results, output_dir, strategy1_name="Strategy1", strategy2_name="Strategy2"):
    """生成策略比较报告"""
    # 生成文件名，避免冲突
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(output_dir, f'strategy_comparison_{strategy1_name}_vs_{strategy2_name}_{timestamp}.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"=== {strategy1_name} vs {strategy2_name} 策略比较报告 ===\n\n")
        f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"训练轮数: {len(strategy1_results)}\n\n")
        
        # 训练集损失比较
        f.write(f"=== 训练集损失比较 ===\n")
        f.write(f"Epoch\t{strategy1_name}\t{strategy2_name}\t差异\t优势策略\n")
        f.write("-" * 60 + "\n")
        
        for i in range(len(strategy1_results)):
            s1_train = strategy1_results[i]['train_loss']
            s2_train = strategy2_results[i]['train_loss']
            diff = s1_train - s2_train
            advantage = strategy1_name if diff < 0 else strategy2_name
            f.write(f"{i+1}\t{s1_train:.4f}\t{s2_train:.4f}\t{diff:+.4f}\t{advantage}\n")
        
        # 测试集损失比较
        f.write(f"\n=== 测试集损失比较 ===\n")
        f.write(f"Epoch\t{strategy1_name}\t{strategy2_name}\t差异\t优势策略\n")
        f.write("-" * 60 + "\n")
        
        for i in range(len(strategy1_results)):
            s1_test = strategy1_results[i]['test_loss']
            s2_test = strategy2_results[i]['test_loss']
            diff = s1_test - s2_test
            advantage = strategy1_name if diff < 0 else strategy2_name
            f.write(f"{i+1}\t{s1_test:.4f}\t{s2_test:.4f}\t{diff:+.4f}\t{advantage}\n")
        
        # 过拟合分析
        f.write(f"\n=== 过拟合分析 ===\n")
        f.write(f"Epoch\t{strategy1_name}过拟合\t{strategy2_name}过拟合\t{strategy1_name}优势\n")
        f.write("-" * 60 + "\n")
        
        for i in range(len(strategy1_results)):
            s1_overfit = strategy1_results[i]['train_loss'] - strategy1_results[i]['test_loss']
            s2_overfit = strategy2_results[i]['train_loss'] - strategy2_results[i]['test_loss']
            s1_advantage = "是" if s1_overfit < s2_overfit else "否"
            f.write(f"{i+1}\t{s1_overfit:.4f}\t{s2_overfit:.4f}\t{s1_advantage}\n")
        
        # 丢弃率分析（如果策略支持）
        if 'discard_rate' in strategy1_results[0] and 'discard_rate' in strategy2_results[0]:
            f.write(f"\n=== 丢弃率分析 ===\n")
            f.write(f"Epoch\t{strategy1_name}丢弃率\t{strategy2_name}丢弃率\t说明\n")
            f.write("-" * 60 + "\n")
            
            for i in range(len(strategy1_results)):
                s1_discard = strategy1_results[i]['discard_rate']
                s2_discard = strategy2_results[i]['discard_rate']
                explanation = "正常" if s1_discard < 0.1 else "较高" if s1_discard < 0.3 else "很高"
                f.write(f"{i+1}\t{s1_discard:.2%}\t{s2_discard:.2%}\t{explanation}\n")
        
        # 总体评估
        f.write(f"\n=== 总体评估 ===\n")
        
        # 计算平均指标
        avg_s1_train = np.mean([r['train_loss'] for r in strategy1_results])
        avg_s2_train = np.mean([r['train_loss'] for r in strategy2_results])
        avg_s1_test = np.mean([r['test_loss'] for r in strategy1_results])
        avg_s2_test = np.mean([r['test_loss'] for r in strategy2_results])
        
        f.write(f"平均训练损失:\n")
        f.write(f"  {strategy1_name}: {avg_s1_train:.4f}\n")
        f.write(f"  {strategy2_name}: {avg_s2_train:.4f}\n")
        f.write(f"  差异: {avg_s1_train - avg_s2_train:+.4f}\n\n")
        
        f.write(f"平均测试损失:\n")
        f.write(f"  {strategy1_name}: {avg_s1_test:.4f}\n")
        f.write(f"  {strategy2_name}: {avg_s2_test:.4f}\n")
        f.write(f"  差异: {avg_s1_test - avg_s2_test:+.4f}\n\n")
        
        # 判断最佳策略
        if avg_s1_test < avg_s2_test:
            f.write(f"🏆 最佳策略: {strategy1_name}\n")
            f.write("   理由: 在测试集上表现更好，泛化能力更强\n")
        else:
            f.write(f"🏆 最佳策略: {strategy2_name}\n")
            f.write("   理由: 在测试集上表现更好，泛化能力更强\n")
        
        # 过拟合分析
        avg_s1_overfit = avg_s1_train - avg_s1_test
        avg_s2_overfit = avg_s2_train - avg_s2_test
        
        f.write(f"\n过拟合分析:\n")
        f.write(f"  {strategy1_name}过拟合程度: {avg_s1_overfit:.4f}\n")
        f.write(f"  {strategy2_name}过拟合程度: {avg_s2_overfit:.4f}\n")
        
        if avg_s1_overfit < avg_s2_overfit:
            f.write(f"  {strategy1_name}过拟合程度更低，泛化能力更好\n")
        else:
            f.write(f"  {strategy2_name}过拟合程度更低，泛化能力更好\n")
    
    print(f"策略比较报告已生成: {report_path}")
    
    # 同时保存NumPy格式的结果用于进一步分析
    np.save(os.path.join(output_dir, f'{strategy1_name.lower()}_vs_{strategy2_name.lower()}_results.npy'), 
            {'strategy1': strategy1_results, 'strategy2': strategy2_results})
    print(f"NumPy格式结果已保存到: {output_dir}")

def generate_three_way_comparison_report(mystrategy_results, baseline_results, sliding_window_results, output_dir):
    """生成三策略比较报告"""
    report_path = os.path.join(output_dir, 'three_way_strategy_comparison_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== 三策略比较报告 ===\n\n")
        f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"训练轮数: {len(mystrategy_results)}\n\n")
        
        # 训练集损失比较
        f.write("=== 训练集损失比较 ===\n")
        f.write("Epoch\tMyStrategy\tBaseline\tSlidingWindow\t最佳策略\n")
        f.write("-" * 70 + "\n")
        
        for i in range(len(mystrategy_results)):
            ms_train = mystrategy_results[i]['train_loss']
            bl_train = baseline_results[i]['train_loss']
            sw_train = sliding_window_results[i]['train_loss']
            
            min_loss = min(ms_train, bl_train, sw_train)
            if min_loss == ms_train:
                best = "MyStrategy"
            elif min_loss == bl_train:
                best = "Baseline"
            else:
                best = "SlidingWindow"
            
            f.write(f"{i+1}\t{ms_train:.4f}\t{bl_train:.4f}\t{sw_train:.4f}\t{best}\n")
        
        # 测试集损失比较
        f.write("\n=== 测试集损失比较 ===\n")
        f.write("Epoch\tMyStrategy\tBaseline\tSlidingWindow\t最佳策略\n")
        f.write("-" * 70 + "\n")
        
        for i in range(len(mystrategy_results)):
            ms_test = mystrategy_results[i]['test_loss']
            bl_test = baseline_results[i]['test_loss']
            sw_test = sliding_window_results[i]['test_loss']
            
            min_loss = min(ms_test, bl_test, sw_test)
            if min_loss == ms_test:
                best = "MyStrategy"
            elif min_loss == bl_test:
                best = "Baseline"
            else:
                best = "SlidingWindow"
            
            f.write(f"{i+1}\t{ms_test:.4f}\t{bl_test:.4f}\t{sw_test:.4f}\t{best}\n")
        
        # 总体评估
        f.write("\n=== 总体评估 ===\n")
        
        # 计算平均指标
        avg_ms_train = np.mean([r['train_loss'] for r in mystrategy_results])
        avg_bl_train = np.mean([r['train_loss'] for r in baseline_results])
        avg_sw_train = np.mean([r['train_loss'] for r in sliding_window_results])
        
        avg_ms_test = np.mean([r['test_loss'] for r in mystrategy_results])
        avg_bl_test = np.mean([r['test_loss'] for r in baseline_results])
        avg_sw_test = np.mean([r['test_loss'] for r in sliding_window_results])
        
        f.write(f"平均训练损失:\n")
        f.write(f"  MyStrategy: {avg_ms_train:.4f}\n")
        f.write(f"  Baseline: {avg_bl_train:.4f}\n")
        f.write(f"  SlidingWindow: {avg_sw_train:.4f}\n\n")
        
        f.write(f"平均测试损失:\n")
        f.write(f"  MyStrategy: {avg_ms_test:.4f}\n")
        f.write(f"  Baseline: {avg_bl_test:.4f}\n")
        f.write(f"  SlidingWindow: {avg_sw_test:.4f}\n\n")
        
        # 判断最佳策略
        test_losses = [avg_ms_test, avg_bl_test, avg_sw_test]
        strategy_names = ["MyStrategy", "Baseline", "SlidingWindow"]
        best_strategy = strategy_names[np.argmin(test_losses)]
        
        f.write(f"🏆 最佳策略: {best_strategy}\n")
        f.write("   理由: 在测试集上表现最好，泛化能力最强\n")
        
        # 过拟合分析
        f.write(f"\n过拟合分析:\n")
        ms_overfit = avg_ms_train - avg_ms_test
        bl_overfit = avg_bl_train - avg_bl_test
        sw_overfit = avg_sw_train - avg_sw_test
        
        f.write(f"  MyStrategy过拟合程度: {ms_overfit:.4f}\n")
        f.write(f"  Baseline过拟合程度: {bl_overfit:.4f}\n")
        f.write(f"  SlidingWindow过拟合程度: {sw_overfit:.4f}\n")
        
        overfits = [ms_overfit, bl_overfit, sw_overfit]
        best_overfit = strategy_names[np.argmin(overfits)]
        f.write(f"  {best_overfit}过拟合程度最低，泛化能力最好\n")
    
    print(f"三策略比较报告已生成: {report_path}")
    
    # 保存NumPy格式的结果
    np.save(os.path.join(output_dir, 'three_way_comparison_results.npy'), {
        'mystrategy': mystrategy_results,
        'baseline': baseline_results,
        'sliding_window': sliding_window_results
    })
    print(f"三策略比较NumPy结果已保存到: {output_dir}")

if __name__ == "__main__":
    main()
