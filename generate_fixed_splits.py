#!/usr/bin/env python3
"""
生成固定的训练集和测试集分割

使用方法:
python generate_fixed_splits.py --data_path /path/to/data.json --output_dir ./fixed_splits --test_size 0.2 --random_state 42

生成的文件:
- train_data.json: 训练集数据
- test_data.json: 测试集数据  
- split_info.json: 分割信息（包含索引、统计等）
"""

import os
import json
import argparse
import random
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image

def extract_image_path(item):
    """从数据项中提取图片路径"""
    try:
        for message in item.get('messages', []):
            if message.get('role') == 'user':
                for content in message.get('content', []):
                    if content.get('type') == 'image':
                        return content.get('image')
        return None
    except:
        return None

def extract_caption(item):
    """从数据项中提取描述"""
    try:
        return item.get('target', {}).get('description', '').strip()
    except:
        return ''

def process_path_separators(image_path):
    """处理路径分隔符"""
    return image_path.replace('\\', '/')

def validate_sample(image_path, caption, json_dir):
    """验证样本有效性"""
    # 处理路径
    image_path = process_path_separators(image_path)
    if not os.path.isabs(image_path):
        image_path = os.path.join(json_dir, image_path)
    
    # 检查文件存在性
    if not os.path.exists(image_path):
        print(f"警告: 图片文件不存在: {image_path}")
        return False, None
    
    # 检查文件格式
    if not image_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        print(f"警告: 不支持的图片格式: {image_path}")
        return False, None
    
    # 检查描述长度
    if len(caption) < 5:
        print(f"警告: 描述过短: {caption[:20]}...")
        return False, None
    
    if len(caption) > 500:
        print(f"警告: 描述过长，截断到500字符")
        caption = caption[:500]
    
    return True, image_path

def load_and_validate_data(data_path):
    """加载并验证数据"""
    print(f"加载数据文件: {data_path}")
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError("JSON数据应该是列表格式")
        
        json_dir = os.path.dirname(os.path.abspath(data_path))
        valid_samples = []
        invalid_count = 0
        
        for i, item in enumerate(data):
            try:
                # 提取数据
                image_path = extract_image_path(item)
                if not image_path:
                    invalid_count += 1
                    continue
                
                caption = extract_caption(item)
                if not caption:
                    invalid_count += 1
                    continue
                
                # 验证样本
                is_valid, processed_path = validate_sample(image_path, caption, json_dir)
                if is_valid:
                    valid_samples.append({
                        'image_path': processed_path,
                        'caption': caption,
                        'original_index': i  # 保存原始索引
                    })
                else:
                    invalid_count += 1
                    
            except Exception as e:
                print(f"处理第{i}项时出错: {e}")
                invalid_count += 1
                continue
        
        print(f"数据验证完成:")
        print(f"  总样本数: {len(data)}")
        print(f"  有效样本数: {len(valid_samples)}")
        print(f"  无效样本数: {invalid_count}")
        
        if len(valid_samples) == 0:
            raise ValueError("没有找到有效的训练样本")
        
        return valid_samples
        
    except Exception as e:
        print(f"加载数据失败: {e}")
        raise

def generate_fixed_splits(data_path, output_dir, test_size=0.2, random_state=42):
    """生成固定的训练集和测试集分割"""
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载并验证数据
    valid_samples = load_and_validate_data(data_path)
    
    # 设置随机种子确保可重复性
    random.seed(random_state)
    np.random.seed(random_state)
    
    # 生成索引
    indices = list(range(len(valid_samples)))
    
    # 分割数据
    train_indices, test_indices = train_test_split(
        indices, 
        test_size=test_size, 
        random_state=random_state,
        shuffle=True
    )
    
    # 提取训练集和测试集
    train_samples = [valid_samples[i] for i in train_indices]
    test_samples = [valid_samples[i] for i in test_indices]
    
    # 保存训练集
    train_path = os.path.join(output_dir, 'train_data.json')
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_samples, f, ensure_ascii=False, indent=2)
    
    # 保存测试集
    test_path = os.path.join(output_dir, 'test_data.json')
    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(test_samples, f, ensure_ascii=False, indent=2)
    
    # 生成分割信息
    split_info = {
        'original_data_path': data_path,
        'test_size': test_size,
        'random_state': random_state,
        'total_samples': len(valid_samples),
        'train_samples': len(train_samples),
        'test_samples': len(test_samples),
        'train_indices': train_indices,
        'test_indices': test_indices,
        'train_original_indices': [valid_samples[i]['original_index'] for i in train_indices],
        'test_original_indices': [valid_samples[i]['original_index'] for i in test_indices],
        'generation_timestamp': str(np.datetime64('now')),
        'statistics': {
            'avg_caption_length_train': np.mean([len(s['caption']) for s in train_samples]),
            'avg_caption_length_test': np.mean([len(s['caption']) for s in test_samples]),
            'min_caption_length_train': min([len(s['caption']) for s in train_samples]),
            'max_caption_length_train': max([len(s['caption']) for s in train_samples]),
            'min_caption_length_test': min([len(s['caption']) for s in test_samples]),
            'max_caption_length_test': max([len(s['caption']) for s in test_samples])
        }
    }
    
    # 保存分割信息
    info_path = os.path.join(output_dir, 'split_info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)
    
    # 打印结果
    print(f"\n=== 固定分割生成完成 ===")
    print(f"输出目录: {output_dir}")
    print(f"训练集: {len(train_samples)} 样本 -> {train_path}")
    print(f"测试集: {len(test_samples)} 样本 -> {test_path}")
    print(f"分割信息: {info_path}")
    print(f"随机种子: {random_state}")
    print(f"测试集比例: {test_size:.1%}")
    
    print(f"\n=== 数据统计 ===")
    print(f"训练集平均描述长度: {split_info['statistics']['avg_caption_length_train']:.1f}")
    print(f"测试集平均描述长度: {split_info['statistics']['avg_caption_length_test']:.1f}")
    print(f"训练集描述长度范围: {split_info['statistics']['min_caption_length_train']} - {split_info['statistics']['max_caption_length_train']}")
    print(f"测试集描述长度范围: {split_info['statistics']['min_caption_length_test']} - {split_info['statistics']['max_caption_length_test']}")
    
    return train_path, test_path, split_info

def verify_splits(output_dir):
    """验证生成的分割是否正确"""
    print(f"\n=== 验证分割结果 ===")
    
    # 加载分割信息
    info_path = os.path.join(output_dir, 'split_info.json')
    with open(info_path, 'r', encoding='utf-8') as f:
        split_info = json.load(f)
    
    # 加载训练集和测试集
    train_path = os.path.join(output_dir, 'train_data.json')
    test_path = os.path.join(output_dir, 'test_data.json')
    
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    with open(test_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # 验证数量
    assert len(train_data) == split_info['train_samples'], "训练集样本数不匹配"
    assert len(test_data) == split_info['test_samples'], "测试集样本数不匹配"
    assert len(train_data) + len(test_data) == split_info['total_samples'], "总样本数不匹配"
    
    # 验证无重复
    train_image_paths = set(s['image_path'] for s in train_data)
    test_image_paths = set(s['image_path'] for s in test_data)
    intersection = train_image_paths & test_image_paths
    
    if intersection:
        print(f"警告: 发现重复的图片路径: {intersection}")
    else:
        print("✓ 训练集和测试集无重复")
    
    # 验证文件存在性
    all_paths = list(train_image_paths) + list(test_image_paths)
    missing_files = [p for p in all_paths if not os.path.exists(p)]
    
    if missing_files:
        print(f"警告: 发现缺失的图片文件: {missing_files[:5]}...")
    else:
        print("✓ 所有图片文件都存在")
    
    print("✓ 分割验证通过")

def main():
    parser = argparse.ArgumentParser(description='生成固定的训练集和测试集分割')
    
    # 必需参数
    parser.add_argument('--data_path', type=str, required=True, 
                       help='原始数据文件路径（JSON格式）')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='输出目录路径')
    
    # 可选参数
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='测试集比例（0.0-1.0，默认0.2）')
    parser.add_argument('--random_state', type=int, default=42,
                       help='随机种子（默认42）')
    parser.add_argument('--verify', action='store_true',
                       help='生成后验证分割结果')
    
    args = parser.parse_args()
    
    # 参数验证
    if not os.path.exists(args.data_path):
        print(f"错误: 数据文件不存在: {args.data_path}")
        return
    
    if args.test_size <= 0 or args.test_size >= 1:
        print(f"错误: test_size 必须在 0-1 之间，当前值: {args.test_size}")
        return
    
    # 生成固定分割
    try:
        train_path, test_path, split_info = generate_fixed_splits(
            args.data_path, 
            args.output_dir, 
            args.test_size, 
            args.random_state
        )
        
        # 验证分割
        if args.verify:
            verify_splits(args.output_dir)
        
        print(f"\n=== 使用说明 ===")
        print(f"1. 在训练时使用 --data_path {train_path} 训练模型")
        print(f"2. 在评估时使用 --data_path {test_path} 测试模型")
        print(f"3. 或者修改 train.py 支持直接指定训练集和测试集文件")
        print(f"4. 确保所有策略使用相同的分割文件以确保公平比较")
        
    except Exception as e:
        print(f"生成固定分割失败: {e}")
        return

if __name__ == "__main__":
    main() 