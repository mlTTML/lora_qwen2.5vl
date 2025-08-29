#!/usr/bin/env python3
"""
找到训练集中被替换的样本。
脚本功能：找出train_data.json和replaced_indices.json的共有内容。

输入文件：
- /home/tione/notebook/qwen_2.5vl/fixed_splits/train_data.json
- /home/tione/notebook/qwen_2.5vl/10_replaced_indices.json

输出文件：
- /home/tione/notebook/qwen_2.5vl/10_common_error_samples.json

例：
python find_common_samples.py \
    --train_data_path /home/tione/notebook/qwen_2.5vl/fixed_splits/train_data.json \
    --replaced_indices_path /home/tione/notebook/qwen_2.5vl/5_replaced_indices.json \
    --output_path /home/tione/notebook/qwen_2.5vl/5_common_error_samples.json
"""

import json
import os
from typing import List, Dict, Any, Set
from datetime import datetime

def load_json_file(file_path: str) -> Any:
    """加载JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载文件 {file_path} 时出错: {e}")
        return None

def save_json_file(data: Any, file_path: str) -> bool:
    """保存JSON文件"""
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"保存文件 {file_path} 时出错: {e}")
        return False

def extract_indices_from_train_data(train_data: List[Dict]) -> Set[int]:
    """从train_data.json中提取索引"""
    indices = set()
    for i, item in enumerate(train_data):
        # 假设train_data中的每个项目都有一个隐含的索引（基于位置）
        # 或者如果数据中有明确的索引字段，使用那个字段
        if isinstance(item, dict):
            # 如果item中有original_index字段，使用它
            if 'original_index' in item:
                indices.add(item['original_index'])
            else:
                # 否则使用列表位置作为索引
                indices.add(i)
        else:
            # 如果item不是字典，使用列表位置
            indices.add(i)
    return indices

def find_common_samples(train_data_path: str, replaced_indices_path: str) -> List[Dict]:
    """找出两个文件的共有样本"""
    print("开始加载文件...")
    
    # 加载train_data.json
    train_data = load_json_file(train_data_path)
    if train_data is None:
        return []
    
    # 加载replaced_indices.json
    replaced_indices_data = load_json_file(replaced_indices_path)
    if replaced_indices_data is None:
        return []
    
    print(f"train_data.json 包含 {len(train_data)} 个样本")
    print(f"replaced_indices.json 包含 {len(replaced_indices_data.get('indices', []))} 个索引")
    
    # 提取replaced_indices中的索引
    replaced_indices_set = set(replaced_indices_data.get('indices', []))
    print(f"replaced_indices 中的索引数量: {len(replaced_indices_set)}")
    
    # 提取train_data中的索引
    train_indices_set = extract_indices_from_train_data(train_data)
    print(f"train_data 中的索引数量: {len(train_indices_set)}")
    
    # 找出共同的索引
    common_indices = train_indices_set.intersection(replaced_indices_set)
    print(f"共同的索引数量: {len(common_indices)}")
    
    # 生成结果
    common_samples = []
    for i, item in enumerate(train_data):
        current_index = None
        if isinstance(item, dict) and 'original_index' in item:
            current_index = item['original_index']
        else:
            current_index = i
            
        if current_index in common_indices:
            # 创建与error_samples.json相同格式的样本
            sample = {
                "image_path": item.get('image_path', ''),
                "caption": item.get('caption', ''),
                "original_index": current_index
            }
            common_samples.append(sample)
    
    print(f"生成的共同样本数量: {len(common_samples)}")
    return common_samples

def main():
    """主函数"""
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='查找训练集中被替换的样本')
    parser.add_argument('--train_data_path', 
                       default="/home/tione/notebook/qwen_2.5vl/fixed_splits/train_data.json",
                       help='训练数据JSON文件路径')
    parser.add_argument('--replaced_indices_path', 
                       default="/home/tione/notebook/qwen_2.5vl/replaced_indices.json",
                       help='替换索引JSON文件路径')
    parser.add_argument('--output_path', 
                       default="/home/tione/notebook/qwen_2.5vl/common_error_samples.json",
                       help='输出JSON文件路径')
    
    args = parser.parse_args()
    
    # 文件路径
    train_data_path = args.train_data_path
    replaced_indices_path = args.replaced_indices_path
    output_path = args.output_path
    
    print("=== 查找共同样本脚本 ===")
    print(f"输入文件1: {train_data_path}")
    print(f"输入文件2: {replaced_indices_path}")
    print(f"输出文件: {output_path}")
    print()
    
    # 检查输入文件是否存在
    if not os.path.exists(train_data_path):
        print(f"错误: 文件不存在 - {train_data_path}")
        return
    
    if not os.path.exists(replaced_indices_path):
        print(f"错误: 文件不存在 - {replaced_indices_path}")
        return
    
    # 查找共同样本
    common_samples = find_common_samples(train_data_path, replaced_indices_path)
    
    if not common_samples:
        print("未找到共同的样本")
        return
    
    # 保存结果
    print(f"\n保存结果到: {output_path}")
    if save_json_file(common_samples, output_path):
        print("✅ 保存成功!")
        print(f"共找到 {len(common_samples)} 个共同样本")
    else:
        print("❌ 保存失败!")

if __name__ == "__main__":
    main() 