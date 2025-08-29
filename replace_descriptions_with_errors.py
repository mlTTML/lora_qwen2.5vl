#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
随机替换数据集描述脚本

核心目的：
1. 将数据集中的正确描述随机替换为错误描述
2. 用于测试模型对错误数据的鲁棒性和处理能力
3. 生成包含错误样本的训练数据，用于对抗训练
4. 评估模型在噪声数据下的表现

使用方法：
python replace_descriptions_with_errors.py \
    --input /path/to/image_descriptions.json \    # 原始数据集JSON文件
    --errors /path/to/error.txt \                 # 错误描述文本文件（每行一条）
    --output /path/to/output.json \              # 输出包含错误的数据集
    --log /path/to/log.json \                    # 替换日志文件
    --percent 0.1 \                              # 替换比例(0-1)，可选
    --count 100 \                                # 替换条数，与percent互斥
    --seed 42 \                                  # 随机种子
    --strip_prefix                               # 去除错误描述中的序号前缀

参数说明：
- --input: 输入数据集JSON路径
- --errors: 错误描述文本文件，每行一条错误描述
- --output: 输出数据集JSON路径（包含错误描述的数据集）
- --log: 替换日志输出路径（JSON格式,记录着被替换的样本的索引、原描述、新描述）
- --percent: 替换比例(0-1)，与--count互斥
- --count: 替换条数，与--percent互斥
- --seed: 随机种子，确保结果可重现
- --strip_prefix: 去除error.txt每行开头的序号/项目符号等前缀

例：
python replace_descriptions_with_errors.py \
    --input /home/tione/notebook/qwen_2.5vl/image_descriptions.json \
    --errors /home/tione/notebook/qwen_2.5vl/error.txt \
    --output /home/tione/notebook/qwen_2.5vl/5_image_descriptions_adderror_fromfile.json \
    --log /home/tione/notebook/qwen_2.5vl/5_replaced_indices.json \
    --percent 0.05 \
    --seed 42 \
    --strip_prefix
"""

import argparse
import json
import os
import random
import re
from typing import List, Dict, Any, Tuple


def read_json(path: str) -> List[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError('输入JSON应为列表(list)格式')
    return data


def normalize_error_line(line: str, strip_prefix: bool) -> str:
    if not strip_prefix:
        return line
    s = line
    # 常见前缀形式："32. ", "1: ", "2) ", "3] ", "（4） ", "(5) ", "- ", "• "，以及中文序号如"一、"等
    patterns = [
        r"^\s*\d+\s*[\.、:：\)\]]\s*",     # 数字+分隔符
        r"^\s*[（(]\s*\d+\s*[）)]\s*",      # 括号包裹数字
        r"^\s*[-•—]\s+",                      # 项目符号
        r"^\s*[一二三四五六七八九十百千]+[、.]\s*"  # 中文数字 + 顿号/点
    ]
    for p in patterns:
        s = re.sub(p, "", s)
    return s.strip()


def read_error_lines(path: str, strip_prefix: bool = False) -> List[str]:
    lines: List[str] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(normalize_error_line(line, strip_prefix))
    if not lines:
        raise ValueError('error.txt 内容为空或仅包含空行')
    return lines


def extract_image_path(sample: Dict[str, Any]) -> str:
    # 兼容两种结构：
    # 1) messages -> user.content[].image
    # 2) image_path
    try:
        msgs = sample.get('messages', [])
        for msg in msgs:
            if msg.get('role') == 'user':
                for c in msg.get('content', []):
                    if c.get('type') == 'image':
                        img = c.get('image')
                        if img:
                            return str(img)
    except Exception:
        pass
    return str(sample.get('image_path', ''))


def get_current_description(sample: Dict[str, Any]) -> Tuple[str, str]:
    # 返回 (desc, field_type) 其中 field_type ∈ {'target.description', 'caption', 'none'}
    try:
        tgt = sample.get('target', {})
        if isinstance(tgt, dict) and 'description' in tgt:
            return str(tgt.get('description', '')), 'target.description'
    except Exception:
        pass
    if 'caption' in sample:
        return str(sample.get('caption', '')), 'caption'
    return '', 'none'


def set_description(sample: Dict[str, Any], new_desc: str) -> str:
    # 优先写入 target.description；若不存在且存在 caption，则写 caption；否则创建 target.description
    try:
        if 'target' in sample and isinstance(sample['target'], dict):
            sample['target']['description'] = new_desc
            return 'target.description'
    except Exception:
        pass
    if 'caption' in sample:
        sample['caption'] = new_desc
        return 'caption'
    # 创建 target.description
    sample['target'] = sample.get('target', {}) if isinstance(sample.get('target', {}), dict) else {}
    sample['target']['description'] = new_desc
    return 'target.description'


def choose_replacement_indices(n_items: int, n_replace: int, rng: random.Random) -> List[int]:
    n = min(n_items, n_replace)
    return rng.sample(range(n_items), n)


def main():
    ap = argparse.ArgumentParser(description='随机用 error.txt 的内容替换数据集 JSON 中的描述，并记录替换日志')
    ap.add_argument('--input', required=True, help='输入数据集JSON路径，例如 /root/autodl-tmp/image_descriptions.json')
    ap.add_argument('--errors', required=True, help='错误描述文本文件，每行一条，例如 /root/autodl-tmp/error.txt')
    ap.add_argument('--output', required=True, help='输出数据集JSON路径，例如 /root/autodl-tmp/image_descriptions_adderror_fromfile.json')
    ap.add_argument('--log', required=True, help='替换日志输出路径（JSON），例如 /root/autodl-tmp/replaced_indices.json')
    ap.add_argument('--percent', type=float, default=None, help='替换比例(0-1)。不指定则默认按 error.txt 行数替换')
    ap.add_argument('--count', type=int, default=None, help='替换条数。与 --percent 互斥')
    ap.add_argument('--seed', type=int, default=42, help='随机种子')
    ap.add_argument('--strip_prefix', action='store_true', help='去除 error.txt 每行开头的序号/项目符号等前缀')
    args = ap.parse_args()

    if (args.percent is not None) and (args.count is not None):
        raise ValueError('--percent 与 --count 不能同时设置')
    if args.percent is not None and not (0.0 < args.percent <= 1.0):
        raise ValueError('--percent 应在 (0, 1] 区间')

    rng = random.Random(args.seed)

    data = read_json(args.input)
    err_lines = read_error_lines(args.errors, strip_prefix=args.strip_prefix)

    # 计算替换数量
    if args.count is not None:
        n_replace = min(max(0, int(args.count)), len(data))
    elif args.percent is not None:
        n_replace = int(round(len(data) * float(args.percent)))
        n_replace = max(1, min(n_replace, len(data)))
    else:
        # 默认按错误行数替换（不超过数据集大小）
        n_replace = min(len(err_lines), len(data))

    if n_replace <= 0:
        raise ValueError('替换数量为0，请检查 --count/--percent 或 error.txt 内容')

    # 选择被替换的索引，随机分布
    indices = choose_replacement_indices(len(data), n_replace, rng)

    # 准备要使用的错误描述：随机打乱，按需取前 n_replace 条；若错误条数不足则循环使用
    shuffled_errs = err_lines[:]
    rng.shuffle(shuffled_errs)
    if len(shuffled_errs) < n_replace:
        # 循环复用
        times = (n_replace + len(shuffled_errs) - 1) // len(shuffled_errs)
        shuffled_errs = (shuffled_errs * times)[:n_replace]
    else:
        shuffled_errs = shuffled_errs[:n_replace]

    # 执行替换并记录日志
    replace_log: List[Dict[str, Any]] = []
    for idx, new_desc in zip(indices, shuffled_errs):
        sample = data[idx]
        old_desc, field_type = get_current_description(sample)
        field_written = set_description(sample, new_desc)
        replace_log.append({
            'index': idx,
            'image_path': extract_image_path(sample),
            'field_old_type': field_type,
            'field_new_type': field_written,
            'old_description': old_desc,
            'new_description': new_desc
        })

    # 保存新的数据集
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # 保存替换日志
    os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)
    with open(args.log, 'w', encoding='utf-8') as f:
        json.dump({
            'input': os.path.abspath(args.input),
            'errors': os.path.abspath(args.errors),
            'output': os.path.abspath(args.output),
            'seed': args.seed,
            'total_samples': len(data),
            'replaced_count': len(replace_log),
            'indices': sorted([r['index'] for r in replace_log]),
            'details': replace_log
        }, f, ensure_ascii=False, indent=2)

    print(f'完成：共替换 {len(replace_log)} 条，输出文件：{args.output}')
    print(f'替换日志：{args.log}')


if __name__ == '__main__':
    main() 