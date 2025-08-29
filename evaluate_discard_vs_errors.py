#!/usr/bin/env python3
"""
评估丢弃策略与错误数据的匹配度

度量：
- precision_discard_is_error = 被丢弃中属于错误数据的比例（精确率）
- recall_error_was_discarded = 错误数据被丢弃的比例（召回率）

输入：
- --error_samples: 错误数据集合文件（如 /home/tione/notebook/qwen_2.5vl/10_common_error_samples.json）
- --log_path: 策略异常日志文件或目录（如 .../sliding_window_proportional_outlier_log.json 或其所在目录）
- 可多次传入 --log_path 聚合多个运行；也可给目录以递归查找 *outlier_log.json

用法示例：
python evaluate_discard_vs_errors.py \
  --error_samples /home/tione/notebook/qwen_2.5vl/10_common_error_samples.json \
  --log_path /home/tione/notebook/qwen_2.5vl/results/strategy_training/tensorboard/california_1755965877/SlidingWindowProportional_Factor12.0/sliding_window_proportional_outlier_log.json \
  --output ./10_discard_eval_report.json

或评估一个目录下所有运行：
python evaluate_discard_vs_errors.py \
  --error_samples /home/tione/notebook/qwen_2.5vl/5_common_error_samples.json \
  --log_path /home/tione/notebook/qwen_2.5vl/results/strategy_training/tensorboard \
  --output ./5_discard_eval_report.json
"""

import os
import json
import argparse
from typing import List, Dict, Any, Set, Tuple


def _normalize_path(p: str) -> str:
    if not isinstance(p, str) or not p:
        return None
    p = p.replace('\\', '/').strip()
    try:
        p = os.path.normpath(p)
    except Exception:
        pass
    return p.replace('\\', '/').lower()


def _gather_log_files(paths: List[str]) -> List[str]:
    files: List[str] = []
    for p in paths:
        if not os.path.exists(p):
            continue
        if os.path.isfile(p):
            files.append(p)
        else:
            # 目录：递归查找 *outlier_log.json
            for root, _, fnames in os.walk(p):
                for fn in fnames:
                    if fn.endswith('outlier_log.json'):
                        files.append(os.path.join(root, fn))
    # 去重并保持稳定顺序
    seen = set()
    uniq = []
    for f in files:
        if f not in seen:
            seen.add(f)
            uniq.append(f)
    return uniq


def _load_error_paths(error_samples_file: str) -> Set[str]:
    data = json.load(open(error_samples_file, 'r', encoding='utf-8'))
    error_paths: Set[str] = set()
    if isinstance(data, list):
        for item in data:
            # 兼容简单或完整格式
            p = item.get('image_path') if isinstance(item, dict) else None
            npth = _normalize_path(p)
            if npth:
                error_paths.add(npth)
    elif isinstance(data, dict) and 'error_samples' in data:
        for item in data['error_samples']:
            p = item.get('image_path')
            npth = _normalize_path(p)
            if npth:
                error_paths.add(npth)
    return error_paths


def _extract_paths_from_log_entry(entry: Dict[str, Any]) -> List[str]:
    paths: List[str] = []
    # 1) 顶层 image_path 可能是字符串或列表
    v = entry.get('image_path')
    if isinstance(v, str):
        paths.append(v)
    elif isinstance(v, list):
        for x in v:
            if isinstance(x, str):
                paths.append(x)
    # 2) batch_info.image_path 兜底
    b = entry.get('batch_info', {}) if isinstance(entry, dict) else {}
    bv = b.get('image_path') if isinstance(b, dict) else None
    if isinstance(bv, str):
        paths.append(bv)
    elif isinstance(bv, list):
        for x in bv:
            if isinstance(x, str):
                paths.append(x)
    return paths


def _load_discarded_paths_from_log(log_file: str) -> Set[str]:
    try:
        data = json.load(open(log_file, 'r', encoding='utf-8'))
    except Exception as e:
        print(f"跳过无法读取的日志: {log_file} ({e})")
        return set()
    paths: Set[str] = set()
    if isinstance(data, list):
        for entry in data:
            for p in _extract_paths_from_log_entry(entry):
                npth = _normalize_path(p)
                if npth:
                    paths.add(npth)
    elif isinstance(data, dict):
        # 某些实现可能存为字典，尝试取 'outliers'
        for entry in data.get('outliers', []):
            for p in _extract_paths_from_log_entry(entry):
                npth = _normalize_path(p)
                if npth:
                    paths.add(npth)
    return paths


def evaluate(error_samples_file: str, log_paths: List[str]) -> Dict[str, Any]:
    error_paths = _load_error_paths(error_samples_file)
    logs = _gather_log_files(log_paths)
    results: Dict[str, Any] = {
        'error_samples_total': len(error_paths),
        'runs': [],
        'aggregate': {}
    }

    # 聚合集合
    union_discarded: Set[str] = set()
    union_intersection: Set[str] = set()

    for lf in logs:
        discarded_paths = _load_discarded_paths_from_log(lf)
        inter = discarded_paths & error_paths
        false_pos = discarded_paths - error_paths
        false_neg = error_paths - discarded_paths
        precision = (len(inter) / len(discarded_paths)) if discarded_paths else 0.0
        recall = (len(inter) / len(error_paths)) if error_paths else 0.0
        results['runs'].append({
            'log_file': lf,
            'discarded_total': len(discarded_paths),
            'error_in_discarded': len(inter),
            'precision_discard_is_error': precision,
            'recall_error_was_discarded': recall,
            'false_positive': len(false_pos),
            'false_negative': len(false_neg)
        })
        union_discarded |= discarded_paths
        union_intersection |= inter

    # 总体（并集）
    agg_precision = (len(union_intersection) / len(union_discarded)) if union_discarded else 0.0
    agg_recall = (len(union_intersection) / len(error_paths)) if error_paths else 0.0
    results['aggregate'] = {
        'logs_count': len(logs),
        'discarded_union': len(union_discarded),
        'error_intersection_union': len(union_intersection),
        'precision_discard_is_error': agg_precision,
        'recall_error_was_discarded': agg_recall
    }
    return results


def main():
    parser = argparse.ArgumentParser(description='评估丢弃策略与错误数据的匹配度（精确率/召回率）')
    parser.add_argument('--error_samples', required=True, help='错误样本集合文件，如 error_samples.json')
    parser.add_argument('--log_path', action='append', required=True, help='异常日志路径（文件或目录），可多次传入')
    parser.add_argument('--output', default='', help='评估结果输出到JSON文件（可选）')
    args = parser.parse_args()

    results = evaluate(args.error_samples, args.log_path)

    # 打印概览
    print('\n=== 评估结果（按运行）===')
    for run in results['runs']:
        print(f"- {run['log_file']}")
        print(f"  丢弃总数: {run['discarded_total']}")
        print(f"  丢弃中错误数: {run['error_in_discarded']}")
        print(f"  精确率(丢弃是错误): {run['precision_discard_is_error']:.4f}")
        print(f"  召回率(错误被丢弃): {run['recall_error_was_discarded']:.4f}")
        print(f"  误丢(假阳)数: {run['false_positive']} | 漏丢(假阴)数: {run['false_negative']}")

    agg = results['aggregate']
    print('\n=== 汇总(并集) ===')
    print(f"  运行数: {agg['logs_count']}")
    print(f"  并集丢弃数: {agg['discarded_union']}")
    print(f"  并集中错误数: {agg['error_intersection_union']}")
    print(f"  精确率(丢弃是错误): {agg['precision_discard_is_error']:.4f}")
    print(f"  召回率(错误被丢弃): {agg['recall_error_was_discarded']:.4f}")

    if args.output:
        # 生成简化版本，直接对应控制台输出格式
        simple_results = {
            "评估结果按运行": [],
            "汇总并集": {
                "运行数": agg['logs_count'],
                "并集丢弃数": agg['discarded_union'], 
                "并集中错误数": agg['error_intersection_union'],
                "精确率丢弃是错误": round(agg['precision_discard_is_error'], 4),
                "召回率错误被丢弃": round(agg['recall_error_was_discarded'], 4)
            }
        }
        
        for run in results['runs']:
            simple_results["评估结果按运行"].append({
                "日志文件": run['log_file'],
                "丢弃总数": run['discarded_total'],
                "丢弃中错误数": run['error_in_discarded'], 
                "精确率丢弃是错误": round(run['precision_discard_is_error'], 4),
                "召回率错误被丢弃": round(run['recall_error_was_discarded'], 4),
                "误丢假阳数": run['false_positive'],
                "漏丢假阴数": run['false_negative']
            })
            
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(simple_results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {args.output}")


if __name__ == '__main__':
    main() 