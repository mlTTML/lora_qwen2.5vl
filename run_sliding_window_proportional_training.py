#!/usr/bin/env python3
"""
自动执行sliding_window_proportional策略训练的脚本
proportion_factor从2开始，每次加1，直到15
"""

import os
import subprocess
import time
from datetime import datetime

def run_training(proportion_factor: int, output_dir: str):
    """执行单次训练"""
    
    # 构建训练命令
    cmd = [
        "python", "train.py",
        "--strategy", "sliding_window_proportional",
        "--data_path", "/home/tione/notebook/qwen_2.5vl/5_image_descriptions_adderror_fromfile.json",
        "--output_dir", output_dir,
        "--model_name", "/home/tione/notebook/qwen_2.5vl/Qwen2.5-VL-7B-Instruct",
        "--batch_size", "1",
        "--epochs", "1",
        "--learning_rate", "1e-4",
        "--weight_decay", "0.01",
        "--max_length", "640",
        "--test_size", "0.2",
        "--random_state", "42",
        "--lora_rank", "8",
        "--lora_alpha", "16",
        "--lora_dropout", "0.1",
        "--window_size", "5",
        "--proportion_factor", str(proportion_factor)
    ]
    
    print(f"\n{'='*60}")
    print(f"开始训练: proportion_factor = {proportion_factor}")
    print(f"输出目录: {output_dir}")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    # 记录开始时间
    start_time = time.time()
    
    try:
        # 执行训练命令
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # 计算训练时间
        training_time = time.time() - start_time
        
        print(f"✅ 训练完成: proportion_factor = {proportion_factor}")
        print(f"⏱️  训练时间: {training_time:.2f} 秒")
        print(f"📁 输出目录: {output_dir}")
        
        print(f"📝 详细日志已保存到: {output_dir}")
        return True, training_time
        
    except subprocess.CalledProcessError as e:
        training_time = time.time() - start_time
        print(f"❌ 训练失败: proportion_factor = {proportion_factor}")
        print(f"⏱️  运行时间: {training_time:.2f} 秒")
        print(f"错误代码: {e.returncode}")
        print(f"错误输出: {e.stderr}")
        
        print(f"📝 错误信息已记录，详细日志在: {output_dir}")
        return False, training_time

def main():
    """主函数"""
    print("=== 自动执行sliding_window_proportional策略训练 ===")
    print("proportion_factor范围: 2-15")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 训练参数范围
    proportion_factors = list(range(13, 16))  # 13到15
    
    # 统计信息
    total_training = len(proportion_factors)
    successful_training = 0
    failed_training = 0
    total_time = 0
    
    # 训练结果记录
    training_results = []
    
    # 依次执行训练
    for i, proportion_factor in enumerate(proportion_factors, 1):
        print(f"\n进度: {i}/{total_training}")
        
        # 构建输出目录名
        output_dir = f"./sliding_window_proportional_{proportion_factor}"
        
        # 执行训练
        success, training_time = run_training(proportion_factor, output_dir)
        
        # 记录结果
        result = {
            'proportion_factor': proportion_factor,
            'output_dir': output_dir,
            'success': success,
            'training_time': training_time,
            'timestamp': datetime.now().isoformat()
        }
        training_results.append(result)
        
        if success:
            successful_training += 1
            total_time += training_time
        else:
            failed_training += 1
        
        # 在训练之间稍作休息，避免资源冲突
        if i < total_training:
            print("⏳ 等待5秒后继续下一个训练...")
            time.sleep(5)
    
    # 打印总结
    print(f"\n{'='*60}")
    print("🎯 训练总结")
    print(f"{'='*60}")
    print(f"总训练次数: {total_training}")
    print(f"成功次数: {successful_training}")
    print(f"失败次数: {failed_training}")
    print(f"成功率: {successful_training/total_training*100:.1f}%")
    print(f"总训练时间: {total_time:.2f} 秒")
    print(f"平均训练时间: {total_time/max(1, successful_training):.2f} 秒")
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 保存总结报告
    summary_file = "training_summary.json"
    
    # 分类成功和失败的训练
    successful_factors = [r['proportion_factor'] for r in training_results if r['success']]
    failed_factors = [r['proportion_factor'] for r in training_results if not r['success']]
    
    summary = {
        'total_training': total_training,
        'successful_training': successful_training,
        'failed_training': failed_training,
        'success_rate': successful_training/total_training*100,
        'total_time': total_time,
        'average_time': total_time/max(1, successful_training),
        'start_time': training_results[0]['timestamp'] if training_results else None,
        'end_time': datetime.now().isoformat(),
        'successful_factors': successful_factors,
        'failed_factors': failed_factors,
        'training_results': training_results
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        import json
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 总结报告已保存到: {summary_file}")
    
    # 显示成功和失败的训练
    if successful_factors:
        print(f"\n✅ 成功完成的训练:")
        for factor in successful_factors:
            print(f"  - proportion_factor = {factor}")
    
    if failed_factors:
        print(f"\n❌ 失败的训练:")
        for factor in failed_factors:
            print(f"  - proportion_factor = {factor}")
    
    print(f"\n🎯 总结:")
    print(f"  成功: {len(successful_factors)} 个")
    print(f"  失败: {len(failed_factors)} 个")
    print(f"  成功率: {successful_training/total_training*100:.1f}%")
    
    print(f"\n🎉 训练脚本执行完成！")

if __name__ == "__main__":
    main() 