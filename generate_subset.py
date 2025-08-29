import json
import random
import os
from datetime import datetime

def generate_subset_with_validation(input_file, output_file, percentage=5, seed=42):
    """
    生成数据子集，包含验证和统计信息
    """
    
    # 设置随机种子
    random.seed(seed)
    
    # 读取原始数据
    print(f"🔄 正在读取 {input_file}...")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return None
    
    total_samples = len(data)
    subset_size = int(total_samples * percentage / 100)
    
    print(f"📊 数据统计:")
    print(f"   原始数据总量: {total_samples:,}")
    print(f"   抽取比例: {percentage}%")
    print(f"   抽取数量: {subset_size:,}")
    print(f"   随机种子: {seed}")
    
    # 随机抽取数据
    subset_data = random.sample(data, subset_size)
    
    # 验证抽取结果
    print(f"\n🔍 验证抽取结果:")
    print(f"   实际抽取数量: {len(subset_data):,}")
    print(f"   抽取比例: {len(subset_data)/total_samples*100:.2f}%")
    
    # 生成输出文件名（包含时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file_with_timestamp = f"{output_file.replace('.json', '')}_{timestamp}.json"
    
    # 保存子集数据
    print(f"\n 正在保存到 {output_file_with_timestamp}...")
    try:
        with open(output_file_with_timestamp, 'w', encoding='utf-8') as f:
            json.dump(subset_data, f, indent=2, ensure_ascii=False)
        print(f"✅ 成功生成子集文件！")
    except Exception as e:
        print(f"❌ 保存文件失败: {e}")
        return None
    
    # 生成统计报告
    report = {
        "生成时间": datetime.now().isoformat(),
        "原始文件": input_file,
        "输出文件": output_file_with_timestamp,
        "原始数据量": total_samples,
        "抽取比例": f"{percentage}%",
        "抽取数量": subset_size,
        "随机种子": seed,
        "文件大小_MB": round(os.path.getsize(output_file_with_timestamp) / (1024*1024), 2)
    }
    
    # 保存统计报告
    report_file = f"{output_file_with_timestamp.replace('.json', '_report.json')}"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n📈 统计报告:")
    for key, value in report.items():
        print(f"   {key}: {value}")
    
    return subset_data

def main():
    # 文件路径
    input_file = "/root/autodl-tmp/image_descriptions_5percent_20250814_205316.json"
    output_file = "/root/autodl-tmp/easy_5percent.json"
    
    # 检查输入文件
    if not os.path.exists(input_file):
        print(f"❌ 错误：输入文件 {input_file} 不存在！")
        return
    
    # 检查文件大小
    file_size_mb = os.path.getsize(input_file) / (1024*1024)
    print(f" 输入文件大小: {file_size_mb:.2f} MB")
    
    # 生成5%的子集
    subset_data = generate_subset_with_validation(
        input_file, 
        output_file, 
        percentage=5, 
        seed=42
    )
    
    if subset_data:
        print(f"\n🎉 子集生成完成！")
        print(f"📁 主文件: {output_file.replace('.json', '')}_*.json")
        print(f" 报告文件: {output_file.replace('.json', '')}_*_report.json")

if __name__ == "__main__":
    main()
