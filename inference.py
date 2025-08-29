import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

def load_model(base_model_path, lora_path):
    """加载微调后的模型"""
    # 加载基础模型
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_path,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # 加载LoRA权重
    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)
    
    return model

def generate_caption(model, processor, image_path, prompt="请描述这张图片的内容。"):
    """生成图片描述"""
    # 加载图片
    image = Image.open(image_path).convert('RGB')
    
    # 构建输入
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt}
        ]
    }]
    
    # 应用聊天模板
    chat_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # 处理输入
    inputs = processor(
        text=[chat_prompt],
        images=[image],
        return_tensors="pt"
    )
    
    # 生成描述
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=processor.tokenizer.eos_token_id
        )
    
    # 解码输出
    generated_text = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取assistant部分的回复
    if "assistant" in generated_text:
        response = generated_text.split("assistant")[-1].strip()
    else:
        response = generated_text
    
    return response

if __name__ == "__main__":
    # 配置路径
    base_model_path = "Qwen/Qwen2.5-VL-7B-Instruct"  # 或你的微调模型路径
    lora_path = "./output/final_model"  # LoRA权重路径
    image_path = "test_image.jpg"  # 测试图片路径
    
    # 加载模型
    print("加载模型...")
    model = load_model(base_model_path, lora_path)
    processor = AutoProcessor.from_pretrained(base_model_path)
    
    # 生成描述
    print("生成图片描述...")
    caption = generate_caption(model, processor, image_path)
    
    print(f"图片描述: {caption}")
