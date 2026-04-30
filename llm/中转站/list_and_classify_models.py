import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("MODELSCOPE_API_KEY"),
    base_url=os.getenv("MODELSCOPE_BASE_URL")
)

def classify_and_print_models():
    try:
        print("正在获取 ModelScope 模型列表并分类...")
        response = client.models.list()
        all_models = [model.id for model in response.data]
        
        # 定义多模态关键词（视觉、音频等）
        multimodal_keywords = ['vl', 'vision', 'image', 'audio', 'video']
        
        multimodal_models = []
        text_models = []
        
        for m in sorted(all_models):
            if any(k in m.lower() for k in multimodal_keywords):
                multimodal_models.append(m)
            else:
                text_models.append(m)
        
        print("\n" + "="*50)
        print(f"🎨 多模态模型 (共 {len(multimodal_models)} 个)")
        print("="*50)
        for m in multimodal_models:
            print(f"  [Multimodal] {m}")
            
        print("\n" + "="*50)
        print(f"📝 文本及其他模型 (共 {len(text_models)} 个)")
        print("="*50)
        for m in text_models:
            print(f"  [Text] {m}")
            
        print("\n" + "="*50)
        print(f"统计总计: {len(all_models)} 个模型")
        
    except Exception as e:
        print(f"分类获取失败: {e}")

if __name__ == "__main__":
    classify_and_print_models()
