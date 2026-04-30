from openai import OpenAI

client = OpenAI(
    api_key="ms-bfe9b66e-c071-4c29-b0b9-f5a569823064",
    base_url="https://api-inference.modelscope.cn/v1/"
)

def list_models():
    try:
        print("正在获取 ModelScope 支持的模型列表...")
        response = client.models.list()
        models = [model.id for model in response.data]
        print(f"找到 {len(models)} 个模型:")
        for m in sorted(models):
            print(f"- {m}")
        
        # 检查是否包含常见的多模态关键词
        multimodal_keywords = ['vl', 'vision', 'audio', 'video']
        multimodal_models = [m for m in models if any(k in m.lower() for k in multimodal_keywords)]
        
        if multimodal_models:
            print("\n发现可能的多模态模型:")
            for m in multimodal_models:
                print(f"- {m}")
        else:
            print("\n未在列表中发现明显的多模态模型标识。")
            
    except Exception as e:
        print(f"获取模型列表失败: {e}")

if __name__ == "__main__":
    list_models()
