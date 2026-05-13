from openai import OpenAI

from utils.embeddings_utils import image_to_base64

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:6006/v1",
    timeout=3600
)

# Task-specific base prompts
TASKS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
}

img_base64 = image_to_base64(r'E:\my_project\Multimodal_RAG\glm_ocr\test1.png')[0]

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": img_base64
                }
            },
            {
                "type": "text",
                "text": TASKS["ocr"]
            }
        ]
    }
]

response = client.chat.completions.create(
    model="paddleocr-vl",
    messages=messages,
    temperature=0.0,
)
print(f"Generated text: {response.choices[0].message.content}")