import torch
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor
import requests
from io import BytesIO

url = 'https://storage.googleapis.com/master_base_bucket/building_571_a89cb34dbc8f4ac1bae484419d1eba9a.jpg'  # Замените на нужный URL
response = requests.get(url)
image = Image.open(BytesIO(response.content)).convert('RGB')

model = AutoModel.from_pretrained(
    'OpenGVLab/InternViT-300M-448px-V2_5',
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).cuda().eval()

image_processor = CLIPImageProcessor.from_pretrained('OpenGVLab/InternViT-300M-448px-V2_5')

pixel_values = image_processor(images=image, return_tensors='pt').pixel_values
pixel_values = pixel_values.to(torch.bfloat16).cuda()

outputs = model(pixel_values)
print(outputs)