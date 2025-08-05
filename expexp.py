import torch
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor

model = AutoModel.from_pretrained(
    'OpenGVLab/InternViT-300M-448px-V2_5',
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).cuda().eval()

# Отключение FlashAttention (если модель поддерживает этот параметр)
if hasattr(model.config, 'use_flash_attn'):
    model.config.use_flash_attn = False

image = Image.open('./1.png').convert('RGB')

image_processor = CLIPImageProcessor.from_pretrained('OpenGVLab/InternViT-300M-448px-V2_5')

pixel_values = image_processor(images=image, return_tensors='pt').pixel_values
pixel_values = pixel_values.to(torch.bfloat16).cuda()

outputs = model(pixel_values)
print(outputs)
embedding = outputs.pooler_output
print(embedding)
print('len embd:',len(embedding[0]))