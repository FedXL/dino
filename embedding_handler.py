from abc import ABC, abstractmethod
from typing import Union, BinaryIO, Optional, Tuple
import torch
from PIL.ImageFile import ImageFile
from torchvision import transforms
from PIL import Image
import numpy as np
import time
import requests
from io import BytesIO
from transformers import AutoProcessor, AutoModel, AutoModelForVision2Seq
import torch
from PIL import Image
from transformers import CLIPImageProcessor
DEVICE = 'cuda'


class EmbeddingExtractor(ABC):
    @abstractmethod
    def extract(self, image: Image.Image) -> np.ndarray:
        ...


class ImageLoader(ABC):
    @abstractmethod
    def load(self, source: str) -> Image.Image:
        ...


class URLImageLoader(ImageLoader):
    def load(self, source: str) -> tuple[ImageFile | None, str]:
        print('[Загрузка изображения]')
        start = time.perf_counter()
        img, message = None, None
        try:
            response = requests.get(source)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert("RGB")
            message = 'Ok'
        except requests.RequestException as e:
            message = f"Ошибка при загрузке изображения: {e}"
        except Exception as e:
            message = f"Ошибка при обработке изображения: {e}"
        time_left = time.perf_counter() - start
        print(f'[Конец загрузки изображения] {time_left}')
        return img, message


class Dino2ExtractorV1(EmbeddingExtractor):
    def __init__(self, image_size=518, model_name='dinov2_vitg14', device=DEVICE):
        start = time.perf_counter()
        self.image_size = image_size
        self.device = device

        print("[Загрузка модели...]")
        self.model = torch.hub.load('facebookresearch/dinov2', model_name, trust_repo=True).to(device)
        self.model.eval()
        print(f"[Модель загружена за {time.perf_counter() - start:.2f} сек]")

        print("[Создание transform...]")
        start = time.perf_counter()
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size, interpolation=Image.BICUBIC),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
        ])
        print(f"[Transform готов за {time.perf_counter() - start:.2f} сек]")



    def extract(self, image: Image.Image) -> np.ndarray:

        print("[Начало извлечения эмбеддинга]")
        start = time.perf_counter()
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(img_tensor)
        elapsed = time.perf_counter() - start
        result = features.squeeze(0).cpu().numpy()
        print(f"[Эмбеддинг извлечён за {elapsed:.2f} сек]")
        return result


class InternVIT600mbExtractor(EmbeddingExtractor):
    def __init__(self, model_id="OpenGVLab/InternVL3-1B", image_size=448, device=DEVICE):
        self.device = device
        self.image_size = image_size

        self.model = AutoModel.from_pretrained(
            'OpenGVLab/InternViT-300M-448px-V2_5',
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).cuda().eval()
        self.image_processor = CLIPImageProcessor.from_pretrained('OpenGVLab/InternViT-300M-448px-V2_5')


    def extract(self, pil_image: Image.Image):
        pixel_values = self.image_processor(images=pil_image, return_tensors='pt').pixel_values
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        outputs = self.model(pixel_values)
        embedding = outputs.pooler_output
        return embedding.squeeze(0).cpu()




class EmbeddingService:
    def __init__(self, loader: ImageLoader, extractor: EmbeddingExtractor):
        self.loader = loader
        self.extractor = extractor

    def extract(self, url: str) -> np.ndarray:
        image, message = self.loader.load(url)
        if image is None:
            raise ValueError(message)
        return self.extractor.extract(image)