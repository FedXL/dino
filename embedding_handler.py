from abc import ABC, abstractmethod
import torch
from PIL.ImageFile import ImageFile
from torchvision import transforms
from PIL import Image
import numpy as np
import time
import requests
from io import BytesIO



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
        return img, message




class Dino2ExtractorV1(EmbeddingExtractor):
    def __init__(self, image_size=518, model_name='dinov2_vitg14', device='cuda'):
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
        print(f"Peak memory usage: {torch.cuda.max_memory_allocated(device) / 1024 ** 3:.3f} GB")


    def extract(self, image: Image.Image) -> np.ndarray:

        print("[Начало извлечения эмбеддинга]")
        start = time.perf_counter()
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(img_tensor)
        elapsed = time.perf_counter() - start
        print(f"[Эмбеддинг извлечён за {elapsed:.2f} сек]")
        return features.squeeze(0).cpu().numpy()


class EmbeddingService:
    def __init__(self, loader: ImageLoader, extractor: EmbeddingExtractor):
        self.loader = loader
        self.extractor = extractor

    def extract(self, url: str) -> np.ndarray:
        image, message = self.loader.load(url)
        if image is None:
            raise ValueError(message)
        return self.extractor.extract(image)


# if __name__ == "__main__":
#     start_time = time.time()
#     url = "https://storage.googleapis.com/artcracker-bucket/137fff8c-340e-404e-ab3d-e1a4cca5d5fe.jpg"
#     service = EmbeddingService(URLImageLoader(), Dino2ExtractorV1())
#     emb = service.extract(url)
#     print(f"Embedding shape:", emb.shape)
#     print(f"First 5 values:", emb[:5])
#     print(f"Time elapsed: {time.time() - start_time} seconds")