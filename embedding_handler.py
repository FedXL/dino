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
    def extract(self, image: Image.Image,**kwargs) -> np.ndarray:
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


class InternVITThreeLevelExtractor(EmbeddingExtractor):
    def __init__(self,
                 model_id="OpenGVLab/InternViT-300M-448px-V2_5",
                 input_size=448,
                 device=DEVICE):
        """
        Three-level feature extractor with global, focused, and tile-based features

        Args:
            model_id: HuggingFace model identifier
            input_size: Model's expected input size (e.g., 448 for InternViT)
            focus_percentage: Percentage of center square to crop for focused features (e.g., 70)
            grid_size: Grid size for tiling (e.g., 3 for 3x3 grid)
            global_weight: Weight for global context features
            focused_weight: Weight for focused detail features
            tile_weight: Weight for tile-based features
            device: Computing device
        """

        self.device = device
        self.input_size = input_size


        # Validate weights sum to 1.0


        print(f"[Загрузка InternViT трёхуровневой модели...]")
        start = time.perf_counter()

        # Load model and processor
        self.model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to(device).eval()

        self.image_processor = CLIPImageProcessor.from_pretrained(model_id)

        print(f"[Модель загружена за {time.perf_counter() - start:.2f} сек]")


    def _crop_to_center_square(self, image):
        """Crop rectangle image to centered square"""
        width, height = image.size
        size = min(width, height)

        left = (width - size) // 2
        top = (height - size) // 2
        right = left + size
        bottom = top + size

        return image.crop((left, top, right, bottom))

    def _crop_center_percentage(self, image, percentage):
        """Crop center portion of image by percentage"""
        width, height = image.size
        new_size = int(min(width, height) * percentage)

        left = (width - new_size) // 2
        top = (height - new_size) // 2
        right = left + new_size
        bottom = top + new_size

        return image.crop((left, top, right, bottom))

    def _split_into_grid(self, image, grid_size):
        """Split image into grid_size x grid_size tiles"""
        width, height = image.size
        tile_width = width // grid_size
        tile_height = height // grid_size

        tiles = []
        for row in range(grid_size):
            for col in range(grid_size):
                left = col * tile_width
                top = row * tile_height
                right = left + tile_width
                bottom = top + tile_height

                tile = image.crop((left, top, right, bottom))
                tiles.append(tile)

        return tiles

    def _extract_single_features(self, pil_image: Image.Image) -> torch.Tensor:
        with torch.no_grad():
            pixel_values = self.image_processor(images=pil_image, return_tensors='pt').pixel_values
            pixel_values = pixel_values.to(self.device, dtype=torch.bfloat16)

            if hasattr(self.model, "get_image_features"):
                feats = self.model.get_image_features(pixel_values=pixel_values)
            else:
                outputs = self.model(pixel_values)
                if hasattr(outputs, "image_embeds"):
                    feats = outputs.image_embeds
                elif hasattr(outputs, "last_hidden_state"):
                    # mean-pool patch tokens
                    feats = outputs.last_hidden_state.mean(dim=1)
                else:
                    feats = outputs.pooler_output  # fallback if nothing else

            feats = torch.nn.functional.normalize(feats, dim=-1)
            return feats.squeeze(0).to(torch.float32).cpu()

    def extract(self, pil_image: Image.Image,
                focus_percentage=50,
                 grid_size=3,
                 global_weight=0.2,
                 focused_weight=0.4,
                 tile_weight=0.4) -> np.ndarray:
        """
        Extract three-level features: global, focused, and tile-based

        Args:
            pil_image: Input PIL image

        Returns:
            Combined feature embedding as numpy array
            :param pil_image:
            :param tile_weight:
            :param focused_weight:
            :param global_weight:
            :param grid_size:
            :param focus_percentage:
        """
        self.focus_percentage = focus_percentage / 100.0
        total_weight = global_weight + focused_weight + tile_weight
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
        print(f"[Параметры: размер={self.input_size}, фокус={focus_percentage}%, сетка={grid_size}x{grid_size}]")

        # Convert to fraction
        self.grid_size = grid_size
        self.global_weight = global_weight
        self.focused_weight = focused_weight
        self.tile_weight = tile_weight
        print("[Начало трёхуровневого извлечения эмбеддинга]")
        start = time.perf_counter()

        # Step 1: Convert rectangle to centered square
        square_image = self._crop_to_center_square(pil_image)
        print(f"[Обрезка до квадрата: {pil_image.size} → {square_image.size}]")

        # Level 1: Global context (full square)
        global_start = time.perf_counter()
        global_emb = self._extract_single_features(square_image)
        global_time = time.perf_counter() - global_start
        print(f"[Глобальный контекст извлечён за {global_time:.2f} сек]")

        # Level 2: Focused detail (center crop)
        focused_start = time.perf_counter()
        focused_square = self._crop_center_percentage(square_image, self.focus_percentage)
        focused_emb = self._extract_single_features(focused_square)
        focused_time = time.perf_counter() - focused_start
        print(f"[Фокусированные детали извлечены за {focused_time:.2f} сек, размер {focused_square.size}]")

        # Level 3: Tile-based features (grid from full square)
        tiles_start = time.perf_counter()
        tiles = self._split_into_grid(square_image, self.grid_size)
        tile_embs = []

        for i, tile in enumerate(tiles):
            tile_emb = self._extract_single_features(tile)
            tile_embs.append(tile_emb)

        # Aggregate tile embeddings (simple mean for now)
        tile_combined = torch.stack(tile_embs).mean(dim=0)
        tiles_time = time.perf_counter() - tiles_start
        print(f"[{len(tiles)} тайлов обработано за {tiles_time:.2f} сек]")

        # Combine all three levels
        elapsed = time.perf_counter() - start
        final_emb = (self.global_weight * global_emb +
                     self.focused_weight * focused_emb +
                     self.tile_weight * tile_combined)
        final_emb = torch.nn.functional.normalize(final_emb, dim=-1)
        result = final_emb.cpu().numpy()
        print(f"[Трёхуровневый эмбеддинг извлечён за {elapsed:.2f} сек]")
        print(f"[Веса: глобал={self.global_weight}, фокус={self.focused_weight}, тайлы={self.tile_weight}]")

        return result


class InternVITSimpleExtractor(EmbeddingExtractor):
    def __init__(self,
                 model_id="OpenGVLab/InternViT-300M-448px-V2_5",
                 processor_id="OpenGVLab/InternViT-300M-448px-V2_5",
                 device=DEVICE):
        """
        Simple InternViT feature extractor using the recommended approach
        NO cutting into tiles, just feature extraction
        Args:
            model_id: HuggingFace model identifier for the main model
            processor_id: HuggingFace model identifier for the image processor
            device: Computing device
        """
        self.device = device
        
        print(f"[Загрузка {model_id} модели...]")
        start = time.perf_counter()
        
        # Load model using the recommended approach
        self.model = AutoModel.from_pretrained(
            model_id,
            # torch_dtype=torch.bfloat32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation="eager"
        ).to(device).eval()
        
        # Load image processor
        self.image_processor = CLIPImageProcessor.from_pretrained(processor_id)
        
        print(f"[Модель загружена за {time.perf_counter() - start:.2f} сек]")
    
    def extract(self, pil_image: Image.Image) -> np.ndarray:
        """
        Extract embedding from PIL image using the simple InternViT approach
        
        Args:
            pil_image: Input PIL image
            
        Returns:
            Feature embedding as numpy array
        """
        print("[Начало извлечения эмбеддинга]")
        start = time.perf_counter()
        
        with torch.no_grad():
            # Convert image to RGB if needed
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Follow the documentation example exactly
            pixel_values = self.image_processor(images=pil_image, return_tensors='pt').pixel_values
            pixel_values = pixel_values.to(torch.bfloat32).to(self.device)
            
            outputs = self.model(pixel_values)
            
            # Extract the embedding from outputs
            embedding = outputs.pooler_output
            
            # Convert to float32 first, then to numpy array
            result = embedding.squeeze(0).cpu().to(torch.float32).numpy()
        
        elapsed = time.perf_counter() - start
        print(f"[Эмбеддинг извлечён за {elapsed:.2f} сек]")
        
        return result



class EmbeddingService:
    def __init__(self, loader: ImageLoader, extractor: EmbeddingExtractor):
        self.loader = loader
        self.extractor = extractor

    def extract(self, url: str,**kwargs) -> np.ndarray:
        image, message = self.loader.load(url)
        if image is None:
            raise ValueError(message)
        return self.extractor.extract(image,**kwargs)

