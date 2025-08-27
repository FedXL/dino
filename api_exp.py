import asyncio
import gc
import os
import time
import torch

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from enum import Enum

from embedding_handler import (
    Dino2ExtractorV1,
    Dino3ExtractorV1,
    Dino3ExtractorV1pipeline,
    EmbeddingService,
    InternVIT600mbExtractor,
    # InternVITSimpleExtractor,
    # InternVITThreeLevelExtractor,
    URLImageLoader,
)

# ===== CONFIGURATION =====

# Available model classes enum for FastAPI dropdown
class AvailableModels(str, Enum):
    DINO2_V1 = "Dino2ExtractorV1"
    DINO3_V1 = "Dino3ExtractorV1" 
    DINO3_PIPELINE = "Dino3ExtractorV1pipeline"
    INTERNVIT_600MB = "InternVIT600mbExtractor"
    # INTERNVIT_THREE_LEVEL = "InternVITThreeLevelExtractor"
    # INTERNVIT_SIMPLE = "InternVITSimpleExtractor"

# Available model classes mapping
AVAILABLE_MODELS = {
    "Dino2ExtractorV1": Dino2ExtractorV1,
    "Dino3ExtractorV1": Dino3ExtractorV1,
    "Dino3ExtractorV1pipeline": Dino3ExtractorV1pipeline,
    "InternVIT600mbExtractor": InternVIT600mbExtractor,
}

# ===== GLOBAL STATE =====

# Global embedding service - will be dynamically switched
current_model_class = AvailableModels.DINO3_V1
current_embedding_service = EmbeddingService(URLImageLoader(), AVAILABLE_MODELS[current_model_class.value]())
embedding_semaphore = asyncio.Semaphore(1)

# ===== PYDANTIC MODELS =====

class SimpleEmbeddingRequest(BaseModel):
    url: str

class ModelSwitchRequest(BaseModel):
    model_class: AvailableModels = Field(..., description="Select the model to switch to")

class ParamsExp(BaseModel):
    focus_percentage: int
    grid_size: int
    global_weight: float
    focused_weight: float
    tile_weight: float

class InternVITExperiment(BaseModel):
    url: str
    title: str
    id: int
    task_or_image: str
    params: ParamsExp

# ===== FASTAPI APP =====

app_exp = FastAPI()

# ===== MIDDLEWARE & EXCEPTION HANDLERS =====

@app_exp.middleware("http")
async def add_process_id_header(request: Request, call_next):
    """Add process ID to response headers for debugging"""
    pid = os.getpid()
    response = await call_next(request)
    response.headers["X-Process-ID"] = str(pid)
    print(f"Request handled by PID: {pid}")
    return response

@app_exp.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors with detailed response"""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "message": "Неверный формат данных в запросе",
            "errors": exc.errors(),
            "body": exc.body
        },
    )

# ===== API ENDPOINTS =====

@app_exp.post("/")
async def hello(request):
    """Health check endpoint"""
    return {'hello': 'im here'}


# ----- EMBEDDING EXTRACTION -----

@app_exp.post("/embedding/extract")
async def extract_embedding_dynamic(request: SimpleEmbeddingRequest):
    """Extract embeddings using the currently loaded model"""
    start = time.perf_counter()
    print(f"\n[{request.url}] 🌐 Запрос получен для модели {current_model_class}")

    # Load image
    try:
        image, message = current_embedding_service.loader.load(request.url)
        if image is None:
            raise ValueError(message)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    loaded = time.perf_counter()
    print(f"[{request.url}] ✅ Изображение загружено за {loaded - start:.2f} сек")

    try:
        async with asyncio.timeout(10):
            queue_start = time.perf_counter()
            print(f"[{request.url}] ⏳ Ожидаем доступ к модели ({current_model_class})...")

            async with embedding_semaphore:
                waited = time.perf_counter()
                print(f"[{request.url}] 🔓 Доступ получен через {waited - queue_start:.2f} сек")

                result = current_embedding_service.extractor.extract(image)
                embedding = result.tolist()

                finished = time.perf_counter()
                print(f"[{request.url}] 🧠 Обработка завершена ({current_model_class}) за {finished - waited:.2f} сек")
    except TimeoutError:
        raise HTTPException(status_code=503, detail="Модель занята. Повторите позже.")

    total = time.perf_counter()
    print(f"[{request.url}] ✅ Общая длительность: {total - start:.2f} сек")

    return {"embedding": embedding, "url": request.url, "model_used": current_model_class}


# ===== UTILITY FUNCTIONS =====

def cleanup_gpu_memory():
    """Clean up GPU memory with garbage collection only - avoid clearing shared GPU cache"""
    # Only run garbage collection to clean up Python references
    # Avoid torch.cuda.empty_cache() to not interfere with other services
    gc.collect()
    if torch.cuda.is_available():
        # Just synchronize to ensure current operations complete
        torch.cuda.synchronize()
    print("🧹 Memory references cleaned (GPU cache preserved for other services)")


# ----- MODEL MANAGEMENT -----

@app_exp.post("/model/switch")
async def switch_model(request: ModelSwitchRequest):
    """Switch between different embedding models"""
    global current_embedding_service, current_model_class
    
    print(f"\n🔄 Model switch request: {request.model_class}")
    print(f"📊 Current model: {current_model_class}")
    print(f"📊 Current service extractor type: {type(current_embedding_service.extractor).__name__}")
    
    # Check if model class is available
    if request.model_class.value not in AVAILABLE_MODELS:
        available_models = [model.value for model in AvailableModels]
        raise HTTPException(
            status_code=400, 
            detail=f"Model class '{request.model_class}' not available. Available models: {available_models}"
        )
    
    # Check if already using the requested model
    if request.model_class == current_model_class:
        print(f"✅ Already using {request.model_class}")
        return {
            "status": "no_change",
            "message": f"Already using {request.model_class}",
            "current_model": current_model_class,
            "actual_extractor": type(current_embedding_service.extractor).__name__
        }
    
    # Acquire semaphore to prevent extraction during model switch
    async with embedding_semaphore:
        # Store the previous model state for potential rollback
        previous_model_class = current_model_class
        previous_extractor_class = AVAILABLE_MODELS[current_model_class.value]
        
        try:
            start_time = time.perf_counter()
            print(f"🔓 Acquired semaphore for model switch")
            
            # Load new model FIRST before unloading the old one
            print(f"🚀 Loading new model: {request.model_class}")
            model_class = AVAILABLE_MODELS[request.model_class.value]
            new_extractor = model_class()
            new_service = EmbeddingService(URLImageLoader(), new_extractor)
            
            # Only after successful new model loading, clean up the old model
            print(f"🗑️ Unloading previous model: {current_model_class}")
            old_model = current_embedding_service.extractor
            
            # Move old model to CPU and delete references
            if hasattr(old_model, 'model') and hasattr(old_model.model, 'cpu'):
                old_model.model.cpu()
            
            del current_embedding_service.extractor
            del current_embedding_service
            cleanup_gpu_memory()
            
            print(f"✅ {previous_model_class} unloaded")
            
            # Switch to the new model
            current_embedding_service = new_service
            current_model_class = request.model_class
            
            elapsed_time = time.perf_counter() - start_time
            print(f"✅ Model switch completed in {elapsed_time:.2f} seconds")
            print(f"🎯 New active model: {current_model_class}")
            
            return {
                "status": "success",
                "message": f"Successfully switched to {request.model_class}",
                "previous_model": previous_model_class,
                "current_model": current_model_class,
                "actual_extractor": type(current_embedding_service.extractor).__name__,
                "switch_time_seconds": round(elapsed_time, 2)
            }
            
        except Exception as e:
            print(f"❌ Error during model switch: {str(e)}")
            # Try to recover by restoring the previous model
            try:
                print(f"🔄 Attempting fallback to previous model: {previous_model_class}...")
                fallback_extractor = previous_extractor_class()
                current_embedding_service = EmbeddingService(URLImageLoader(), fallback_extractor)
                current_model_class = previous_model_class
                print(f"✅ Fallback to {previous_model_class} successful")
            except Exception as fallback_error:
                print(f"❌ Fallback to {previous_model_class} failed: {str(fallback_error)}")
                print("🆘 System is now in an unstable state - no valid model loaded")
            
            raise HTTPException(
                status_code=500,
                detail=f"Failed to switch model: {str(e)}"
            )


@app_exp.get("/model/status")
async def get_model_status():
    """Get current model status and system information"""
    print(f"📊 Model status request - Current: {current_model_class}")
    
    return {
        "current_model": current_model_class,
        "available_models": [model.value for model in AvailableModels],
        "gpu_available": torch.cuda.is_available(),
        "gpu_memory_allocated": torch.cuda.memory_allocated() if torch.cuda.is_available() else None,
        "gpu_memory_cached": torch.cuda.memory_reserved() if torch.cuda.is_available() else None
    }



# @app_exp.post("/embedding/test_extract")
# async def extract_embedding(request: InternVITExperiment):
#     start = time.perf_counter()
#     print(f"\n[{request.url}] 🌐 Запрос получен")
#     try:
#         image, message = embedding_vit_600m.loader.load(request.url)
#         if image is None:
#             raise ValueError(message)
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))

#     loaded = time.perf_counter()
#     print(f"[{request.url}] ✅ Изображение загружено за {loaded - start:.2f} сек")

#     try:
#         async with asyncio.timeout(10):
#             queue_start = time.perf_counter()
#             print(f"[{request.url}] ⏳ Ожидаем доступ к модели...")
#             async with embedding_semaphore:
#                 waited = time.perf_counter()
#                 print(f"[{request.url}] 🔓 Доступ получен через {waited - queue_start:.2f} сек")
#                 result = embedding_vit_600m.extractor.extract(image,
#                                                               focus_percentage=request.params.focus_percentage,
#                                                               grid_size=request.params.grid_size,
#                                                               global_weight=request.params.global_weight,
#                                                               focused_weight=request.params.focused_weight,
#                                                               tile_weight=request.params.tile_weight
#                                                               )
#                 embedding = result.tolist()
#                 finished = time.perf_counter()
#                 print(f"[{request.url}] 🧠 Обработка завершена за {finished - waited:.2f} сек")
#     except TimeoutError:
#         raise HTTPException(status_code=503, detail="Модель занята. Повторите позже.")
#     total = time.perf_counter()
#     print(f"[{request.url}] ✅ Общая длительность: {total - start:.2f} сек")
#     result = request.dict(exclude_unset=False)
#     result['embedding'] = embedding
#     return result

