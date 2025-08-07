import asyncio
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from embedding_handler import  EmbeddingService, URLImageLoader, InternVITThreeLevelExtractor

embedding_vit_600m = EmbeddingService(URLImageLoader(), InternVITThreeLevelExtractor())
app_exp = FastAPI()
embedding_semaphore = asyncio.Semaphore(1)

# 💬 Запрос
class EmbeddingRequest(BaseModel):
    url: str

class InternVITExperiment(BaseModel):
    url: str
    focus_percentage : int
    grid_size: int
    global_weight: float
    focused_weight: float
    tile_weight: float


@app_exp.post("/")
async def hello(request):
    return {'hello': 'im here'}


@app_exp.post("/embedding/test_extract")
async def extract_embedding(request: InternVITExperiment):
    start = time.perf_counter()
    print(f"\n[{request.url}] 🌐 Запрос получен")
    try:
        image, message = embedding_vit_600m.loader.load(request.url)
        if image is None:
            raise ValueError(message)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    loaded = time.perf_counter()
    print(f"[{request.url}] ✅ Изображение загружено за {loaded - start:.2f} сек")

    try:
        async with asyncio.timeout(10):
            queue_start = time.perf_counter()
            print(f"[{request.url}] ⏳ Ожидаем доступ к модели...")
            async with embedding_semaphore:
                waited = time.perf_counter()
                print(f"[{request.url}] 🔓 Доступ получен через {waited - queue_start:.2f} сек")
                result = embedding_vit_600m.extractor.extract(image,
                                                              focus_percentage=request.focus_percentage,
                                                              grid_size=request.grid_size,
                                                              global_weight=request.global_weight,
                                                              focused_weight=request.focused_weight,
                                                              tile_weight=request.tile_weight
                                                              )
                embedding = result.tolist()
                finished = time.perf_counter()
                print(f"[{request.url}] 🧠 Обработка завершена за {finished - waited:.2f} сек")
    except TimeoutError:
        raise HTTPException(status_code=503, detail="Модель занята. Повторите позже.")
    total = time.perf_counter()
    print(f"[{request.url}] ✅ Общая длительность: {total - start:.2f} сек")
    return {"embedding": embedding, "url": request.url}