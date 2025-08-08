import asyncio
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from embedding_handler import  EmbeddingService, URLImageLoader, InternVITThreeLevelExtractor
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi import status


embedding_vit_600m = EmbeddingService(URLImageLoader(), InternVITThreeLevelExtractor())
app_exp = FastAPI()
embedding_semaphore = asyncio.Semaphore(1)


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
    task_or_image : str
    params: ParamsExp



@app_exp.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Формируем удобный ответ
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "message": "Неверный формат данных в запросе",
            "errors": exc.errors(),
            "body": exc.body
        },
    )

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
                                                              focus_percentage=request.params.focus_percentage,
                                                              grid_size=request.params.grid_size,
                                                              global_weight=request.params.global_weight,
                                                              focused_weight=request.params.focused_weight,
                                                              tile_weight=request.params.tile_weight
                                                              )
                embedding = result.tolist()
                finished = time.perf_counter()
                print(f"[{request.url}] 🧠 Обработка завершена за {finished - waited:.2f} сек")
    except TimeoutError:
        raise HTTPException(status_code=503, detail="Модель занята. Повторите позже.")
    total = time.perf_counter()
    print(f"[{request.url}] ✅ Общая длительность: {total - start:.2f} сек")
    result = request.dict(exclude_unset=False)
    result['embedding'] = embedding
    return result