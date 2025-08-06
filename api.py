import asyncio
import logging
import os
import requests
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from queue import Queue
from contextlib import asynccontextmanager
from starlette.concurrency import run_in_threadpool
from embedding_handler import Dino2ExtractorV1, EmbeddingService, URLImageLoader, InternVIT600mbExtractor
from dotenv import load_dotenv

load_dotenv()
fastapi_logger = logging.getLogger("fastapi")

class EmbeddingRequest(BaseModel):
    url: str

task_queue = Queue()
results = {}
AUTH_TOKEN = os.getenv('TOKEN')
embedding_service = EmbeddingService(URLImageLoader(), Dino2ExtractorV1())

embedding_vit_600m = EmbeddingService(URLImageLoader(), InternVIT600mbExtractor())


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[lifespan] IP handler starting")
    try:
        ip = requests.get("https://api.ipify.org").text
        response = requests.post(
            "https://mb.artcracker.io/api/v1/update_embedding_api",
            json={"ip": ip},
            headers={
                "Authorization": f"Token {AUTH_TOKEN}",
                "Content-Type": "application/json",
                "User-Agent": "embedding-service/1.0"
            },
            timeout=10
        )
        if response.status_code in (200, 201):
            print(f"IP отправлен: {ip}")
        else:
            print(f"Не удалось отправить IP: {response.status_code}, {response.text}")
    except Exception as e:
        print(f"Ошибка при отправке IP: {str(e)}")
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/embedding/fast_extract")
async def extract_embedding(request: EmbeddingRequest):
    start = time.perf_counter()
    print(f'[fastapi start] {start}')
    fastapi_logger.info(f"start {time}")

    result = await run_in_threadpool(embedding_service.extract, request.url)
    embedding = result.tolist()

    # result = embedding_service.extract(request.url)
    # embedding = result.tolist()

    time_left = time.perf_counter() - start
    print(f"[fastapi end handler] {time_left}")

    return {"embedding": embedding, "url": request.url}


@app.get("/")
async def root():
    return {"message": "embedding service is up and running!"}

embedding_semaphore = asyncio.Semaphore(1)  # максимум 1 запрос к модели одновременно

# 💬 Запрос
class EmbeddingRequest(BaseModel):
    url: str

# 💬 FastAPI


@app.post("/embedding/test_extract")
async def extract_embedding(request: EmbeddingRequest):
    start = time.perf_counter()
    print(f"\n[{request.url}] 🌐 Запрос получен")

    # 🔄 1. Параллельно загружаем изображение
    try:
        image, message = embedding_vit_600m.loader.load(request.url)
        if image is None:
            raise ValueError(message)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    loaded = time.perf_counter()
    print(f"[{request.url}] ✅ Изображение загружено за {loaded - start:.2f} сек")

    try:
        async with asyncio.timeout(10):  # таймаут ожидания очереди
            queue_start = time.perf_counter()
            print(f"[{request.url}] ⏳ Ожидаем доступ к модели...")

            async with embedding_semaphore:
                waited = time.perf_counter()
                print(f"[{request.url}] 🔓 Доступ получен через {waited - queue_start:.2f} сек")

                # 💡 3. Извлекаем эмбеддинг
                result = embedding_vit_600m.extractor.extract(image)
                embedding = result.tolist()

                finished = time.perf_counter()
                print(f"[{request.url}] 🧠 Обработка завершена за {finished - waited:.2f} сек")
    except TimeoutError:
        raise HTTPException(status_code=503, detail="Модель занята. Повторите позже.")

    total = time.perf_counter()
    print(f"[{request.url}] ✅ Общая длительность: {total - start:.2f} сек")

    return {"embedding": embedding, "url": request.url}