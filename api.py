import logging
import requests
import time
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from queue import Queue
from threading import Thread
from contextlib import asynccontextmanager

from embedding_handler import Dino2ExtractorV1, EmbeddingService, URLImageLoader

# Логгер FastAPI
fastapi_logger = logging.getLogger("fastapi")


# Модель запроса
class EmbeddingRequest(BaseModel):
    url: str
    building_image_id: int = None


# Очередь задач и результаты
task_queue = Queue()
results = {}
AUTH_TOKEN = "dee4bbc55782819eb8047daf17242c1532d7a6d4"
# Глобальная модель
embedding_service = EmbeddingService(URLImageLoader(), Dino2ExtractorV1())


# Lifespan обработчик



@asynccontextmanager
async def lifespan(app: FastAPI):
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
            timeout=5
        )
        if response.status_code in (200, 201):
            fastapi_logger.info(f"IP отправлен: {ip}")
        else:
            fastapi_logger.error(f"Не удалось отправить IP: {response.status_code}, {response.text}")
    except Exception as e:
        fastapi_logger.error(f"Ошибка при отправке IP: {str(e)}")
    yield


# Инициализация FastAPI с lifespan
app = FastAPI(lifespan=lifespan)


@app.post("/embedding/fast_extract")
async def extract_embedding(request: EmbeddingRequest):
    start = time.perf_counter()
    print(f'[fastapi start] {start}')
    fastapi_logger.info(f"start {time}")

    result = embedding_service.extract(request.url)
    embedding = result.tolist()

    time_left = time.perf_counter() - start
    print(f"[fastapi end handler] {time_left}")

    return {"embedding": embedding, "url": request.url}


@app.get("/")
async def root():
    return {"message": "embedding service is up and running!"}