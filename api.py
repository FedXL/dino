from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from queue import Queue
from threading import Thread
import time

from embedding_handler import Dino2ExtractorV1, EmbeddingService, ImageLoader


# Имитация вашей модели для извлечения эмбеддингов

# Модель данных для обработки
class EmbeddingRequest(BaseModel):
    url: str
    building_image_id: int = None

# Очередь задач
task_queue = Queue()
results = {}

# Глобальная модель
embedding_service = EmbeddingService(ImageLoader(), Dino2ExtractorV1())

# Инициализация FastAPI
app = FastAPI()

# Функция обработчика задач (фоновые задания)
def worker():
    while True:
        task_data = task_queue.get()  # Достаем задачу из очереди
        if task_data is None:  # Завершаем поток, если задача "пустая"
            break
        task_id, url = task_data
        try:
            # Обрабатываем картинку с помощью вашей модели
            embedding = embedding_service.extract(url)
            results[task_id] = {"status": "done", "embedding": embedding}
        except Exception as e:
            results[task_id] = {"status": "error", "detail": str(e)}
        finally:
            task_queue.task_done()

# Запускаем фоновый воркер
worker_thread = Thread(target=worker, daemon=True)
worker_thread.start()






# Эндпоинт для постановки задачи в очередь
@app.post("/embedding/extract")
async def add_task(request: EmbeddingRequest):
    # Генерируем уникальный task_id
    task_id = f"task_{int(time.time() * 1000)}"
    results[task_id] = {"status": "pending"}
    task_queue.put((task_id, request.url))  # Добавляем задачу в очередь
    return {"task_id": task_id}

@app.post("/embedding/fast_extract")
async def extract_embedding(request: EmbeddingRequest):
    result = embedding_service.extract(request.url)
    embedding = result.tolist()
    return {"embedding": embedding,"url":request.url}

# Эндпоинт для получения статуса задачи
@app.get("/embedding/status/{task_id}")
async def get_task_status(task_id: str):
    if task_id not in results:
        raise HTTPException(status_code=404, detail="Task ID not found")
    return results[task_id]

# Эндпоинт для проверки доступности системы
@app.get("/")
async def root():
    return {"message": "embedding service is up and running!"}
