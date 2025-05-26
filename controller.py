import json
import os
from pathlib import Path
from embedding_handler import EmbeddingService, URLImageLoader, Dino2ExtractorV1
from request_handler import get_building_images, send_building_image_embedding, send_image_to_building_images, \
    get_task_images_from_collection, send_task_image_embedding, get_collections_names_list
from request_handler import create_new_building_in_mb

def send_new_building_images_to_mb():
    """send new building images to mb"""
    folder = "building-update"

    # Get all subdirectories (building IDs)
    building_ids = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]

    result = {}
    # Process each building directory
    for building_id in building_ids:
        building_path = os.path.join(folder, building_id)
        images = [os.path.join(building_path, f) for f in os.listdir(building_path)
                  if os.path.isfile(os.path.join(building_path, f))]
        result[building_id] = images
    print(result)
    for building_id, images in result.items():
        for image in images:
            image_file = Path(image)
            send_image_to_building_images(building_id, image_file)
    return result

def send_new_building_images_to_mb2():
    """create new building"""
    folder = "building-update-2"

    # Get all subdirectories (building IDs)
    names = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
    print(names)
    new_building_ids = {}

    for name in names:
        new_name = name[3:]
        building_id = create_new_building_in_mb(new_name)
        new_building_ids[building_id] = name
    print(new_building_ids)

    result = {}
    for building_id, building_name in new_building_ids.items():
        building_path = os.path.join(folder, building_name)
        images = [os.path.join(building_path, f) for f in os.listdir(building_path)
                  if os.path.isfile(os.path.join(building_path, f))]
        result[building_id] = images
    print(result)
    for building_id, images in result.items():
        for image in images:
            image_file = Path(image)
            send_image_to_building_images(building_id, image_file)
    return result

def extract_embedding_from_building_images_flow():
    """extract embeddig flow"""
    image_data_list = get_building_images()
    result = [{'url':image.get('image', None),'id':image.get('id', None)} for image in image_data_list]
    service = EmbeddingService(URLImageLoader(), Dino2ExtractorV1())
    for image in result:
        url=image.get('url')
        if url:
            embedding = service.extract(url)
            data = {
                "building_image": image.get('id'),
                "title": "building embd",
                "content": "0.5 normalization 218 image size",
                "embedding": embedding.tolist()
            }
            print(data)
            result = send_building_image_embedding(data)
            filename = f'{image.get("id")}.json'

            # Убедимся, что папка существует
            os.makedirs('results', exist_ok=True)
            # Сохраняем как JSON
            with open(f'results/{filename}', 'w', encoding='utf-8') as f:
                new_data = {
                    "ok": result,
                    "id": image.get('id')
                }
                json.dump(new_data, f, ensure_ascii=False, indent=2)

def extract_embedding_from_task_collection(collection_name='14-buildings-set'):

    if collection_name == "" or not collection_name:
        collection_name = '14-buildings-set'
        print(f"Введено: {collection_name}")
    else:
        print(f"Введено: {collection_name}")
    tasks = get_task_images_from_collection(collection_name=collection_name)
    result = [{"id": task.get('id'), "url": task.get('image_url')} for task in tasks]
    total = len(result)
    service = EmbeddingService(URLImageLoader(), Dino2ExtractorV1())
    os.makedirs('results2', exist_ok=True)

    for index, task in enumerate(result, start=1):
        embedding = service.extract(task.get('url'))
        data = {
            "task": task.get('id'),
            "title": "task embd",
            "content": "0.5 normalization 518 image size",
            "embedding": embedding.tolist()
        }
        print(index," / ",len(result),' Длинна',len(embedding.tolist()))

        result_api = send_task_image_embedding(data)
        filename = f'{task.get("id")}.json'
        json_path = f'results2/{filename}'

        new_data = {
            "ok": result_api,
            "id": task.get('id')
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, ensure_ascii=False, indent=2)

        print(f"[{index}/{total}] Обработан task ID {task.get('id')}, сохранено в {json_path}")


def main_flow():
    choice = input("Выберите действие: 1. Обработать изображения из коллекции, 2."
                   " Обработать изображения из зданий: "
                   "3. Обработать всё")
    match choice:
        case '1':
            name = input("Введите название коллекции / или ткните enter (default 14-buildings-set): ")
            print(name)
            extract_embedding_from_task_collection(name)
        case '2':
            extract_embedding_from_building_images_flow()

        case '3':
            collections=get_collections_names_list()
            for collection in collections:
                print(f'START COLLECTION:{collection}')
                extract_embedding_from_task_collection(collection)
            extract_embedding_from_building_images_flow()

def main_flow2():
    collections = get_collections_names_list()
    for collection in collections:
        print(f'START COLLECTION:{collection}')
        extract_embedding_from_task_collection(collection)
    extract_embedding_from_building_images_flow()



