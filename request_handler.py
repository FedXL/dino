import os
from typing import List, Any
import requests
from dotenv import load_dotenv
load_dotenv()
token = os.getenv("TOKEN")


headers = {
    "Authorization": f"Token {token}",
    "Content-Type": "application/json"
}

def get_building_images():
    url = "https://mb.artcracker.io/api/v1/building-images/"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Ошибка при получении списка изображений. Код ошибки: {response.status_code}")
        return None

def send_task_image_embedding(data):
    """
            data = {
            "building_image": 25,
            "title": "Some title",
            "content": "Some content",
            "embedding": [0.1, 0.2, 0.3] + [0.0] * 1532 + [0.1536]
        }


    """
    # data = {
    #     "building_image": 25,
    #     "title": "Some title",
    #     "content": "Some content",
    #     "embedding": [0.1, 0.2, 0.3] + [0.0] * 1532 + [0.1536]
    # }
    url = "https://mb.artcracker.io/api/v1/emb_handler_task"
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200 or response.status_code == 201:
        print("Данные успешно отправлены!")
        print("Ответ сервера:", response.json())
        return True
    else:
        print(f"Ошибка при отправке данных. Код ошибки: {response.status_code}")
        print("Ответ сервера:", response.text)
        return False


def send_building_image_embedding(data):
    """
        data = {
            "building_image": 25,
            "title": "Some title",
            "content": "Some content",
            "embedding": [0.1, 0.2, 0.3] + [0.0] * 1532 + [0.1536]
        }
    """

    url = "https://mb.artcracker.io/api/v1/emb_handler"
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200 or response.status_code == 201:
        print("Данные успешно отправлены!")
        print("Ответ сервера:", response.json())
        return True
    else:
        print(f"Ошибка при отправке данных. Код ошибки: {response.status_code}")
        print("Ответ сервера:", response.text)
        return False

def get_collections_names_list() -> list[Any] | bool:
    url = "https://mb.artcracker.io/api/v1/collections/"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        collections = response.json()
        print(collections)
        return [collection.get('name') for collection in collections]
    else:
        print(f"Ошибка при получении списка коллекций. Код ошибки: {response.status_code}")
        print(response.text)
        return False


def send_image_to_building_images(building_id, image_file):
    url = "https://mb.artcracker.io/api/v1/building-images/"

    data = {
        "building": building_id,
        "tags": "",
        "request_to_gpt": "",
        "comment": "waldemar-set",
        "is_active": False,
    }

    # НЕ передавайте 'Content-Type': 'application/json'
    headers = {
        "Authorization": f"Token {token}"  # если нужно
    }

    with open(image_file, "rb") as img:
        files = {
            "image": img
        }
        response = requests.post(url, headers=headers, data=data, files=files)

    print(response.status_code)
    try:
        print(response.json())
    except Exception as e:
        print("Ошибка при разборе JSON:", e)
        print("Ответ:", response.text)


def get_task_images_from_collection(collection_name='14-buildings-set'):
    url = f"https://mb.artcracker.io/api/v1/collection_tasks"

    headers = {
        "Authorization": f"Token {token}",
    }
    data = {
        'name': collection_name,
    }
    response = requests.post(url, headers=headers,data=data)
    if response.status_code == 200:
        print(f"Список изображений получен.{response.json()}")
        return response.json()
    else:
        print(f"Ошибка при получении списка изображений. Код ошибки: {response.status_code}")
        print(response.text)
        return None

def create_new_building_in_mb(name):
    url = "https://mb.artcracker.io/api/v1/buildings/"
    data = {
        "name": name,
    }
    headers = {
        "Authorization": f"Token {token}",
    }
    response = requests.post(url, headers=headers, json=data)
    print(response.status_code)
    try:
        data = response.json()
        id = data.get('id')
        return id
    except:
        print(response.text)

