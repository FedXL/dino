import requests

def get_external_ip():
    metadata_url = "http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip"
    headers = {"Metadata-Flavor": "Google"}
    response = requests.get(metadata_url, headers=headers)
    if response.status_code == 200:
        return response.text
    else:
        raise Exception(f"Unable to fetch external IP: {response.status_code}")


if __name__ == "__main__":
    print(get_external_ip())