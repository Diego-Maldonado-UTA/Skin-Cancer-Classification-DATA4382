import os
from kaggle.api.kaggle_api_extended import KaggleApi

DATASET = "nodoubttome/skin-cancer9-classesisic"
DOWNLOAD_PATH = "data"

def download_dataset():
    os.makedirs(DOWNLOAD_PATH, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    try:
        print("Staring download.")
        api.dataset_download_files(DATASET,path=DOWNLOAD_PATH,unzip=True)
        print("Dataset downloaded successfully.")

    except Exception as e:
        print("Failed to download dataset.")
        print(str(e))

if __name__ == "__main__":
    download_dataset()