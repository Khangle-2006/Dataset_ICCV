import os
import kagglehub
import gdown
import shutil
import zipfile

BASE_DIR = "data"

DATASETS = {
    "fisheye8k": {
        "kaggle_slug": "flap1812/fisheye8k",
        "zip_name": None,
    },
    "fisheye1keval": {
        "kaggle_slug": "duongtran1909/fisheye1keval",
        "zip_name": None,
    },
    "loaf_images": {
        "gdrive_id": "1hb4RhaWrz4n6DRbuhw6yfgb8N0mWGPq6",
        "zip_name": "loaf_images.zip"
    },
    "loaf_annotations": {
        "gdrive_id": "1oA7aGpsmDSH99VHspR7gUr5a5sn9zexc",
        "zip_name": "loaf_annotations.zip"
    },
    "visdrone_raw": {
        "gdrive_id": "1a2oHjcEcwXP8oUF95qiwrqzACb2YlUhn",
        "zip_name": "visdrone_train.zip"
    }
}

def download_and_extract(name, info):
    dest = os.path.join(BASE_DIR, name)
    os.makedirs(dest, exist_ok=True)

    try:
        if "kaggle_slug" in info:
            print(f"▶️ Downloading {name} from Kaggle...")
            dataset_path = kagglehub.dataset_download(info["kaggle_slug"])
            shutil.copytree(dataset_path, dest, dirs_exist_ok=True)
        elif "gdrive_id" in info:
            zip_path = os.path.join(BASE_DIR, info["zip_name"])
            if not os.path.exists(zip_path):
                print(f"▶️ Downloading {name} from GDrive...")
                gdown.download(id=info["gdrive_id"], output=zip_path, quiet=False, fuzzy=True)
            print(f"📦 Extracting {name}...")
            shutil.unpack_archive(zip_path, dest)
    except Exception as e:
        print(f"❌ Failed [{name}]: {e}")

if __name__ == "__main__":
    os.makedirs(BASE_DIR, exist_ok=True)
    for name, info in DATASETS.items():
        download_and_extract(name, info)
    print("\n✅ DONE: All datasets attempted.")
