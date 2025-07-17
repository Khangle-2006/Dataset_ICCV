import os
import shutil

BASE_DIR = r"G:\DEEP LEARNING\AI city\Dataset\data"

TRAIN_FOLDERS = ["Fisheye8K", "LOAF", "Visdrone"]
TEST_FOLDERS = ["Fisheye1keval"]

def move_to_subfolder(subfolder_name, folder_names):
    subfolder_path = os.path.join(BASE_DIR, subfolder_name)
    os.makedirs(subfolder_path, exist_ok=True)

    for folder in folder_names:
        src = os.path.join(BASE_DIR, folder)
        dst = os.path.join(subfolder_path, folder)
        if os.path.exists(src):
            print(f"📁 Moving {folder} to {subfolder_name}...")
            shutil.move(src, dst)
            print(f"✅ {folder} moved to {subfolder_name}.")
        else:
            print(f"❌ Folder not found: {folder}")

if __name__ == "__main__":
    print("\n🚚 Organizing dataset folders...")
    move_to_subfolder("Train", TRAIN_FOLDERS)
    move_to_subfolder("Test", TEST_FOLDERS)
    print("\n🎉 Dataset organization complete!")