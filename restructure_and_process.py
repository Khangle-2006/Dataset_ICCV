# restructure_and_process.py
import os
import shutil
import json
import cv2
from tqdm import tqdm

BASE_DIR = r"G:\DEEP LEARNING\AI city\Dataset\data" 

CATEGORY_MAP = {"0": -1, "1": 3, "2": -1, "3": -1, "4": 2, "5": 2, "6": 4, "7": -1, "8": -1, "9": 0, "10": 1, "11": -1}
CATEGORIES = [{"id": i, "name": name} for i, name in enumerate(["Bus", "Bike", "Car", "Pedestrian", "Truck"])]

def flatten_folder(path):
    while len(os.listdir(path)) == 1 and os.path.isdir(os.path.join(path, os.listdir(path)[0])):
        path = os.path.join(path, os.listdir(path)[0])
    return path

def restructure():
    print("\n📁 Restructuring folders...")

  
    vis_src = os.path.join(BASE_DIR, "visdrone_raw", "VisDrone2019-DET-train")
    vis_dst = os.path.join(BASE_DIR, "Visdrone")

    if os.path.exists(vis_dst):
        print("⚠️  Visdrone folder already exists. Removing it.")
        shutil.rmtree(vis_dst)

    os.makedirs(vis_dst, exist_ok=True)
    shutil.move(os.path.join(vis_src, "images"), os.path.join(vis_dst, "images"))
    shutil.move(os.path.join(vis_src, "annotations"), os.path.join(vis_dst, "annotations"))
    print("✅ Moved Visdrone into one folder")

    # --- LOAF ---
    loaf_img = flatten_folder(os.path.join(BASE_DIR, "loaf_images"))
    loaf_ann = flatten_folder(os.path.join(BASE_DIR, "loaf_annotations"))
    loaf_dst = os.path.join(BASE_DIR, "LOAF")
    if not os.path.exists(loaf_dst):
        os.makedirs(loaf_dst)
        shutil.copytree(loaf_img, loaf_dst, dirs_exist_ok=True)
        shutil.copytree(loaf_ann, loaf_dst, dirs_exist_ok=True)
        print("✅ Merged LOAF")

    # --- Fisheye8K ---
    fs_dir = os.path.join(BASE_DIR, "fisheye8k")
    fs_dir = flatten_folder(fs_dir)
    fs_dst = os.path.join(BASE_DIR, "Fisheye8K")
    if not os.path.exists(fs_dst):
        shutil.move(fs_dir, fs_dst)
        print("✅ Moved Fisheye8K")


def convert_visdrone_to_yolo(src, dst, img_w, img_h):
    with open(src, "r") as f: bboxes = f.readlines()
    with open(dst, "w") as out:
        for bbox in bboxes:
            args = bbox.strip().split(",")
            if len(args) < 6: continue
            cls = CATEGORY_MAP.get(args[5], -1)
            if cls == -1 or int(args[4]) == 0: continue
            x, y, w, h = map(int, args[:4])
            cx, cy = (x + w / 2) / img_w, (y + h / 2) / img_h
            nw, nh = w / img_w, h / img_h
            out.write(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

def create_coco_json(img_dir, label_dir, output_path):
    coco = {"images": [], "annotations": [], "categories": CATEGORIES}
    img_id = ann_id = 0
    for img_name in tqdm(os.listdir(img_dir), desc="🔄 Converting to COCO"):
        if not img_name.lower().endswith((".jpg", ".png")): continue
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None: continue
        h, w = img.shape[:2]
        coco["images"].append({"id": img_id, "file_name": img_name, "width": w, "height": h})
        label_file = os.path.join(label_dir, os.path.splitext(img_name)[0] + ".txt")
        if os.path.exists(label_file):
            with open(label_file, "r") as f:
                for line in f:
                    cid, xc, yc, bw, bh = map(float, line.strip().split())
                    x, y = xc * w - bw * w / 2, yc * h - bh * h / 2
                    coco["annotations"].append({
                        "id": ann_id, "image_id": img_id, "category_id": int(cid),
                        "bbox": [x, y, bw * w, bh * h], "area": bw * w * bh * h,
                        "iscrowd": 0, "segmentation": []
                    })
                    ann_id += 1
        img_id += 1
    with open(output_path, "w") as f:
        json.dump(coco, f, indent=4)

def process_visdrone():
    visdrone = os.path.join(BASE_DIR, "Visdrone")
    vis_images = os.path.join(visdrone, "images")
    vis_annotations = os.path.join(visdrone, "annotations")
    vis_labels = os.path.join(visdrone, "labels")
    os.makedirs(vis_labels, exist_ok=True)

    print("\n🧠 Processing VisDrone...")
    for file in tqdm(os.listdir(vis_images), desc="🔁 VisDrone to YOLO"):
        img_path = os.path.join(vis_images, file)
        ann_path = os.path.join(vis_annotations, os.path.splitext(file)[0] + ".txt")
        out_path = os.path.join(vis_labels, os.path.splitext(file)[0] + ".txt")
        if os.path.exists(ann_path):
            img = cv2.imread(img_path)
            if img is not None:
                h, w = img.shape[:2]
                convert_visdrone_to_yolo(ann_path, out_path, w, h)

    create_coco_json(vis_images, vis_labels, os.path.join(visdrone, "train.json"))
    print("✅ VisDrone: train.json created.")

if __name__ == "__main__":
    restructure()
    process_visdrone()
    print("\n🎉 All folders and VisDrone JSON ready!")
