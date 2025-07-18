# 🚦 Multi-Dataset Preparation Pipeline

This repository provides a complete pipeline to **download, process, and organize** multiple datasets including **LOAF**, **VisDrone**, **Fisheye8K** and **Fisheye1keval** for AI city Track 4 2025.

---
## 📁 Project Structure

```bash
├── Download_dataset.py         # Step 1: Download and extract datasets
├── restruct_and_process.py     # Step 2: Restructure folders & convert VisDrone annotations
├── split_dataset.py            # Step 3: Organize datasets into train/test folders
├── data/                       # Default folder where all datasets will be placed
│   ├── visdrone_raw/           # Raw VisDrone after extraction
│   ├── fisheye8k/              # Raw Fisheye8K dataset
│   ├── fisheye1keval/          # Raw Fisheye1k Eval dataset
│   ├── loaf_images/            # Raw LOAF images
│   ├── loaf_annotations/       # Raw LOAF annotations
│   ├── Visdrone/               # Final Visdrone after restructure
│   ├── LOAF/                   # Final LOAF after merge
│   ├── Fisheye8K/              # Final Fisheye8K after restructure
│   ├── Train/                  # Final training datasets (Visdrone, LOAF, Fisheye8K)
│   └── Test/                   # Final testing dataset (Fisheye1keval)

```
---

This pipeline not only downloads and restructures datasets, but also converts all annotations into standardized formats.

VisDrone annotations are converted from their raw .txt format into both YOLO (for object detection) and COCO (for evaluation). The result is saved as labels/ (YOLO) and train.json (COCO) under the Visdrone/ folder.

LOAF annotations originally use rotated bounding boxes. These are converted into standard axis-aligned bounding boxes with an additional radius_point field (used for downstream LOAF tasks). The final converted file is saved as instances_train_converted.json under the LOAF/ folder.
