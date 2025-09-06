import os
import glob
import random
import shutil
import yaml
from sklearn.model_selection import KFold
from ultralytics import YOLO

def main():
    # ===============================
    # 1️⃣ Load all images and labels
    # ===============================
    image_folder = "./Monkey_Bison_Detection-13/train/images"
    images = glob.glob(os.path.join(image_folder, "*.jpg"))
    labels = [img.replace("images", "labels").replace(".jpg", ".txt") for img in images]

    data_pairs = list(zip(images, labels))
    random.shuffle(data_pairs)

    # ===============================
    # 2️⃣ Define K-Fold
    # ===============================
    k = 3  # number of folds
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # ===============================
    # 3️⃣ K-Fold training loop
    # ===============================
    for fold, (train_idx, val_idx) in enumerate(kf.split(data_pairs)):
        print(f"\n==== Fold {fold+1} ====")

        train_pairs = [data_pairs[i] for i in train_idx]
        val_pairs   = [data_pairs[i] for i in val_idx]

        # Create fold directories
        fold_dir = os.path.abspath(f"./fold_{fold+1}")
        os.makedirs(os.path.join(fold_dir, "images/train"), exist_ok=True)
        os.makedirs(os.path.join(fold_dir, "images/val"), exist_ok=True)
        os.makedirs(os.path.join(fold_dir, "labels/train"), exist_ok=True)
        os.makedirs(os.path.join(fold_dir, "labels/val"), exist_ok=True)

        # Copy images and labels
        for img_path, lbl_path in train_pairs:
            shutil.copy(img_path, os.path.join(fold_dir, "images/train"))
            shutil.copy(lbl_path, os.path.join(fold_dir, "labels/train"))
        for img_path, lbl_path in val_pairs:
            shutil.copy(img_path, os.path.join(fold_dir, "images/val"))
            shutil.copy(lbl_path, os.path.join(fold_dir, "labels/val"))

        # Create data.yaml
        fold_yaml = {
            "train": os.path.join(fold_dir, "images/train"),
            "val": os.path.join(fold_dir, "images/val"),
            "nc": 2,
            "names": ["monkey", "bison"]
        }
        yaml_path = os.path.join(fold_dir, "data.yaml")
        with open(yaml_path, "w") as f:
            yaml.dump(fold_yaml, f)

        # ===============================
        # 4️⃣ Initialize and train model
        # ===============================
        model = YOLO(r"C:\Users\admin\Desktop\yolo\runs\train\swin_yolov8_ft\weights\best.pt")

        model.train(
            data=yaml_path,
            epochs=30,
            imgsz=768,
            batch=2,  # reduce if GPU memory is limited
            device=0,
            workers=0,  # safest on Windows
            lr0=0.001,
            project=os.path.join("runs", "kfold"),
            name=f"fold_{fold+1}_swin_yolov8",
            exist_ok=True
        )

# 🔹 Required for Windows multiprocessing
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # optional but safe
    main()
