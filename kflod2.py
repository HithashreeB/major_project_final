import os
import glob
import random
import shutil
import yaml
from sklearn.model_selection import KFold
from ultralytics import YOLO
from collections import defaultdict
import torch  # for clearing GPU memory

def count_instances(labels):
    """Count instances per class in YOLO label files."""
    class_counts = defaultdict(int)
    for lbl_path in labels:
        if not os.path.exists(lbl_path):
            continue
        with open(lbl_path, 'r') as f:
            for line in f:
                cls_id = int(line.split()[0])
                class_counts[cls_id] += 1
    return class_counts

def main():
    # ===============================
    # 1️⃣ Load images and labels
    # ===============================
    image_folder = "./Monkey_Bison_Detection-13/train/images"
    images = glob.glob(os.path.join(image_folder, "*.jpg"))
    labels = [img.replace("images", "labels").replace(".jpg", ".txt") for img in images]
    data_pairs = list(zip(images, labels))
    random.shuffle(data_pairs)

    # ===============================
    # 2️⃣ Compute class counts and weights
    # ===============================
    class_counts = count_instances(labels)
    print("Class counts:", dict(class_counts))

    # Compute weights inversely proportional to counts
    max_count = max(class_counts.values())
    class_weights = {cls: max_count / count for cls, count in class_counts.items()}
    print("Class weights for YOLO loss:", class_weights)

    # ===============================
    # 3️⃣ Define K-Fold
    # ===============================
    k = 3
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # ===============================
    # 4️⃣ K-Fold training loop
    # ===============================
    for fold, (train_idx, val_idx) in enumerate(kf.split(data_pairs)):
        print(f"\n==== Fold {fold+1}/{k} ====")

        train_pairs = [data_pairs[i] for i in train_idx]
        val_pairs   = [data_pairs[i] for i in val_idx]

        # Create fold directories
        fold_dir = os.path.abspath(f"./kfold2_{fold+1}")
        os.makedirs(os.path.join(fold_dir, "images/train"), exist_ok=True)
        os.makedirs(os.path.join(fold_dir, "images/val"), exist_ok=True)
        os.makedirs(os.path.join(fold_dir, "labels/train"), exist_ok=True)
        os.makedirs(os.path.join(fold_dir, "labels/val"), exist_ok=True)

        # Copy images and labels (no duplication)
        for img_path, lbl_path in train_pairs:
            shutil.copy(img_path, os.path.join(fold_dir, "images/train"))
            shutil.copy(lbl_path, os.path.join(fold_dir, "labels/train"))
        for img_path, lbl_path in val_pairs:
            shutil.copy(img_path, os.path.join(fold_dir, "images/val"))
            shutil.copy(lbl_path, os.path.join(fold_dir, "labels/val"))

        # Create train.txt and val.txt
        train_txt = os.path.join(fold_dir, "train.txt")
        val_txt   = os.path.join(fold_dir, "val.txt")
        with open(train_txt, "w") as f:
            for img_path, _ in train_pairs:
                f.write(f"{os.path.abspath(img_path)}\n")
        with open(val_txt, "w") as f:
            for img_path, _ in val_pairs:
                f.write(f"{os.path.abspath(img_path)}\n")

        # Create data.yaml with class weights
        fold_yaml = {
            "train": train_txt,
            "val": val_txt,
            "nc": 2,
            "names": ["monkey", "bison"],
            "weights": [class_weights.get(0, 1.0), class_weights.get(1, 1.0)]
        }
        yaml_path = os.path.join(fold_dir, "data.yaml")
        with open(yaml_path, "w") as f:
            yaml.dump(fold_yaml, f)

        # ===============================
        # 5️⃣ Initialize and train model
        # ===============================
        model = YOLO(r"C:\Users\admin\Desktop\yolo\runs\train\swin_yolov8_ft1\weights\best.pt")
        model.train(
            data=yaml_path,
            epochs=50,
            patience=10,
            imgsz=512,
            batch=2,           # safe batch size
            device=0,
            workers=2,
            lr0=0.001,
            project=os.path.join("runs", "kfold"),
            name=f"fold_{fold+1}_swin_yolov8_safe",
            exist_ok=True,
            save_period=5,
            resume=False,
            augment=True,      # YOLO applies safe augmentations internally
            mosaic=0.5,
            mixup=0.1,
            flipud=0.3,
            fliplr=0.5,
            degrees=10,
            translate=0.1,
            scale=0.5,
            shear=2.0,
            perspective=0.0005,
            erasing=0.4
        )

        # === Clear GPU memory safely after each fold ===
        del model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
