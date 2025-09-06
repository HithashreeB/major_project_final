from ultralytics import YOLO

# load the best checkpoint as starting point
model = YOLO(r"C:\Users\admin\Desktop\yolo\runs\train\swin_yolov8_ft\weights\best.pt")

# start a NEW fine-tuning run
model.train(
    data="./Monkey_Bison_Detection-13/data.yaml",
    epochs=30,          # fine-tune more
    imgsz=768,
    batch=4,
    device=0,
    workers=0,
    lr0=0.001,          # smaller LR for refinement
    project="runs/train",
    name="swin_yolov8_ft1",
    exist_ok=True
)
