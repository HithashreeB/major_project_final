import cv2
from ultralytics import YOLO

model = YOLO(r"C:\Users\admin\Desktop\yolo\runs\kfold\fold_2_swin_yolov8_safe\weights\best.pt")

video_path = r"C:\Users\admin\Desktop\yolo\Giant Indian Bison With Big Mussel 😱😱😱 __ Asian Wild Gaur __ Gorumara National Park 🐘🐃🦬.publer.com.mp4"  
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Cannot open video")
    exit()

save_output = True
if save_output:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        r'C:\Users\admin\Desktop\yolo\output_detection.mp4', 
        fourcc,
        cap.get(cv2.CAP_PROP_FPS),  
        (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))  # match input size
    )

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video reached")
        break

    # 4️⃣ Run YOLO detection
    results = model(frame, imgsz=512)

    # 5️⃣ Annotate frame with bounding boxes
    annotated_frame = results[0].plot()

    # 6️⃣ Count monkeys and bison
    monkey_count = sum(1 for cls in results[0].boxes.cls if int(cls) == 0)
    bison_count  = sum(1 for cls in results[0].boxes.cls if int(cls) == 1)

    # 7️⃣ Display counts on frame
    cv2.putText(annotated_frame, f"Monkeys: {monkey_count}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"Bison: {bison_count}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 8️⃣ Show frame
    cv2.imshow('Monkey & Bison Detection', annotated_frame)

    # Save to video
    if save_output:
        out.write(annotated_frame)

    # Press 'q' to quit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 9️⃣ Release resources
cap.release()
if save_output:
    out.release()
cv2.destroyAllWindows()
