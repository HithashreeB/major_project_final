import os
import cv2
import threading
import time
from ultralytics import YOLO
from tkinter import Tk, Button, filedialog, Label, messagebox
import pygame
weights_path = r"runs/detect/yolov8_swin_train1/weights/best.pt"
output_dir = r"runs/detect/predict_swin"
os.makedirs(output_dir, exist_ok=True)

model_animals = YOLO(weights_path)   

CONFIDENCE_THRESHOLD = 0.45

pygame.mixer.init()
monkey_sound = pygame.mixer.Sound(r"bell-notification-337658.mp3")
bison_sound = pygame.mixer.Sound(r"bell-notification-337658.mp3")
monkey_sound.set_volume(0.5)
bison_sound.set_volume(0.5)

last_sound_time = {"Monkey": 0, "Bison": 0}
SOUND_COOLDOWN = 5 

def trigger_sound(animal_type):
    global last_sound_time
    current_time = time.time()
    if current_time - last_sound_time[animal_type] > SOUND_COOLDOWN:
        if animal_type == "Monkey":
            monkey_sound.play()
        elif animal_type == "Bison":
            bison_sound.play()
        last_sound_time[animal_type] = current_time

root = Tk()
root.title("Monkey-Bison Detection with Sound Repellent")
root.geometry("450x280")

popup_sent = {"Monkey": False, "Bison": False}

def trigger_popup(animal_type, count):
    if not popup_sent[animal_type]:
        popup_sent[animal_type] = True
        root.after(0, lambda: messagebox.showinfo(f"{animal_type} Alert", f"{count} {animal_type}(s) detected!"))

def reset_alerts():
    popup_sent["Monkey"] = False
    popup_sent["Bison"] = False

def annotate_counts(frame, results):
    monkey_count = 0
    bison_count = 0
    for cls in results[0].boxes.cls:
        if int(cls) == 0:
            monkey_count += 1
        elif int(cls) == 1:
            bison_count += 1
    text = f"Monkeys: {monkey_count}, Bison: {bison_count}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame, monkey_count, bison_count


def run_image_inference(path):
    reset_alerts()
    img = cv2.imread(path)
    animal_results = model_animals.predict(img, conf=CONFIDENCE_THRESHOLD, verbose=False)

    for box in animal_results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        label = "Monkey" if cls_id == 0 else "Bison"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        trigger_sound(label)

    _, m_count, b_count = annotate_counts(img, animal_results)
    if m_count > 0:
        trigger_popup("Monkey", m_count)
    if b_count > 0:
        trigger_popup("Bison", b_count)

    cv2.imshow("Monkey-Bison Detection", img)
    save_path = os.path.join(output_dir, os.path.basename(path))
    cv2.imwrite(save_path, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def select_image():
    path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if path:
        threading.Thread(target=run_image_inference, args=(path,), daemon=True).start()


def run_video_inference(path_or_camera):
    reset_alerts()
    cap = cv2.VideoCapture(path_or_camera)
    if not cap.isOpened():
        print("Error: Cannot open video/camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        animal_results = model_animals.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

        for box in animal_results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label = "Monkey" if cls_id == 0 else "Bison"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            trigger_sound(label)

        _, m_count, b_count = annotate_counts(frame, animal_results)
        if m_count > 0:
            trigger_popup("Monkey", m_count)
        if b_count > 0:
            trigger_popup("Bison", b_count)

        cv2.imshow("Monkey-Bison Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def select_video():
    path = filedialog.askopenfilename(title="Select Video", filetypes=[("Video files", "*.mp4 *.avi *.mov")])
    if path:
        threading.Thread(target=run_video_inference, args=(path,), daemon=True).start()

def open_camera():
    threading.Thread(target=run_video_inference, args=(0,), daemon=True).start()


Label(root, text="Monkey-Bison Detection with Sound Repellent", font=("Arial", 14)).pack(pady=10)
Button(root, text="Select Image", width=25, command=select_image).pack(pady=5)
Button(root, text="Select Video", width=25, command=select_video).pack(pady=5)
Button(root, text="Open Camera", width=25, command=open_camera).pack(pady=5)

root.mainloop()
