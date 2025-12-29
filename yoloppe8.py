import os
import time
import json
import cv2
import numpy as np
from ultralytics import YOLO
from openai import OpenAI
import textwrap
from datetime import datetime
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import winsound

# =============================
# CONFIG
# =============================
MODEL_PATH = "best.pt"
IMG_WIDTH = 900
IMG_HEIGHT = 600
TEXT_PANEL_WIDTH = 480
WINDOW_NAME = "PPE Safety Assistant"

DELAY_ORIGINAL = 1.5
DELAY_ALL_BOXES = 5.0
DELAY_VIOLATION = 1.5
DELAY_QUESTION = 1.5

last_frame = None
last_detections = None

VIOLATION_CLASSES = {
    "NO-Mask": "Wear a mask to protect your lungs.",
    "NO-Hardhat": "Wear a hardhat to protect your head.",
    "NO-Safety Vest": "Wear a safety vest for visibility."
}

# =============================
# INIT
# =============================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
yolo_model = YOLO(MODEL_PATH)

# =============================
# UTILS
# =============================
def resize_for_display(img):
    h, w = img.shape[:2]
    scale = min(IMG_WIDTH / w, IMG_HEIGHT / h)
    return cv2.resize(img, (int(w * scale), int(h * scale)))

def draw_text_panel(question, response, height):
    panel = np.zeros((height, TEXT_PANEL_WIDTH, 3), dtype=np.uint8)

    y = 40
    cv2.putText(panel, "QUERY", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    y += 40
    for line in textwrap.wrap(question, 45):
        cv2.putText(panel, line, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        y += 22

    y += 30
    cv2.putText(panel, "AI RESPONSE", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    y += 35
    for line in textwrap.wrap(response, 45):
        cv2.putText(panel, line, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 22

    return panel

# =============================
# YOLO
# =============================
def detect_all(frame):
    results = yolo_model(frame)
    return results, results[0].plot()

def detect_violations(frame):
    results = yolo_model(frame)
    output = frame.copy()
    detections = []

    for r in results:
        for box in r.boxes:
            label = yolo_model.names[int(box.cls[0])]
            if label in VIOLATION_CLASSES:
                detections.append({"class": label})
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(output, (x1, y1), (x2, y2), (128, 0, 128), 2)
                cv2.putText(output, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    return output, detections

# =============================
# LLM
# =============================
def yolo_to_prompt(detections, question):
    return f"""
User question:
{question}

Detected missing PPE:
{json.dumps(detections, indent=2)}

Rules:
- Answer ONLY based on detected PPE
- Simple English
- Polite and instructive
- One short paragraph
"""

def ask_llm(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content":
                "You are a workplace safety assistant. "
                "Politely instruct the person to wear missing PPE."
            },
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

# =============================
# COMMON IMAGE PIPELINE
# =============================
def process_frame(frame, question):
    global last_frame, last_detections

    frame = resize_for_display(frame)
    last_frame = frame.copy()

    # STEP 1: Original
    cv2.imshow(WINDOW_NAME, frame)
    cv2.waitKey(1)
    time.sleep(DELAY_ORIGINAL)

    # STEP 2: All boxes
    _, all_boxes = detect_all(frame)
    cv2.imshow(WINDOW_NAME, all_boxes)
    cv2.waitKey(1)
    time.sleep(DELAY_ALL_BOXES)

    # STEP 3: Violations only
    violation_img, detections = detect_violations(frame)
    last_detections = detections

    cv2.imshow(WINDOW_NAME, violation_img)
    cv2.waitKey(1)
    time.sleep(DELAY_VIOLATION)

    # STEP 4: Question
    panel_q = draw_text_panel(question, "", violation_img.shape[0])
    cv2.imshow(WINDOW_NAME, np.hstack((violation_img, panel_q)))
    cv2.waitKey(1)
    time.sleep(DELAY_QUESTION)

    # STEP 5: LLM response
    if detections:
        response = ask_llm(yolo_to_prompt(detections, question))
    else:
        response = "No PPE violations detected."

    panel_final = draw_text_panel(question, response, violation_img.shape[0])
    cv2.imshow(WINDOW_NAME, np.hstack((violation_img, panel_final)))

# =============================
# MAIN
# =============================
def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    default_question = "Do you see any safety concerns in the workplace?"

    root = tk.Tk()
    root.title("PPE Safety Assistant")

    canvas = tk.Canvas(root, width=IMG_WIDTH, height=IMG_HEIGHT)
    canvas.pack()

    query_frame = tk.Frame(root)
    query_frame.pack(pady=5)

    tk.Label(query_frame, text="Ask your question:").pack(side="left", padx=5)
    user_query_entry = tk.Entry(query_frame, width=60)
    user_query_entry.pack(side="left", padx=5)

    def update_preview():
        ret, frame = cap.read()
        if ret:
            frame = resize_for_display(frame)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(rgb))
            canvas.imgtk = img
            canvas.create_image(0, 0, anchor="nw", image=img)
        root.after(30, update_preview)

    def capture_live():
        ret, frame = cap.read()
        if not ret:
            return
        winsound.Beep(1000, 200)
        process_frame(frame, default_question)

    def load_image():
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.png *.jpeg")]
        )
        if not path:
            return
        frame = cv2.imread(path)
        if frame is None:
            return
        process_frame(frame, default_question)

    def ask_user_question():
        if last_frame is None:
            return
        question = user_query_entry.get().strip()
        if not question:
            return
        if last_detections:
            response = ask_llm(yolo_to_prompt(last_detections, question))
        else:
            response = "No PPE violations detected."
        panel = draw_text_panel(question, response, last_frame.shape[0])
        cv2.imshow(WINDOW_NAME, np.hstack((last_frame, panel)))

    btns = tk.Frame(root)
    btns.pack(pady=10)

    tk.Button(btns, text="📸 Capture Live", width=18, command=capture_live).grid(row=0, column=0, padx=8)
    tk.Button(btns, text="🗂️ Load Image", width=18, command=load_image).grid(row=0, column=1, padx=8)
    tk.Button(btns, text="❓ Ask Question", width=18, command=ask_user_question).grid(row=0, column=2, padx=8)
    tk.Button(btns, text="❌ Exit", width=18, command=root.destroy).grid(row=0, column=3, padx=8)

    update_preview()
    root.mainloop()

    cap.release()
    cv2.destroyAllWindows()

# =============================
# RUN
# =============================
if __name__ == "__main__":
    main()
