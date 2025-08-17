import cv2
from ultralytics import YOLO
from PIL import Image, ImageTk
import tkinter as tk
import threading
import os
import pyttsx3
import datetime
import requests

MODEL_PATH = "../weights/best.pt"
CLASSES = ["ToolBox", "FireExtinguisher", "OxygenTank"]

API_KEY = "AIzaSyCfCsYRycue9HjUtmEUzqN5LEkTkGuTnY4"

def get_ai_sop(object_name):
    prompt = f"""You are an expert safety assistant on a space station.
For the object "{object_name}", provide a numbered Standard Operating Procedure (SOP) for immediate use (max 25 words, 2-4 steps), and below that a one-line critical safety warning.
Format: SOP: <steps-list>, Warning: <one-line warn>. No intro/extras."""
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
    headers = {"Content-Type": "application/json"}
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        r = requests.post(f"{url}?key={API_KEY}", headers=headers, json=data, timeout=10)
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        return "SOP: 1. Standard steps here. Warning: AI API error/timeout!"

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Space Station Safety App – Gemini AI LIVE SOP")
        self.model = YOLO(MODEL_PATH)
        self.cap = None
        self.running = False

        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 165)
        self.prev_objects = set()
        self.last_detected_frame = None

        banner_frame = tk.Frame(root, bg="#23272e")
        banner_frame.pack(fill='x')
        tk.Label(banner_frame, text="🚀 SPACE STATION SAFETY DASHBOARD 🚀",
                 font=("Segoe UI Black", 20, "bold"), fg="#4dfcff", bg="#23272e").pack(pady=3)
        tk.Label(banner_frame, text="by Team Yantrik Minds",
                 font=("Segoe UI", 16, "italic"), fg="#ffbf00", bg="#23272e").pack(pady=(0, 8))

        stats_panel = tk.Frame(root, bg="#222")
        stats_panel.pack(fill='x', pady=(0,5))
        self.stats_labels = {}
        for cl in CLASSES:
            lbl = tk.Label(stats_panel, text=f"{cl}: 0", font=("Arial", 13, "bold"), fg="#16f579", bg="#222", width=18)
            lbl.pack(side="left", padx=6, pady=1)
            self.stats_labels[cl] = lbl

        self.video_lbl = tk.Label(root)
        self.video_lbl.pack(fill='both', expand=True)

        controls = tk.Frame(root)
        controls.pack(fill='x')
        self.toggle_btn = tk.Button(controls, text="▶️ Start Camera", command=self.toggle, width=20, bg="#28a745", fg="white", font=("Arial", 14, "bold"))
        self.toggle_btn.pack(side="left", padx=10, pady=8)
        self.snapshot_btn = tk.Button(controls, text="📸 Save Snapshot", command=self.save_snapshot, width=18, bg="#007FFF", fg="white", font=("Arial", 14, "bold"))
        self.snapshot_btn.pack(side="left", padx=10, pady=8)
        self.clear_btn = tk.Button(controls, text="🗑️ Clear Info", command=self.clear_info, width=16, bg="#dc3545", fg="white", font=("Arial", 14, "bold"))
        self.clear_btn.pack(side="left", padx=10, pady=8)

        self.info_text = tk.Text(root, height=7, font=("Consolas", 13), bg="#222", fg="#0ff", bd=2, relief="groove")
        self.info_text.pack(fill='x', padx=8, pady=1)
        self.info_text.config(state=tk.DISABLED)

        self.detected_objects_info = {}
        self.live_counts = {cl:0 for cl in CLASSES}
        self.info_key_ai_cache = {}

    def update_stats(self):
        for cl in CLASSES:
            self.stats_labels[cl].config(text=f"{cl}: {self.live_counts[cl]}")

    def toggle(self):
        if self.running:
            self.running = False
            self.toggle_btn.config(text="▶️ Start Camera", bg="#28a745")
            if self.cap:
                self.cap.release()
        else:
            self.running = True
            self.toggle_btn.config(text="⏹️ Stop Camera", bg="#dc3545")
            self.cap = cv2.VideoCapture(0)
            self.detected_objects_info = {}
            self.live_counts = {cl:0 for cl in CLASSES}
            self.prev_objects = set()
            self.update_stats()
            self.update_frame()

    def clear_info(self):
        self.detected_objects_info = {}
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete('1.0', tk.END)
        self.info_text.config(state=tk.DISABLED)
        self.live_counts = {cl:0 for cl in CLASSES}
        self.update_stats()
        self.prev_objects = set()
        self.info_key_ai_cache = {}

    def save_snapshot(self):
        if self.last_detected_frame is not None:
            dt = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            savepath = f"snapshot_{dt}.jpg"
            img = Image.fromarray(self.last_detected_frame)
            img.save(savepath)
            tk.messagebox.showinfo("Snapshot Saved", f"Image saved as {savepath}")
        else:
            tk.messagebox.showwarning("No Frame", "No frame to save. Start the camera first.")

    def update_frame(self):
        if not self.running:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(30, self.update_frame)
            return
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model(rgb_frame, verbose=False)

        frame_objects = set()
        if results and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            cls_ids = results[0].boxes.cls.cpu().numpy()
            names = results[0].names
            confs = results[0].boxes.conf.cpu().numpy()
            for i, box in enumerate(boxes):
                cls_id = int(cls_ids[i])
                label = names[cls_id]
                if label not in CLASSES: continue
                frame_objects.add(label)
                x1, y1, x2, y2 = map(int, box)
                conf = confs[i]
                conf_pct = int(conf*100)
                color = (34,204,68) if conf > 0.85 else (255,140,0) if conf > 0.6 else (220,30,30)
                cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), color, 6)
                label_text = f"{label} ({conf_pct}%)"
                (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 1.3, 2)
                cv2.rectangle(rgb_frame, (x1, y1 - h - 12), (x1 + w + 28, y1), color, -1)
                cv2.putText(rgb_frame, label_text, (x1 + 10, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (20, 20, 20), 3)
                if label not in self.detected_objects_info:
                    # Fetching live from Gemini/Google API (cached)
                    if label not in self.info_key_ai_cache:
                        self.info_key_ai_cache[label] = get_ai_sop(label)
                    self.detected_objects_info[label] = self.info_key_ai_cache[label]
                    self.append_info(f"{label}\n{self.detected_objects_info[label]}\n")
        # Event-based counting, TTS + info
        new_objects = frame_objects - self.prev_objects
        for label in new_objects:
            self.live_counts[label] += 1
            if label in self.detected_objects_info and label in ["FireExtinguisher", "OxygenTank"]:
                threading.Thread(target=self.tts_speak, args=(f"{label} detected. {self.detected_objects_info[label]}",)).start()
        self.update_stats()
        self.prev_objects = frame_objects.copy()
        self.last_detected_frame = rgb_frame.copy()
        img = Image.fromarray(rgb_frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_lbl.imgtk = imgtk
        self.video_lbl.config(image=imgtk)
        if self.running: self.root.after(30, self.update_frame)

    def tts_speak(self, text):
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except:
            pass

    def append_info(self, text):
        self.info_text.config(state=tk.NORMAL)
        self.info_text.insert(tk.END, text + "\n")
        self.info_text.see(tk.END)
        self.info_text.config(state=tk.DISABLED)

if __name__ == "__main__":
    from tkinter import messagebox
    root = tk.Tk()
    root.geometry("1150x900")
    app = App(root)
    root.mainloop()
