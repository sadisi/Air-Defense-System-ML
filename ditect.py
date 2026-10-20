import cv2
import torch
import numpy as np
import warnings
import time
import os
import glob

# ==============================================================================
# OBJECT DETECTION & TRACKING CONSOLE
# Developer: Vilochana Rajapaksha
# ==============================================================================

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[*] Detection Engine Device: {device}")
if torch.cuda.is_available():
    print(f"[*] GPU Acceleration Active: {torch.cuda.get_device_name(0)}")

def load_yolov5():
    try:
        print("[*] Loading YOLOv5 Neural Network...")
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        model.to(device)
        print("[+] YOLOv5 loaded successfully on GPU!")
        return model
    except Exception as e:
        print(f"[!] Model loading error: {str(e)}")
        return None

def detect_objects(model, frame):
    results = model([frame], size=640)
    detections = results.pandas().xyxy[0]

    boxes = []
    confidences = []
    class_ids = []

    for _, row in detections.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        confidence = float(row['confidence'])
        class_id = int(row['class'])

        boxes.append([x1, y1, x2 - x1, y2 - y1])
        confidences.append(confidence)
        class_ids.append(class_id)

    return boxes, confidences, class_ids

def draw_corner_rect(img, x, y, w, h, color, corner_len=18, thickness=2):
    cv2.line(img, (x, y), (x + min(corner_len, w // 2), y), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x, y), (x, y + min(corner_len, h // 2)), color, thickness, cv2.LINE_AA)

    cv2.line(img, (x + w, y), (x + w - min(corner_len, w // 2), y), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x + w, y), (x + w, y + min(corner_len, h // 2)), color, thickness, cv2.LINE_AA)

    cv2.line(img, (x, y + h), (x + min(corner_len, w // 2), y + h), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x, y + h), (x, y + h - min(corner_len, h // 2)), color, thickness, cv2.LINE_AA)

    cv2.line(img, (x + w, y + h), (x + w - min(corner_len, w // 2), y + h), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x + w, y + h), (x + w, y + h - min(corner_len, h // 2)), color, thickness, cv2.LINE_AA)

    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 1)
    cv2.addWeighted(overlay, 0.20, img, 0.80, 0, img)

def draw_pill_badge(img, text, x, y, bg_color, text_color=(255, 255, 255), font_scale=0.45, padding=6):
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    x1, y1 = x, y - th - padding * 2
    x2, y2 = x + tw + padding * 2, y + baseline

    h, w = img.shape[:2]
    if y1 < 5:
        y1 = y + 5
        y2 = y1 + th + padding * 2
        text_y = y1 + th + padding
    else:
        text_y = y - padding

    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)

    cv2.putText(img, text, (x1 + padding, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1, cv2.LINE_AA)

def track_and_identify(video_path):
    if not os.path.exists(video_path):
        print(f"[!] Error: Target video not found: {video_path}")
        return

    model = load_yolov5()
    if model is None:
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[!] Error: Could not open video: {video_path}")
        return

    window_name = "DETECTION CONSOLE - Realtime Optical Recognition"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 950)

    prev_time = time.time()
    fps = 0.0
    paused = False

    base_dir = os.path.dirname(os.path.abspath(__file__))
    vid_folder = os.path.join(base_dir, "vid")
    available_videos = sorted(glob.glob(os.path.join(vid_folder, "*.mp4")))
    current_vid_idx = 0
    if video_path in available_videos:
        current_vid_idx = available_videos.index(video_path)

    print(f"[*] Detection Console Active: {os.path.basename(video_path)}")
    print("[*] Keyboard Controls:")
    print("    [Q] Quit | [SPACE] Pause/Resume | [N] Next Video | [S] Save Snapshot")

    while cap.isOpened():
        if not paused:
            ret, raw_frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(0.001, (curr_time - prev_time)))
            prev_time = curr_time

            orig_h, orig_w = raw_frame.shape[:2]
            target_h = 920
            scale = target_h / float(orig_h)
            target_w = int(orig_w * scale)
            display = cv2.resize(raw_frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

            boxes, confidences, class_ids = detect_objects(model, raw_frame)

            threat_count = 0
            for i in range(len(boxes)):
                bx, by, bw, bh = boxes[i]
                dx = int(bx * scale)
                dy = int(by * scale)
                dw = int(bw * scale)
                dh = int(bh * scale)

                label = str(model.names[class_ids[i]])
                conf = confidences[i]

                # Air Defense Reclassification: COCO models frequently misclassify swept-wing/high-altitude planes as 'kite'
                if label in ["kite", "bird"]:
                    label = "airplane"

                is_aerial = label in ["airplane", "plane", "missile", "rocket", "projectile"]
                is_vehicle = label in ["car", "bus", "truck", "van", "bicycle", "ship", "boat"]

                if is_aerial:
                    threat_count += 1
                    draw_corner_rect(display, dx, dy, dw, dh, (0, 0, 255), corner_len=24, thickness=2)
                    draw_pill_badge(display, f"AIR TARGET: {label.upper()} [{int(conf * 100)}%]", dx, dy - 6, (0, 0, 180), (255, 255, 255), 0.5)
                elif is_vehicle:
                    draw_corner_rect(display, dx, dy, dw, dh, (255, 120, 0), corner_len=14, thickness=1)
                    draw_pill_badge(display, f"{label.upper()} {int(conf * 100)}%", dx, dy - 4, (100, 40, 0), (255, 255, 255), 0.45)
                else:
                    draw_corner_rect(display, dx, dy, dw, dh, (0, 255, 120), corner_len=12, thickness=1)
                    draw_pill_badge(display, f"{label} {int(conf * 100)}%", dx, dy - 4, (0, 80, 20), (255, 255, 255), 0.4)

            # TOP & BOTTOM HUD BARS
            hud_overlay = display.copy()
            cv2.rectangle(hud_overlay, (0, 0), (target_w, 42), (15, 15, 15), -1)
            cv2.line(hud_overlay, (0, 42), (target_w, 42), (0, 255, 255), 1)

            cv2.rectangle(hud_overlay, (0, target_h - 38), (target_w, target_h), (15, 15, 15), -1)
            cv2.line(hud_overlay, (0, target_h - 38), (target_w, target_h - 38), (0, 255, 255), 1)
            cv2.addWeighted(hud_overlay, 0.82, display, 0.18, 0, display)

            cv2.putText(display, "AIR DEFENSE - OPTICAL RECOGNITION", (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
            badge_text = f"TARGETS DETECTED: {len(boxes)}"
            cv2.putText(display, badge_text, (target_w // 2 - 70, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

            gpu_dev = torch.cuda.get_device_name(0).replace("NVIDIA GeForce ", "") if torch.cuda.is_available() else "CPU"
            gpu_status = f"ENGINE: {gpu_dev} | {int(fps)} FPS"
            (gw, _), _ = cv2.getTextSize(gpu_status, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.putText(display, gpu_status, (target_w - gw - 15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

            controls_text = "[SPACE] PAUSE | [N] NEXT VID | [S] SNAPSHOT | [Q] QUIT"
            cv2.putText(display, controls_text, (15, target_h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

            vname = os.path.basename(video_path)
            (vw, _), _ = cv2.getTextSize(vname, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            cv2.putText(display, vname, (target_w - vw - 15, target_h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)

            cv2.imshow(window_name, display)

        key = cv2.waitKey(20 if not paused else 100) & 0xFF
        if key == ord('q') or key == 27:
            print("[*] User quit requested.")
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('s') or key == ord('S'):
            snap_path = os.path.join(base_dir, f"detect_snapshot_{int(time.time())}.jpg")
            cv2.imwrite(snap_path, display)
            print(f"[+] Snapshot saved: {snap_path}")
        elif key == ord('n') or key == ord('N'):
            if available_videos:
                current_vid_idx = (current_vid_idx + 1) % len(available_videos)
                video_path = available_videos[current_vid_idx]
                cap.release()
                cap = cv2.VideoCapture(video_path)
                print(f"[*] Switched to next video: {os.path.basename(video_path)}")

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            print("[*] Window closed by user.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[+] Detection Console Closed.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(base_dir, "vid", "15285393_1080_1922_60fps.mp4")
    track_and_identify(video_path)
