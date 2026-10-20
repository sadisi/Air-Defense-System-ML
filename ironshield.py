import cv2
import torch
import numpy as np
import warnings
import pygame
import random
import time
import os
import glob

# ==============================================================================
# IRONSHIELD - TACTICAL AIR DEFENSE & INTERCEPTION SYSTEM
# Developer: Vilochana Rajapaksha
# ==============================================================================

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Initialize Pygame Audio System
try:
    pygame.mixer.init()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sound_path = os.path.join(base_dir, "warn.wav")
    warning_sound = pygame.mixer.Sound(sound_path) if os.path.exists(sound_path) else None
except Exception:
    warning_sound = None

# Hardware Acceleration Setup (CUDA)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[*] IronShield Compute Engine: {device}")
if torch.cuda.is_available():
    print(f"[*] GPU Acceleration Active: {torch.cuda.get_device_name(0)}")

def load_yolov5():
    try:
        print("[*] Initializing YOLOv5 Military Detection Neural Net...")
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        model.to(device)
        print("[+] Model loaded successfully on GPU!")
        return model
    except Exception as e:
        print(f"[!] Model initialization failure: {str(e)}")
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

# Audio Alert Controller
audio_muted = False
def trigger_alarm():
    global audio_muted
    if audio_muted or warning_sound is None:
        return
    try:
        if not pygame.mixer.get_busy():
            warning_sound.play()
    except Exception:
        pass

# ==============================================================================
# TACTICAL HUD DRAWING UTILITIES (CRYSTAL CLEAR, ANTI-ALIASED)
# ==============================================================================

def draw_corner_rect(img, x, y, w, h, color, corner_len=18, thickness=2):
    """Draws sleek military corner brackets instead of basic rectangles."""
    cv2.line(img, (x, y), (x + min(corner_len, w // 2), y), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x, y), (x, y + min(corner_len, h // 2)), color, thickness, cv2.LINE_AA)

    cv2.line(img, (x + w, y), (x + w - min(corner_len, w // 2), y), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x + w, y), (x + w, y + min(corner_len, h // 2)), color, thickness, cv2.LINE_AA)

    cv2.line(img, (x, y + h), (x + min(corner_len, w // 2), y + h), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x, y + h), (x, y + h - min(corner_len, h // 2)), color, thickness, cv2.LINE_AA)

    cv2.line(img, (x + w, y + h), (x + w - min(corner_len, w // 2), y + h), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x + w, y + h), (x + w, y + h - min(corner_len, h // 2)), color, thickness, cv2.LINE_AA)

    # Subtle translucent center box
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 1)
    cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)

def draw_pill_badge(img, text, x, y, bg_color, text_color=(255, 255, 255), font_scale=0.45, padding=6):
    """Draws high-contrast label with dark rounded background for 100% readable text."""
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    x1, y1 = x, y - th - padding * 2
    x2, y2 = x + tw + padding * 2, y + baseline

    # Keep inside frame
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


# ==============================================================================
# MAIN TRACKING & DEFENSE PIPELINE
# ==============================================================================

def track_and_identify(video_path):
    global audio_muted

    if not os.path.exists(video_path):
        print(f"[!] Error: Target video not found at: {video_path}")
        return

    model = load_yolov5()
    if model is None:
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[!] Error: Could not open video: {video_path}")
        return

    # Window Configuration
    window_name = "IRONSHIELD - Aerial Defense Tactical Command"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 950)

    # Frame timing for FPS calculation
    prev_time = time.time()
    fps = 0.0
    frame_idx = 0
    paused = False

    # Get available videos for playlist switching
    base_dir = os.path.dirname(os.path.abspath(__file__))
    vid_folder = os.path.join(base_dir, "vid")
    available_videos = sorted(glob.glob(os.path.join(vid_folder, "*.mp4")))
    current_vid_idx = 0
    if video_path in available_videos:
        current_vid_idx = available_videos.index(video_path)

    print(f"[*] IronShield Console Active: {os.path.basename(video_path)}")
    print("[*] Keyboard Controls:")
    print("    [Q] Quit | [SPACE] Pause/Resume | [M] Mute Sound | [N] Next Video | [S] Save Snapshot")

    while cap.isOpened():
        if not paused:
            ret, raw_frame = cap.read()
            if not ret:
                # Continuous seamless loop
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame_idx += 1

            # Compute real FPS
            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(0.001, (curr_time - prev_time)))
            prev_time = curr_time

            # ------------------------------------------------------------------
            # 1. DISPLAY NORMALIZATION (Maintain Aspect Ratio within Standard Window)
            # ------------------------------------------------------------------
            orig_h, orig_w = raw_frame.shape[:2]
            target_h = 920
            scale = target_h / float(orig_h)
            target_w = int(orig_w * scale)
            display = cv2.resize(raw_frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

            # ------------------------------------------------------------------
            # 2. RUN NEURAL DETECTION
            # ------------------------------------------------------------------
            boxes, confidences, class_ids = detect_objects(model, raw_frame)

            # ------------------------------------------------------------------
            # 3. DRAW TACTICAL RETICLES & THREAT TELEMETRY
            # ------------------------------------------------------------------
            threat_count = 0
            primary_target = None

            for i in range(len(boxes)):
                bx, by, bw, bh = boxes[i]
                # Map raw frame coords directly to display resolution
                dx = int(bx * scale)
                dy = int(by * scale)
                dw = int(bw * scale)
                dh = int(bh * scale)

                label = str(model.names[class_ids[i]])
                conf = confidences[i]
                cx = dx + dw // 2
                cy = dy + dh // 2

                # Air Defense Reclassification: COCO models frequently misclassify swept-wing/high-altitude planes as 'kite'
                if label in ["kite", "bird"]:
                    label = "airplane"

                is_aerial_threat = label in ["airplane", "plane", "missile", "rocket", "projectile"]
                is_vehicle = label in ["car", "bus", "truck", "van", "bicycle", "ship", "boat"]

                if is_aerial_threat:
                    threat_count += 1
                    color = (0, 0, 255)  # Tactical Red for Airborne Threats

                    if primary_target is None or conf > primary_target['conf']:
                        primary_target = {
                            'label': label,
                            'conf': conf,
                            'cx': cx,
                            'cy': cy,
                            'dx': dx,
                            'dy': dy,
                            'dw': dw,
                            'dh': dh
                        }

                    # Tactical corner brackets around target
                    draw_corner_rect(display, dx, dy, dw, dh, (0, 0, 255), corner_len=24, thickness=2)

                    # Dynamic Target Crosshair
                    cv2.line(display, (cx - 15, cy), (cx + 15, cy), (0, 255, 255), 1, cv2.LINE_AA)
                    cv2.line(display, (cx, cy - 15), (cx, cy + 15), (0, 255, 255), 1, cv2.LINE_AA)
                    cv2.circle(display, (cx, cy), 18, (0, 0, 255), 1, cv2.LINE_AA)

                    # Lead trajectory axis lines
                    cv2.line(display, (cx, 0), (cx, target_h), (0, 100, 255), 1, cv2.LINE_AA)
                    cv2.line(display, (0, cy), (target_w, cy), (0, 100, 255), 1, cv2.LINE_AA)

                    # Crystal clear target tag
                    tag_text = f"THREAT: {label.upper()} [{int(conf * 100)}%]"
                    draw_pill_badge(display, tag_text, dx, dy - 6, (0, 0, 180), (255, 255, 255), 0.5)

                elif is_vehicle:
                    color = (255, 120, 0)  # Cyan/Blue for Ground/Naval Craft
                    draw_corner_rect(display, dx, dy, dw, dh, color, corner_len=14, thickness=1)
                    draw_pill_badge(display, f"{label.upper()} {int(conf * 100)}%", dx, dy - 4, (100, 40, 0), (255, 255, 255), 0.45)
                else:
                    color = (0, 255, 120)  # High-contrast Green for general objects
                    draw_corner_rect(display, dx, dy, dw, dh, color, corner_len=12, thickness=1)
                    draw_pill_badge(display, f"{label} {int(conf * 100)}%", dx, dy - 4, (0, 80, 20), (255, 255, 255), 0.4)

            # Sound Alert
            if threat_count > 0:
                trigger_alarm()

            # ------------------------------------------------------------------
            # 4. MILITARY HUD GLASSMORPHISM PANELS (CRYSTAL CLEAR TELEMETRY)
            # ------------------------------------------------------------------
            hud_overlay = display.copy()

            # TOP STATUS BANNER
            cv2.rectangle(hud_overlay, (0, 0), (target_w, 42), (15, 15, 15), -1)
            cv2.line(hud_overlay, (0, 42), (target_w, 42), (0, 200, 255), 1)

            # BOTTOM CONTROLS & DIAGNOSTICS BAR
            cv2.rectangle(hud_overlay, (0, target_h - 38), (target_w, target_h), (15, 15, 15), -1)
            cv2.line(hud_overlay, (0, target_h - 38), (target_w, target_h - 38), (0, 200, 255), 1)

            # LEFT RADAR TELEMETRY DRAWER (if threats exist)
            if primary_target:
                box_w = 260
                box_h = 160
                cv2.rectangle(hud_overlay, (12, 52), (12 + box_w, 52 + box_h), (10, 10, 15), -1)
                cv2.rectangle(hud_overlay, (12, 52), (12 + box_w, 52 + box_h), (0, 0, 255), 1)

            # Blend translucent HUD background (80% opacity)
            cv2.addWeighted(hud_overlay, 0.82, display, 0.18, 0, display)

            # --- DRAW SHARP TEXT ON TOP STATUS BAR ---
            status_text = "IRONSHIELD AIR DEFENSE CONSOLE"
            cv2.putText(display, status_text, (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

            threat_badge = f"THREATS: {threat_count}" if threat_count > 0 else "AIRSPACE: SECURE"
            threat_color = (0, 0, 255) if threat_count > 0 else (0, 255, 0)
            cv2.putText(display, threat_badge, (target_w // 2 - 50, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, threat_color, 2, cv2.LINE_AA)

            gpu_dev = torch.cuda.get_device_name(0).replace("NVIDIA GeForce ", "") if torch.cuda.is_available() else "CPU"
            gpu_status = f"ENGINE: {gpu_dev} | {int(fps)} FPS"
            (gw, _), _ = cv2.getTextSize(gpu_status, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.putText(display, gpu_status, (target_w - gw - 15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

            # --- DRAW SHARP TEXT ON TELEMETRY PANEL ---
            if primary_target:
                p = primary_target
                cv2.putText(display, "TARGET ACQUIRED: LOCK CONFIRMED", (24, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.putText(display, f"CLASSIFICATION: {p['label'].upper()}", (24, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(display, f"PROBABILITY: {int(p['conf'] * 100)}% (CONFIRMED)", (24, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(display, f"AZIMUTH GRID: X:{p['cx']} Y:{p['cy']}", (24, 142), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
                cv2.putText(display, f"DEFENSE STATUS: ACTIVE TARGET LOCK", (24, 164), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)

                # Simulated Range Bar
                cv2.rectangle(display, (24, 182), (24 + 180, 192), (50, 50, 50), -1)
                fill_w = int(180 * p['conf'])
                cv2.rectangle(display, (24, 182), (24 + fill_w, 192), (0, 0, 255), -1)

            # --- BOTTOM SHORTCUT BAR ---
            mute_str = "[M] SOUND: OFF" if audio_muted else "[M] SOUND: ON"
            controls_text = f"[SPACE] PAUSE | {mute_str} | [N] NEXT VID | [S] SNAPSHOT | [Q] QUIT"
            cv2.putText(display, controls_text, (15, target_h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

            # Display Video Name
            vname = os.path.basename(video_path)
            (vw, _), _ = cv2.getTextSize(vname, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            cv2.putText(display, vname, (target_w - vw - 15, target_h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)

            # ------------------------------------------------------------------
            # 6. RENDER FRAME
            # ------------------------------------------------------------------
            cv2.imshow(window_name, display)

        # Keyboard Interactivity
        key = cv2.waitKey(20 if not paused else 100) & 0xFF

        if key == ord('q') or key == 27:  # Q or ESC
            print("[*] User quit requested.")
            break
        elif key == ord(' '):  # SPACE to pause/resume
            paused = not paused
            print(f"[*] Simulation {'PAUSED' if paused else 'RESUMED'}")
        elif key == ord('m') or key == ord('M'):  # Mute toggle
            audio_muted = not audio_muted
            print(f"[*] Sound alerts: {'MUTED' if audio_muted else 'ACTIVE'}")
        elif key == ord('s') or key == ord('S'):  # Screenshot
            snap_path = os.path.join(base_dir, f"snapshot_{int(time.time())}.jpg")
            cv2.imwrite(snap_path, display)
            print(f"[+] Snapshot saved to: {snap_path}")
        elif key == ord('n') or key == ord('N'):  # Next Video
            if available_videos:
                current_vid_idx = (current_vid_idx + 1) % len(available_videos)
                video_path = available_videos[current_vid_idx]
                cap.release()
                cap = cv2.VideoCapture(video_path)
                print(f"[*] Switched to next video: {os.path.basename(video_path)}")

        # Check if window was closed via [X]
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            print("[*] Window closed by user.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[+] IronShield Tactical Console Shutdown.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(base_dir, "vid", "15285393_1080_1922_60fps.mp4")
    track_and_identify(video_path)
