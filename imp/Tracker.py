import cv2
import torch
import warnings
import pygame
import random
from ultralytics import YOLO

# Initialize pygame mixer for sound
pygame.mixer.init()

warnings.filterwarnings("ignore", category=UserWarning)


def load_yolov8():
    try:
        # Load YOLOv8 model from the official ultralytics package
        model = YOLO('yolov8n.pt')  # YOLOv8 small model, you can use 'yolov8n.pt' or other models
        print("YOLOv8 model loaded successfully!")
        return model
    except Exception as e:
        print(f"Failed to load YOLOv8 model: {str(e)}")
        return None


model = load_yolov8()


def detect_objects(model, frame):
    # Convert frame to YOLOv8 format and perform inference
    results = model(frame)

    boxes = []
    confidences = []
    class_ids = []

    # Parse results and extract bounding boxes, confidence, and class ids
    for result in results:
        for r in result.boxes:
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            confidences.append(float(r.conf))
            class_ids.append(int(r.cls))

    return boxes, confidences, class_ids


def play_warning_sound():
    # Play warning sound for detected missile or high-priority object
    sound_file = "D:/mlvideo/warn.mp3"
    pygame.mixer.music.load(sound_file)
    pygame.mixer.music.play()


def simulate_interception(frame, x, y):
    # Simulate missile interception with squares
    outer_square_color = (0, 0, 255)  # Red for outer square
    inner_square_color = (0, 255, 0)  # Green for inner square

    outer_square_size = 100  # Size of outer square
    inner_square_size = 60  # Size of inner square

    top_left_outer = (x - outer_square_size // 2, y - outer_square_size // 2)
    bottom_right_outer = (x + outer_square_size // 2, y + outer_square_size // 2)

    top_left_inner = (x - inner_square_size // 2, y - inner_square_size // 2)
    bottom_right_inner = (x + inner_square_size // 2, y + inner_square_size // 2)

    # Draw outer square
    cv2.rectangle(frame, top_left_outer, bottom_right_outer, outer_square_color, 3)

    # Draw inner square
    cv2.rectangle(frame, top_left_inner, bottom_right_inner, inner_square_color, 2)

    cv2.putText(frame, "Threat Neutralized!", (x - 100, y - outer_square_size // 2 - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


def draw_labels_and_simulate_defense(boxes, confidences, class_ids, classes, frame):
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        label = classes[class_ids[i]]
        confidence = confidences[i]

        if label in ["missile", "rocket", "airplane", "plane"]:  # High-priority threats
            color = (0, 0, 255)  # Red for dangerous objects
            play_warning_sound()

            frame_height, frame_width = frame.shape[:2]
            cv2.line(frame, (x + w // 2, 0), (x + w // 2, frame_height), (0, 255, 255), 2)
            cv2.line(frame, (0, y + h // 2), (frame_width, y + h // 2), (0, 255, 255), 2)

            grid_x = x + w // 2
            grid_y = y + h // 2
            cv2.putText(frame, f"Grid Data:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"X: {grid_x}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Y: {grid_y}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if random.random() < 0.8:  # 80% chance of successful interception
                simulate_interception(frame, grid_x, grid_y)
            else:
                cv2.putText(frame, "Interception Failed!", (x, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        elif label in ["car", "bus", "van", "bicycle", "vehicle", "ship"]:
            color = (255, 0, 0)  # Blue for non-living objects
        else:
            color = (0, 255, 0)  # Green for others

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{label} {int(confidence * 100)}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def track_and_identify(video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        boxes, confidences, class_ids = detect_objects(model, frame)
        draw_labels_and_simulate_defense(boxes, confidences, class_ids, model.names, frame)

        cv2.imshow("Tracking Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = "D:/mlvideo/cf.mp4"
    track_and_identify(video_path)
