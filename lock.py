
import cv2
import torch
import numpy as np
import warnings
import pygame

warnings.filterwarnings("ignore", category=FutureWarning)

# Initialize pygame mixer
pygame.mixer.init()

def load_yolov5():
    # Load the pre-trained YOLOv5 model
    try:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        print("YOLOv5 model loaded successfully!")
        return model
    except Exception as e:
        print(f"Failed to load YOLOv5 model: {str(e)}")
        return None

model = load_yolov5()

def detect_objects(model, frame):
    # Convert the frame to a format suitable for YOLOv5
    img = [frame]

    # Perform object detection
    results = model(img)

    # Parse the results
    detections = results.pandas().xyxy[0]

    boxes = []
    confidences = []
    class_ids = []

    for index, row in detections.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        confidence = row['confidence']
        class_id = int(row['class'])
        label = row['name']

        # Append detected box, confidence, and class id
        boxes.append([x1, y1, x2 - x1, y2 - y1])
        confidences.append(confidence)
        class_ids.append(class_id)

    return boxes, confidences, class_ids

def play_warning_sound():
    # Path to the warning sound file
    sound_file = "D:/mlvideo/warn.mp3"
    pygame.mixer.music.load(sound_file)
    pygame.mixer.music.play()

def draw_labels(boxes, confidences, class_ids, classes, frame):
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]

        # Color the box based on the detected object class
        if label in ["missile", "rocket", "airplane", "plane"]:  # model labels
            color = (0, 0, 255)  # Red color for missiles and airplanes
            if label in ["missile", "rocket", "airplane", "plane"]:
                play_warning_sound()

                # Draw the X and Y axis for locking onto the plane
                frame_height, frame_width = frame.shape[:2]
                cv2.line(frame, (x + w // 2, 0), (x + w // 2, frame_height), (0, 255, 255), 2)  # Y axis
                cv2.line(frame, (0, y + h // 2), (frame_width, y + h // 2), (0, 255, 255), 2)  # X axis

                # Draw grid data on the left side of the video screen
                grid_x = x + w // 2
                grid_y = y + h // 2
                cv2.putText(frame, f"Locked Grid Data:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"X: {grid_x}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Y: {grid_y}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        elif label in ["car", "bus", "van", "bicycle", "vehicle", "ship"]:
            color = (255, 0, 0)  # Blue color for non-living objects
        else:
            color = (0, 255, 0)  # Green for other objects

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{label} {int(confidence * 100)}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def track_and_identify(video_path):
    # Check if OpenCV supports setNumThreads before calling it
    if hasattr(cv2, 'setNumThreads'):
        cv2.setNumThreads(0)

    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        boxes, confidences, class_ids = detect_objects(model, frame)
        draw_labels(boxes, confidences, class_ids, model.names, frame)

        cv2.imshow("Tracking Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "D:/mlvideo/cf.mp4"
    track_and_identify(video_path)
