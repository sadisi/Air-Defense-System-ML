
import cv2
import torch
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def load_yolov5():
    try:
        # Attempt to load the pre-trained YOLOv5 model
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        print("YOLOv5 model loaded successfully!")
        return model
    except Exception as e:
        print(f"Failed to load YOLOv5 model: {str(e)}")
        return None


model = load_yolov5()

if model is None:
    print("Exiting the program due to model loading failure.")
else:
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


    def draw_labels(boxes, confidences, class_ids, classes, frame):
        for i in range(len(boxes)):
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]

            # Color the box based on the detected object class
            if label in ["missile", "rocket", "projectile", "airplane", "plane", "router"]:
                color = (0, 0, 255)  # Red color for missiles
            elif label in ["car", "bus", "van", "bicycle", "vehicle", "ship", "phone", "book", "router"]:
                color = (255, 0, 0)  # Blue color for non-living objects
            else:
                color = (0, 255, 0)  # Green for other objects

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {int(confidence * 100)}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,
                        2)


    def track_and_identify(video_path=None, image_path=None):
        # Check if OpenCV supports setNumThreads before calling it
        if hasattr(cv2, 'setNumThreads'):
            cv2.setNumThreads(0)

        if video_path:
            cap = cv2.VideoCapture(video_path)
        elif image_path:
            frame = cv2.imread(image_path)

            # Resize image if it's larger than 1920x1080
            max_height, max_width = 1080, 1920
            height, width = frame.shape[:2]
            if height > max_height or width > max_width:
                scale_factor = min(max_width / width, max_height / height)
                frame = cv2.resize(frame, (int(width * scale_factor), int(height * scale_factor)))

            boxes, confidences, class_ids = detect_objects(model, frame)
            draw_labels(boxes, confidences, class_ids, model.names, frame)
            cv2.imshow("Reading Image", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return
        else:
            cap = cv2.VideoCapture(0)  # Open webcam

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            boxes, confidences, class_ids = detect_objects(model, frame)
            draw_labels(boxes, confidences, class_ids, model.names, frame)

            cv2.imshow("WebCam O-D", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


    if __name__ == "__main__":
        video_path = None  # video file path or None for webcam
        image_path = None # image file path or None for video/webcam
        track_and_identify(video_path=video_path, image_path=image_path)
