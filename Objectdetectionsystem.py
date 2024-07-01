import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolo-Weights/yolov8n.pt")

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Detect objects in the frame
    results = model(frame, stream=True)

    # Draw bounding boxes around the detected objects
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

            # Print the class name and confidence score
            class_id = int(box.cls[0])
            label = model.names[class_id]
            confidence = box.conf[0]
            print(f"{label}: {confidence}")
    break
# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()
# Save the frame with the detected objects
cv2.imwrite("frame_with_objects.jpg", frame)

