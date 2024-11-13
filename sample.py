import torch
import cv2
from PIL import Image
import numpy as np

# Load a larger YOLOv5 model (using 'yolov5x' for more detailed detection)
model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)

# Load the image
image_path = 'C:/users/kanis/Desktop/yolo/yolov5/data/images/1.jpg'  
img = cv2.imread(image_path)

# Check if image is loaded
if img is None:
    print("Error loading image. Please check the image path.")
else:
    print("Image loaded successfully.")

# Run YOLOv5 on the original image without resizing (better for close-up objects like test tubes)
results = model(img)

# Extract the bounding box coordinates and other details from results
detections = results.xyxy[0]  # Get bounding boxes with coordinates, confidence, and class id

# Set a lower NMS threshold to differentiate close objects (default is 0.45, adjust if needed)
confidence_threshold = 0.25

# Draw bounding boxes and average color overlays on the original image
for row in detections:
    x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
    confidence, class_id = row[4], int(row[5])

    # Filter detections with low confidence
    if confidence > confidence_threshold:
        # Draw bounding boxes
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Object {confidence:.2f}"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Extract the bounding box area and compute the average color in RGB
        bounding_box = img[y1:y2, x1:x2]
        if bounding_box.size == 0:
            continue  # Skip if bounding box is empty
        avg_color_per_row = np.average(bounding_box, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)
        avg_color_rgb = avg_color[::-1]  # Convert BGR to RGB

        # Define the overlay area (e.g., bottom of the bounding box)
        overlay_height = 30  # Height of the overlay
        overlay_y1 = y2 - overlay_height
        overlay_y1 = max(overlay_y1, y1)  # Ensure overlay is within the bounding box
        overlay = img[overlay_y1:y2, x1:x2].copy()

        # Create a semi-transparent overlay
        overlay[:] = avg_color.tolist()
        alpha = 0.5  # Transparency factor
        cv2.addWeighted(overlay, alpha, img[overlay_y1:y2, x1:x2], 1 - alpha, 0, img[overlay_y1:y2, x1:x2])

        # Optionally, add text indicating the average color
        text_position = (x1 + 5, y2 - 5)
        cv2.putText(img, f'Color: ({int(avg_color_rgb[0])}, {int(avg_color_rgb[1])}, {int(avg_color_rgb[2])})',
                    text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

# Convert the OpenCV image back to PIL for display
result_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
result_image.show()
result_image.save('C:/users/kanis/Desktop/yolo/yolov5/data/images/result.jpg')
