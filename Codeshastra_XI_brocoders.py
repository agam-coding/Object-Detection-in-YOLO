import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim
from ultralytics import YOLO
import os
from ultralytics import YOLO

yolo_model = YOLO("yolov8n.pt")  # Use the standard small YOLOv8 model




yolo_model = YOLO("/Users/agamkamdar/Documents/GitHub/Codeshastra_XI_brocoders/yolov8n.pt")
def detect_objects_with_yolo(image):
    results = yolo_model(image)[0]  # Get prediction result
    annotated_frame = image.copy()
    detected_objects = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0].item())
        label = yolo_model.names[cls_id]
        conf = float(box.conf[0])

        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        detected_objects.append({
            "label": label,
            "confidence": conf,
            "bbox": (x1, y1, x2, y2),
            "center": ((x1 + x2) // 2, (y1 + y2) // 2)
        })

    return annotated_frame, detected_objects

def compare_images(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    yolo_img1, detections1 = detect_objects_with_yolo(img1)
    yolo_img2, detections2 = detect_objects_with_yolo(img2)

    cv2.imshow("YOLO Detection - Image 1", yolo_img1)
    cv2.imshow("YOLO Detection - Image 2", yolo_img2)

    # Detect Missing and Shifted Objects
    missing_objects = []
    shifted_objects = []

    for obj1 in detections1:
        label1 = obj1["label"]
        center1 = obj1["center"]

        matched = False
        for obj2 in detections2:
            if obj1["label"] == obj2["label"]:
                dist = np.linalg.norm(np.array(center1) - np.array(obj2["center"]))
                if dist < 50:  # Small distance means it's same object
                    matched = True
                    break
                else:
                    shifted_objects.append((label1, center1, obj2["center"]))
                    matched = True
                    break

        if not matched:
            missing_objects.append(label1)

    print("\n--- Object Analysis ---")
    if missing_objects:
        print("Missing Objects:")
        for obj in missing_objects:
            print(f" - {obj}")
    else:
        print("No missing objects.")

    if shifted_objects:
        print("Shifted Objects:")
        for label, old, new in shifted_objects:
            print(f" - {label} shifted from {old} to {new}")
    else:
        print("No shifted objects.")

    # Optional: Draw shifts on yolo_img2
    for label, old, new in shifted_objects:
        cv2.arrowedLine(yolo_img2, old, new, (0, 0, 255), 2)
        cv2.putText(yolo_img2, f"{label} moved", new, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imshow("Changes Visualized", yolo_img2)

def real_time_detection():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    prev_frame = None
    print("Real-time mode started. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        frame = cv2.resize(frame, (640, 480))

        if prev_frame is not None:
            compare_images_live(prev_frame, frame)

        prev_frame = frame.copy()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def compare_images_live(prev_frame, curr_frame):
    gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    yolo_frame_prev, detections_prev = detect_objects_with_yolo(prev_frame)
    yolo_frame_curr, detections_curr = detect_objects_with_yolo(curr_frame)

    annotated_frame = yolo_frame_curr.copy()

    # Compare objects
    missing_objects = []
    shifted_objects = []

    for obj1 in detections_prev:
        label1 = obj1["label"]
        center1 = obj1["center"]
        bbox1 = obj1["bbox"]

        matched = False
        for obj2 in detections_curr:
            if obj1["label"] == obj2["label"]:
                dist = np.linalg.norm(np.array(center1) - np.array(obj2["center"]))
                if dist < 50:
                    matched = True
                    break
                else:
                    shifted_objects.append((label1, center1, obj2["center"]))
                    matched = True
                    break
        if not matched:
            missing_objects.append((label1, bbox1))

    # 🔴 Draw Missing Objects
    for label, bbox in missing_objects:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(annotated_frame, f"{label} Missing", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 🟡 Draw Shifted Objects
    for label, old_center, new_center in shifted_objects:
        cv2.arrowedLine(annotated_frame, old_center, new_center, (0, 255, 255), 2)
        cv2.putText(annotated_frame, f"{label} Shifted", new_center,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Show final visualized output
    cv2.imshow("YOLO Detection - Live Changes", annotated_frame)

    # SSIM & subtraction visuals (unchanged)
    score, ssim_diff = ssim(gray1, gray2, full=True)
    ssim_diff = (ssim_diff * 255).astype("uint8")
    ssim_thresh = cv2.threshold(ssim_diff, 200, 255, cv2.THRESH_BINARY_INV)[1]

    diff_sub = cv2.absdiff(gray1, gray2)
    _, thresh_sub = cv2.threshold(diff_sub, 30, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh_sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = curr_frame.copy()
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(contour_img, (x + w, y + h), (x, y), (0, 255, 0), 2)

    cv2.imshow("Live Feed", curr_frame)
    cv2.imshow("SSIM Difference", ssim_thresh)
    cv2.imshow("Subtracted Differences", thresh_sub)
    cv2.imshow("Contours (Detected Changes)", contour_img)

    print(f"[SSIM Live] Score: {score:.4f}")

def capture_images_from_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not accessible.")
        return

    print("Press 'c' to capture. Press 'q' to quit after 2 captures.")
    images = []
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Webcam", frame)
        key = cv2.waitKey(1)

        if key == ord('c'):
            images.append(frame.copy())
            count += 1
            print(f"Captured Image {count}")
            if count == 2:
                break

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(images) == 2:
        compare_images(images[0], images[1])
    else:
        print("Not enough images captured.")

def load_images_from_disk():
    path1 = input("Enter path to Image 1: ").strip()
    path2 = input("Enter path to Image 2: ").strip()

    if not os.path.exists(path1) or not os.path.exists(path2):
        print("One or both image paths are invalid.")
        return

    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)

    if img1 is None or img2 is None:
        print("Error loading one or both images.")
        return

    compare_images(img1, img2)

# === MAIN MENU ===
def main_menu():
    print("\n--- Image Comparison Tool ---")
    print("1. Capture 2 images using webcam")
    print("2. Load 2 images from disk")
    print("3. Real-time frame comparison")
    print("4. Exit")

    while True:
        choice = input("Choose an option (1-4): ").strip()

        if choice == '1':
            capture_images_from_webcam()
        elif choice == '2':
            load_images_from_disk()
        elif choice == '3':
            real_time_detection()
        elif choice == '4':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main_menu()
