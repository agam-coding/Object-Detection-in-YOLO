import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from skimage.metrics import structural_similarity as ssim
import tempfile

# Load YOLO model
yolo_model = YOLO("yolov8n.pt")

# === Utility Functions ===

def detect_objects_with_yolo(image):
    results = yolo_model(image)[0]
    annotated = image.copy()
    detected = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0].item())
        label = yolo_model.names[cls_id]
        conf = float(box.conf[0])

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        detected.append({
            "label": label,
            "confidence": conf,
            "bbox": (x1, y1, x2, y2),
            "center": ((x1 + x2) // 2, (y1 + y2) // 2)
        })
    return annotated, detected

def compare_images(img1, img2):
    _, det1 = detect_objects_with_yolo(img1)
    annotated2, det2 = detect_objects_with_yolo(img2)

    missing, shifted = [], []

    for obj1 in det1:
        label1, center1 = obj1["label"], obj1["center"]
        matched = False
        for obj2 in det2:
            if label1 == obj2["label"]:
                dist = np.linalg.norm(np.array(center1) - np.array(obj2["center"]))
                if dist < 50:
                    matched = True
                    break
                else:
                    shifted.append((label1, center1, obj2["center"]))
                    matched = True
                    break
        if not matched:
            missing.append(label1)

    for label, old, new in shifted:
        cv2.arrowedLine(annotated2, old, new, (0, 0, 255), 2)
        cv2.putText(annotated2, f"{label} moved", new, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return annotated2, missing, shifted

def calculate_ssim(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    score, diff = ssim(gray1, gray2, full=True)
    diff = (diff * 255).astype("uint8")
    return score, diff

def load_image(uploaded_file):
    image = Image.open(uploaded_file)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

# === Streamlit UI ===

st.set_page_config(page_title="YOLO + Change Detection", layout="wide")
st.title("📸 AI-Powered Change Detection using YOLOv8")

mode = st.radio("Choose Mode:", ["Upload Two Images", "Use Webcam (Beta)"])

if mode == "Upload Two Images":
    col1, col2 = st.columns(2)
    with col1:
        uploaded1 = st.file_uploader("Upload Image 1", type=["jpg", "png", "jpeg"])
    with col2:
        uploaded2 = st.file_uploader("Upload Image 2", type=["jpg", "png", "jpeg"])

    if uploaded1 and uploaded2:
        img1 = load_image(uploaded1)
        img2 = load_image(uploaded2)

        st.subheader("Original Images")
        st.image([uploaded1, uploaded2], caption=["Image 1", "Image 2"], width=300)

        st.subheader("Change Detection Results")
        result_img, missing, shifted = compare_images(img1, img2)
        st.image(result_img, caption="Changes Annotated", channels="BGR", use_column_width=True)

        if missing:
            st.warning("❌ Missing Objects:")
            st.write(missing)
        else:
            st.success("✅ No missing objects detected.")

        if shifted:
            st.warning("🔀 Shifted Objects:")
            for label, old, new in shifted:
                st.write(f"{label} moved from {old} to {new}")
        else:
            st.success("✅ No shifted objects detected.")

        st.subheader("SSIM (Structural Similarity Index)")
        ssim_score, diff_img = calculate_ssim(img1, img2)
        st.metric(label="SSIM Score", value=f"{ssim_score:.4f}")
        st.image(diff_img, caption="SSIM Difference", clamp=True, use_column_width=True)

elif mode == "Use Webcam (Beta)":
    st.info("🔧 Webcam support in Streamlit requires custom JavaScript. For full webcam support, consider using a frontend (React) or `streamlit-webrtc` module.")
    st.warning("Not implemented here for brevity. Let me know if you want help with that too!")

