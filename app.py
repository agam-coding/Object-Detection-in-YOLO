from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import cv2
import numpy as np
import os
import threading
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

# Load YOLOv8 model
model = YOLO("best.pt")  # Replace with your trained YOLO model path

# Global variables for camera thread
camera = None
capture = False
frame_lock = threading.Lock()
output_frame = None


def detect_objects(frame):
    results = model(frame)
    annotated_frame = results[0].plot()
    return annotated_frame, results[0].boxes.data.cpu().numpy().tolist()


# Real-Time Video Feed with YOLO Detection
def generate_frames():
    global camera, output_frame
    while True:
        success, frame = camera.read()
        if not success:
            break

        annotated_frame, _ = detect_objects(frame)
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        
@app.route('/')
def home():
    return 'Flask backend is running!'

@app.route('/favicon.ico')
def favicon():
    return '', 204  # No Content


@app.route("/start-camera")
def start_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
    return jsonify({"status": "Camera started"})


@app.route("/stop-camera")
def stop_camera():
    global camera
    if camera:
        camera.release()
        camera = None
    return jsonify({"status": "Camera stopped"})


@app.route("/video-feed")
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Capture and detect single frame
@app.route("/capture-image", methods=['GET'])
def capture_image():
    global camera
    if camera is None:
        return jsonify({"error": "Camera not started"}), 400

    success, frame = camera.read()
    if not success:
        return jsonify({"error": "Failed to capture frame"}), 500

    annotated_frame, detections = detect_objects(frame)
    _, buffer = cv2.imencode('.jpg', annotated_frame)
    img_bytes = buffer.tobytes()

    return Response(img_bytes, mimetype='image/jpeg')


# Upload and detect on imported image
@app.route("/upload-image", methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    annotated_img, detections = detect_objects(img)
    _, buffer = cv2.imencode('.jpg', annotated_img)
    img_bytes = buffer.tobytes()

    return Response(img_bytes, mimetype='image/jpeg')


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)