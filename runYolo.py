from flask import Flask, Response
import cv2
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO("best.pt")  # Load YOLO model
camera_device = "/dev/video2"
cap = cv2.VideoCapture(camera_device)

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)  # Run detection
        annotated_frame = results[0].plot()

        _, buffer = cv2.imencode(".jpg", annotated_frame)
        frame_bytes = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               frame_bytes + b"\r\n")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

