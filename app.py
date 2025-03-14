import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
from datetime import datetime
from sqlalchemy import create_engine, text
import joblib

app = Flask(__name__)

# Load face detection model (Haar cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Use your existing label dictionary
label_dict = {'abdillahi': 0, 'bashir': 1, 'biihi': 2, 'jiir': 3, 'yusuf': 4}
reverse_label_dict = {v: k for k, v in label_dict.items()}  # Reverse mapping

# Initialize database
DATABASE_URL = "sqlite:///attendance.db"
engine = create_engine(DATABASE_URL)

# Drop the existing table and recreate it
with engine.connect() as conn:
    conn.execute(text("DROP TABLE IF EXISTS attendance"))
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            timestamp TEXT
        )
    """))

detected_name = None  # Store the last detected name globally

def generate_frames():
    """Capture video frames and perform face recognition."""
    global detected_name
    cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60))

        detected_name = None  # Reset detected name before processing frame
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (64, 64))
            face_normalized = face_resized / 255.0
            face_flattened = face_normalized.reshape(1, -1)

            # Predict with SVM
            svm_model = joblib.load('svm_face_recognition.pkl')
            probabilities = svm_model.predict_proba(face_flattened)[0]
            max_prob = max(probabilities)
            prediction = svm_model.predict(face_flattened)[0]

            if max_prob >= 0.5:
                detected_name = reverse_label_dict.get(prediction, "Unknown")
            else:
                detected_name = None

            # Draw rectangle & label
            color = (0, 255, 0) if detected_name else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label = detected_name if detected_name else "Unknown"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Encode frame for streaming
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/attendance_status')
def attendance_status():
    global detected_name
    return jsonify({"name": detected_name if detected_name else "Unknown"})

@app.route('/record_attendance', methods=['POST'])
def record_attendance():
    global detected_name
    if detected_name is None:
        return jsonify({"success": False, "message": "No recognized face detected."})

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with engine.connect() as conn:
        result = conn.execute(text("SELECT * FROM attendance WHERE name = :name"), {"name": detected_name})
        if result.fetchone() is None:
            conn.execute(text("INSERT INTO attendance (name, timestamp) VALUES (:name, :timestamp)"),
                         {"name": detected_name, "timestamp": timestamp})
            conn.commit()
            return jsonify({"success": True, "message": f"Attendance recorded for {detected_name}."})
        else:
            return jsonify({"success": False, "message": "Attendance already recorded for this person."})

@app.route('/attendance')
def attendance():
    with engine.connect() as conn:
        result = conn.execute(text("SELECT * FROM attendance"))
        attendance_records = result.fetchall()
    return render_template('attendance.html', records=attendance_records)

if __name__ == "__main__":
    app.run(debug=True)
