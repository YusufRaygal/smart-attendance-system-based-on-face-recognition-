import os
import cv2
import pickle
import numpy as np

# Try importing sklearn and handle errors
try:
    from sklearn.neighbors import KNeighborsClassifier
except ImportError:
    print("Error: scikit-learn is not installed. Run `pip install scikit-learn` and try again.")
    exit()

# Load the Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

video = cv2.VideoCapture(0)  # Use camera index 0

# Ensure required files exist
if not os.path.exists("data/names.pkl") or not os.path.exists("data/faces_data.pkl"):
    print("Error: Pickle files not found! Ensure `names.pkl` and `faces_data.pkl` are in the `data/` folder.")
    exit()

# Load preprocessed face data
try:
    with open('data/names.pkl', 'rb') as f:
        Labels = pickle.load(f)
    with open('data/faces_data.pkl', 'rb') as f:
        faces = pickle.load(f)

    # Ensure faces data is a NumPy array
    faces = np.array(faces, dtype=np.float32)

except (pickle.UnpicklingError, ValueError):
    print("Error: Failed to load data. The pickle files might be corrupted.")
    exit()

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(faces, Labels)

while True:
    ret, frame = video.read()
    if not ret:
        print("Error: Unable to capture video.")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    detected_faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in detected_faces:
        crop_img = frame[y:y+h, x:x+w, :]

        try:
            resized_img = cv2.resize(crop_img, (50, 50)).reshape(1, -1)
            output = knn.predict(resized_img)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
            cv2.putText(frame, str(output[0]), (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        except Exception as e:
            print(f"Error during prediction: {e}")

    cv2.imshow('Face Recognition', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
