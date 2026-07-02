import tensorflow as tf
import cv2
import numpy as np

model = tf.keras.models.load_model("models/best_emotion_model.keras")
# model = tf.keras.models.load_model("models/emotion_recognition_model.keras")

emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

print("Webcam started. Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Process each face
    for (x, y, w, h) in faces:
        # Extract face region
        face_roi = gray[y:y+h, x:x+w]
        
        face_roi = cv2.equalizeHist(face_roi)

        face_roi = cv2.GaussianBlur(face_roi, (3,3), 0)

        # Resize to model input size (48x48)
        face_roi_resized = cv2.resize(face_roi, (48, 48))
        
        # Normalize the image
        face_roi_normalized = face_roi_resized / 255.0
        
        # Reshape for model prediction
        face_input = np.expand_dims(np.expand_dims(face_roi_normalized, axis=-1), axis=0)
        
        # Make prediction
        prediction = model.predict(face_input, verbose=0)
        emotion_idx = np.argmax(prediction)
        emotion = emotion_labels[emotion_idx]
        confidence = prediction[0][emotion_idx]
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        label_text = f"{emotion} ({confidence:.2f})"
        cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    cv2.imshow("Emotion Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()