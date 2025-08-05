import cv2
import numpy as np
import tensorflow as tf
from playsound import playsound
import pygame

# Model Loading
model = tf.keras.models.load_model("model_tf2.h5")

labels = ["yawn", "no_yawn", "Closed", "Open"]

# alarm setup
pygame.mixer.init()
alarm_playing = False

def play_alarm():
    global alarm_playing
    if not alarm_playing:
        alarm_playing = True
        pygame.mixer.music.load("alarm.mp3")
        pygame.mixer.music.play()


# frame processing to show real time video
def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (145, 145))
    frame_normalized = frame_resized / 255.0
    return frame_normalized.reshape(1, 145, 145, 3)

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (145, 145))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_preprocessed = preprocess_input(frame_rgb) 
    return frame_preprocessed.reshape(1, 145, 145, 3)


cap = cv2.VideoCapture(0)

print("Real-time drowsiness detection started. Press 'q' to exit.")
sleep_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    processed = preprocess_frame(frame_rgb)
    prediction = model.predict(processed)
    predicted_class = np.argmax(prediction)
    label = labels[predicted_class]
    confidence = prediction[0][predicted_class]

    cv2.putText(frame, f"{label} ({confidence*100:.2f}%)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Alarm if drowsiness detected
    if label in ["yawn", "Closed"]:
        sleep_counter += 1
        if sleep_counter >= 5:
            print("⚠️ Drowsiness detected! Triggering alarm...")
            play_alarm()
    else:
        sleep_counter = 0

    if not pygame.mixer.music.get_busy():
        alarm_playing = False

    cv2.imshow("Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
