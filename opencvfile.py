import cv2
import os
import numpy as np
import pickle
import threading
import time

# Paths for storing data
ENCODINGS_FILE = 'face_encodings.pkl'
SAVE_FOLDER = 'saved_faces'

# Load face detector and recognizer
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load known face encodings and names (if available)
if os.path.exists(ENCODINGS_FILE):
    with open(ENCODINGS_FILE, 'rb') as f:
        known_face_encodings, known_face_names = pickle.load(f)
        face_recognizer.read('trained_model.yml')
else:
    known_face_encodings = []
    known_face_names = []

# Create save folder if not existing
os.makedirs(SAVE_FOLDER, exist_ok=True)

# Global variables
current_name = None
capturing = False
image_count = 0
target_image_count = 20
captured_images = []
focused_face = None

# FPS calculation variables
fps = 0
frame_count = 0
start_time = time.time()

# Capture from webcam
cap = cv2.VideoCapture(0)
lock = threading.Lock()

def save_encodings():
    with open(ENCODINGS_FILE, 'wb') as f:
        pickle.dump((known_face_encodings, known_face_names), f)
    face_recognizer.write('trained_model.yml')

def generate_person_id():
    person_ids = [
        int(name.split('')[-1]) for name in known_face_names if name.startswith("Person") and name.split('_')[-1].isdigit()
    ]
    next_id = max(person_ids) + 1 if person_ids else 1
    return f"Person_{next_id:02d}"

def create_person_folder(person_id):
    person_folder = os.path.join(SAVE_FOLDER, person_id)
    os.makedirs(person_folder, exist_ok=True)
    return person_folder

def capture_face(face_img, face_gray, person_id):
    global capturing, image_count, captured_images

    if capturing and image_count < target_image_count:
        captured_images.append(face_gray)
        image_count += 1

        # Save face images in the person's folder
        person_folder = create_person_folder(person_id)
        img_path = os.path.join(person_folder, f"img_{image_count:02d}.jpg")
        cv2.imwrite(img_path, face_img)

        if image_count == target_image_count:
            print(f"[INFO] Captured {target_image_count} images for '{person_id}'.")

            # Train recognizer with the new face
            labels = [len(known_face_names)] * len(captured_images)
            face_recognizer.update(captured_images, np.array(labels))

            # Save encoding and name
            known_face_encodings.append(face_gray)
            known_face_names.append(person_id)

            # Save to file
            save_encodings()

            print(f"[INFO] Face for '{person_id}' saved and model trained.")
            capturing = False
            image_count = 0
            captured_images = []

def recognize_face(face_gray):
    try:
        label, confidence = face_recognizer.predict(face_gray)
        if confidence < 70:
            return known_face_names[label], confidence
        else:
            return "Unknown", confidence
    except:
        return "Unknown", 100

def process_frame(frame):
    global capturing, current_name, focused_face, frame_count, start_time, fps

    # ✅ Initialize name and confidence to avoid UnboundLocalError
    name = "No face"
    confidence = 0

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        # Focus on the FIRST face and blur others
        focused_face = faces[0]
        (fx, fy, fw, fh) = focused_face
        face_gray = gray[fy:fy + fh, fx:fx + fw]
        face_img = frame[fy:fy + fh, fx:fx + fw]

        # Attempt recognition
        name, confidence = recognize_face(face_gray)

        if name != "Unknown":
            color = (0, 255, 0)  # Green for known face
            if not capturing:
                print(f"[INFO] Recognized as '{name}'")
                # Save face image in existing folder
                capture_face(face_img, face_gray, name)
        else:
            color = (0, 0, 255)  # Red for unknown face
            if not capturing:
                # Auto-assign new ID and create folder only for new faces
                new_person = generate_person_id()
                create_person_folder(new_person)
                print(f"[INFO] New face detected. Creating ID: {new_person}")
                capturing = True
                current_name = new_person

        # Draw box and label
        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), color, 2)
        cv2.putText(frame, f"{name} ({int(confidence)}%)", (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Save face if unknown
        if capturing:
            capture_face(face_img, face_gray, current_name)

        # BLUR OTHER FACES
        for (x, y, w, h) in faces:
            if (x, y, w, h) != tuple(focused_face):
                frame[y:y+h, x:x+w] = cv2.GaussianBlur(frame[y:y+h, x:x+w], (51, 51), 0)

    # === FPS Calculation ===
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()

    # Display FPS and recognition rate
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    if name != "Unknown" and name != "No face":
        cv2.putText(frame, f"Confidence: {int(confidence)}%", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return frame

def video_loop():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        with lock:
            processed_frame = process_frame(frame)

        cv2.imshow('Face Recognition', processed_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

def main():
    print("\n🚀 Face Recognition System Started...")
    print("Press 'q' to quit.\n")

    thread = threading.Thread(target=video_loop)
    thread.start()
    thread.join()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()