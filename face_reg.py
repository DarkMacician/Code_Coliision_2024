import cv2
import face_recognition
import time

# start_time = time.time()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
captured = False
face_img = None
last_processed_time = time.time()
processing_interval = 0.25
scale_factor = 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from camera.")
        break

    current_time = time.time()
    if current_time - last_processed_time >= processing_interval:
        small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (int(x / scale_factor), int(y / scale_factor)),
                          (int((x + w) / scale_factor), int((y + h) / scale_factor)), (255, 0, 0), 2)

            if not captured:
                face_img = frame[int(y / scale_factor):int((y + h) / scale_factor),
                           int(x / scale_factor):int((x + w) / scale_factor)]
                captured = True
                print("Capture successful!")
                break

        last_processed_time = current_time

    if captured:
        break

cap.release()
cv2.destroyAllWindows()

if face_img is not None:
    rgb_face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    captured_encodings = face_recognition.face_encodings(rgb_face_img)

    if len(captured_encodings) == 0:
        print("No face found in the captured image.")
        exit()

    captured_encoding = captured_encodings[0]

    known_image = face_recognition.load_image_file("D:\Aptos\captured_face.png")
    known_encodings = face_recognition.face_encodings(known_image)
    if len(known_encodings) == 0:
        print("No face found in the known image.")
        exit()

    known_encoding = known_encodings[0]

    matches = face_recognition.compare_faces([known_encoding], captured_encoding)
    print("Match result:", matches)
else:
    print("No face image captured.")

# end_time = time.time()
#
# elapsed_time = end_time - start_time
# print(f"Elapsed time: {elapsed_time:.2f} seconds")