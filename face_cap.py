import time
import cv2

# start_time = time.time()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
captured = False
processing_interval = 0.1
last_processed_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from camera.")
        break

    current_time = time.time()

    if current_time - last_processed_time >= processing_interval:
        small_frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            x, y, w, h = int(x * (frame.shape[1] / small_frame.shape[1])), int(
                y * (frame.shape[0] / small_frame.shape[0])), int(w * (frame.shape[1] / small_frame.shape[1])), int(
                h * (frame.shape[0] / small_frame.shape[0]))

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            if not captured:
                face_img = frame[y:y + h, x:x + w]
                cv2.imwrite('captured_face.png', face_img)
                captured = True
                print("Capture successful!")
                break

        last_processed_time = current_time

    if captured:
        break

    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# end_time = time.time()
#
# elapsed_time = end_time - start_time
# print(f"Elapsed time: {elapsed_time:.2f} seconds")
