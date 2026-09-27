import cv2
from main import SimpleFacerec

# Encode faces
sfr = SimpleFacerec()
sfr.load_encoding_images(r"C:\Users\dell\source\repos\PatientSystem\images")

# Load Camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()



while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        print("Error: Failed to capture frame.")
        continue

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # Detect Faces
    try:
        face_locations, face_names = sfr.detect_known_faces(small_frame)
        face_locations = [(top*2, right*2, bottom*2, left*2) for top, right, bottom, left in face_locations]
        for face_loc, name in zip(face_locations, face_names):
            y1, x2, y2, x1 = face_loc
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)
    except Exception as e:
        print(f"Detection error: {e}")

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()
