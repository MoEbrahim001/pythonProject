import face_recognition
import cv2
import os
import glob
import numpy as np
import pyodbc  # Import pyodbc for database connection

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

        # Resize frame for faster speed
        self.frame_resizing = 0.55

    def connect_to_database(self):
        """
        Establish connection to the SQL Server database.
        """
        try:
            connection = pyodbc.connect(
                'DRIVER={SQL Server};'
                'SERVER=DESKTOP-CLQGA5Q\\SQLEXPRESS;'
                'DATABASE=PatientSystemDB;'
                'Trusted_Connection=yes;'
            )
            return connection
        except Exception as e:
            print(f"Database connection error: {e}")
            return None

    def load_encoding_images(self, images_path):
        """
        Load encoding images from path and database
        :param images_path:
        :return:
        """
        # Connect to the database
        connection = self.connect_to_database()
        if connection is None:
            print("Failed to connect to database.")
            return

        # Fetch patient data (FaceImg and Name)
        cursor = connection.cursor()
        cursor.execute("SELECT FaceImg, Name FROM dbo.Patients")  # Fetch FaceImg and Name
        patients = cursor.fetchall()

        if not patients:
            print("No patient data found in the database.")
            return

        print(f"{len(patients)} patients found in the database.")

        # Load Images
        for patient in patients:
            face_img, name = patient  # Extract FaceImg and Name
            image_file = os.path.join(images_path, face_img)  # Use FaceImg as the filename

            if not os.path.exists(image_file):
                print(f"Image file not found: {image_file} for patient {name}")
                continue

            img = cv2.imread(image_file)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            try:
                # Get encoding
                img_encoding = face_recognition.face_encodings(rgb_img)[0]
                self.known_face_encodings.append(img_encoding)
                self.known_face_names.append(name)
                print(f"Loaded encoding for {name}")
            except Exception as e:
                print(f"Error processing {name}: {e}")

        print("Encoding images loaded successfully.")
        connection.close()

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        # Find all the faces and face encodings in the current frame of video
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.5)
            name = "Unknown"

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            face_names.append(name)

        # Convert to numpy array to adjust coordinates with frame resizing quickly
        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int), face_names
