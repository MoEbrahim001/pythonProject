import os
import logging
import threading
import pyodbc
import cv2
import face_recognition
import pickle
from sklearn.neighbors import NearestNeighbors
from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename
from flask_cors import CORS


# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------- Flask ----------------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:4200"}})


# ---------------- Configuration ----------------
IMAGE_FOLDER = r"D:\Face-Recognition\PatientSystem.WebApi\images"
ENCODING_FOLDER = r"D:\Face-Recognition\PatientSystem.WebApi\EncodingFile"
UPLOAD_FOLDER = r"D:\Face-Recognition\PatientSystem.WebApi\uploads"
BASE_URL = "http://localhost:5000"

DATABASE_CONFIG = {
    "DRIVER": "{SQL Server}",
    "SERVER": r"DESKTOP-CLQGA5Q\SQLEXPRESS",
    "DATABASE": "PatientSystemDB",
    "Trusted_Connection": "yes",
}


# Ensure directories exist
os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(ENCODING_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.patient_data = {}
        self.knn = None
        self.lock = threading.Lock()

    def connect_to_database(self):
        """Establish a connection to SQL Server."""
        try:
            return pyodbc.connect(**DATABASE_CONFIG)
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            return None

    def _clear_loaded_encodings(self):
        """Clear the currently loaded in-memory recognition data."""
        with self.lock:
            self.known_face_encodings = []
            self.known_face_names = []
            self.patient_data = {}
            self.knn = None

    def load_encoding_images(self):
        """
        Load only patients that are enrolled in face recognition.

        A patient is considered enrolled when EncodingFile in SQL is not NULL/empty.
        Missing .dat files are skipped silently so demo/internet-image rows do not
        flood the console with errors.
        """
        logger.info("Loading face encodings...")

        connection = self.connect_to_database()
        if connection is None:
            return

        try:
            cursor = connection.cursor()

            cursor.execute(
                """
                SELECT
                    Id,
                    Name,
                    Dob,
                    Mobileno,
                    Nationalno,
                    FaceImg
                FROM dbo.Patients
                WHERE EncodingFile IS NOT NULL
                  AND LTRIM(RTRIM(EncodingFile)) <> ''
                """
            )

            patients = cursor.fetchall()

            encodings = []
            names = []
            patient_metadata = {}

            for patient in patients:
                patient_id, name, dob, mobile_no, national_no, face_img = patient

                # We intentionally use the configured EncodingFile folder and patient ID.
                encoding_file_path = os.path.join(
                    ENCODING_FOLDER,
                    f"{patient_id}_encoding.dat",
                )

                # Do not print an ERROR for old/demo rows whose .dat file does not exist.
                if not os.path.exists(encoding_file_path):
                    continue

                try:
                    with open(encoding_file_path, "rb") as encoding_file:
                        encoding = pickle.load(encoding_file)
                except Exception as e:
                    logger.warning(
                        f"Could not load encoding for patient ID {patient_id}: {e}"
                    )
                    continue

                encodings.append(encoding)
                names.append(name)

                face_img_name = os.path.basename(face_img) if face_img else None
                face_img_url = (
                    f"{BASE_URL}/images/{face_img_name}"
                    if face_img_name
                    else None
                )

                # Kept keyed by name because compare_faces returns the stored name.
                patient_metadata[name] = {
                    "mobileno": mobile_no,
                    "nationalno": national_no,
                    "id": patient_id,
                    "dob": dob,
                    "faceImgUrl": face_img_url,
                }

            with self.lock:
                self.known_face_encodings = encodings
                self.known_face_names = names
                self.patient_data = patient_metadata

                if encodings:
                    self.knn = NearestNeighbors(
                        n_neighbors=1,
                        algorithm="ball_tree",
                    )
                    self.knn.fit(encodings)

                    logger.info(
                        f"Loaded {len(self.known_face_encodings)} face encodings."
                    )
                else:
                    self.knn = None
                    logger.warning("No face encodings were loaded.")

        except Exception as e:
            logger.exception(f"Error loading encodings: {e}")

        finally:
            connection.close()

    def compare_faces(self, unknown_image, tolerance=0.5):
        """Compare an unknown face image with the loaded encodings."""
        try:
            small_image = cv2.resize(unknown_image, (320, 240))
            encodings = face_recognition.face_encodings(small_image)

            if not encodings:
                logger.warning("No face found in the image.")
                return None

            unknown_encoding = encodings[0]

            with self.lock:
                if self.knn is None:
                    logger.error("KNN model is not initialized.")
                    return None

                distances, indices = self.knn.kneighbors(
                    [unknown_encoding],
                    n_neighbors=1,
                )

                distance = float(distances[0][0])
                logger.info(f"Match distance: {distance}")

                if distance >= tolerance:
                    logger.info("No match found within the tolerance threshold.")
                    return None

                return self.known_face_names[indices[0][0]]

        except Exception as e:
            logger.exception(f"Error comparing faces: {e}")
            return None

    def generate_encoding_file(self, patient_id, face_image_path):
        """Generate and save an encoding file for one patient image."""
        import time

        start_time = time.time()

        try:
            logger.info(f"[Patient {patient_id}] Starting encoding")
            logger.info(f"[Patient {patient_id}] Image path: {face_image_path}")

            if not os.path.exists(face_image_path):
                logger.error(
                    f"[Patient {patient_id}] Image not found: {face_image_path}"
                )
                return None

            load_start = time.time()
            face_image = face_recognition.load_image_file(face_image_path)

            logger.info(
                f"[Patient {patient_id}] Image loaded in "
                f"{time.time() - load_start:.2f} seconds"
            )

            encoding_start = time.time()
            logger.info(f"[Patient {patient_id}] Generating face encoding...")

            face_encodings = face_recognition.face_encodings(face_image)

            logger.info(
                f"[Patient {patient_id}] Face encoding took "
                f"{time.time() - encoding_start:.2f} seconds"
            )

            if not face_encodings:
                logger.error(f"No face found for patient {patient_id}")
                return None

            encoding_file_path = os.path.join(
                ENCODING_FOLDER,
                f"{patient_id}_encoding.dat",
            )

            with open(encoding_file_path, "wb") as encoding_file:
                pickle.dump(face_encodings[0], encoding_file)

            logger.info(
                f"[Patient {patient_id}] Encoding saved to {encoding_file_path}"
            )
            logger.info(
                f"[Patient {patient_id}] TOTAL TIME = "
                f"{time.time() - start_time:.2f} seconds"
            )

            # Do not append directly to KNN here.
            # SQL is updated by the caller, then /reload_encodings reloads cleanly.
            return encoding_file_path

        except Exception as e:
            logger.exception(
                f"Error generating encoding for patient {patient_id}: {e}"
            )
            return None


# ---------------- Flask routes ----------------

@app.route("/images/<path:filename>")
def serve_images(filename):
    return send_from_directory(IMAGE_FOLDER, filename)


@app.route("/detectAndFind", methods=["POST"])
def detect_and_find():
    if "file" not in request.files:
        return jsonify(
            {"status": "error", "message": "No file part"}
        ), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify(
            {"status": "error", "message": "No selected file"}
        ), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    try:
        unknown_image = face_recognition.load_image_file(file_path)
        matched_name = facerec.compare_faces(
            unknown_image,
            tolerance=0.5,
        )

        if matched_name is not None:
            patient_data = facerec.patient_data.get(matched_name)

            return jsonify(
                {
                    "status": "success",
                    "isMatch": True,
                    "patientName": matched_name,
                    "patientData": patient_data,
                }
            )

        return jsonify(
            {
                "status": "success",
                "isMatch": False,
                "patientName": "Unknown",
                "patientData": {},
            }
        )

    except Exception as e:
        logger.exception(f"Error in /detectAndFind endpoint: {e}")

        return jsonify(
            {
                "status": "error",
                "message": "An error occurred.",
                "details": str(e),
            }
        ), 500

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@app.route("/add_encoding_to_database", methods=["POST"])
def add_encoding_to_database():
    try:
        data = request.get_json(silent=True) or {}
        logger.info(f"Received data: {data}")

        patient_id = data.get("patientId")
        face_image_path = data.get("faceImage")

        if not patient_id or not face_image_path:
            return jsonify(
                {
                    "status": "error",
                    "message": "Missing patientId or faceImage path.",
                }
            ), 400

        if not os.path.exists(face_image_path):
            return jsonify(
                {
                    "status": "error",
                    "message": f"Face image not found at {face_image_path}",
                }
            ), 400

        encoding_file_path = facerec.generate_encoding_file(
            patient_id,
            face_image_path,
        )

        if not encoding_file_path:
            return jsonify(
                {
                    "status": "error",
                    "message": "Failed to generate encoding file.",
                }
            ), 500

        connection = facerec.connect_to_database()

        if connection is None:
            return jsonify(
                {
                    "status": "error",
                    "message": "Database connection failed.",
                }
            ), 500

        try:
            cursor = connection.cursor()

            cursor.execute(
                """
                UPDATE dbo.Patients
                SET EncodingFile = ?
                WHERE Id = ?
                """,
                encoding_file_path,
                patient_id,
            )

            connection.commit()

            logger.info(
                f"Database updated for patientId {patient_id} "
                f"with encoding file path: {encoding_file_path}"
            )

        except Exception as e:
            logger.exception(f"Error updating database: {e}")

            return jsonify(
                {
                    "status": "error",
                    "message": "Failed to update database.",
                    "details": str(e),
                }
            ), 500

        finally:
            connection.close()

        # Refresh the in-memory model after SQL has been updated.
        facerec.load_encoding_images()

        return jsonify(
            {
                "status": "success",
                "message": (
                    "Encoding file generated, database updated, "
                    "and encodings reloaded successfully."
                ),
                "encodingFilePath": encoding_file_path,
            }
        ), 200

    except Exception as e:
        logger.exception(f"Error in /add_encoding_to_database endpoint: {e}")

        return jsonify(
            {
                "status": "error",
                "message": "An error occurred.",
                "details": str(e),
            }
        ), 500


@app.route("/reload_encodings", methods=["GET"])
def reload_encodings():
    try:
        facerec.load_encoding_images()

        return jsonify(
            {
                "status": "success",
                "message": "Encodings reloaded successfully.",
            }
        ), 200

    except Exception as e:
        logger.exception(f"Error during reload: {e}")

        return jsonify(
            {
                "status": "error",
                "message": "Failed to reload encodings.",
                "details": str(e),
            }
        ), 500


@app.route("/generate_encoding", methods=["POST"])
def generate_encoding():
    data = request.get_json(silent=True) or {}

    patient_id = data.get("patientId")
    face_image_path = data.get("faceImage")

    if not patient_id or not face_image_path:
        return jsonify(
            {
                "status": "error",
                "message": "Missing patientId or faceImage path.",
            }
        ), 400

    if not os.path.exists(face_image_path):
        return jsonify(
            {
                "status": "error",
                "message": f"Face image not found at {face_image_path}",
            }
        ), 400

    encoding_file_path = facerec.generate_encoding_file(
        patient_id,
        face_image_path,
    )

    if encoding_file_path:
        return jsonify(
            {
                "status": "success",
                "encodingFilePath": encoding_file_path,
            }
        ), 200

    return jsonify(
        {
            "status": "error",
            "message": "Failed to generate encoding.",
        }
    ), 500


# ---------------- Startup ----------------

facerec = SimpleFacerec()


def load_encodings_on_restart():
    facerec.load_encoding_images()


if __name__ == "__main__":
    logger.info("Loading existing face encodings...")

    load_encodings_on_restart()

    logger.info("Finished loading face encodings.")
    logger.info("Starting Flask server...")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        use_reloader=False,
        threaded=True,
    )
