import os
import logging
import threading
import pickle
from uuid import uuid4

import cv2
import face_recognition
import pyodbc
from flask import Flask, jsonify, request
from sklearn.neighbors import NearestNeighbors
from werkzeug.utils import secure_filename


# =========================================================
# Logging
# =========================================================
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# =========================================================
# Flask
# =========================================================
app = Flask(__name__)


# =========================================================
# Configuration
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FOLDER = os.getenv(
    "DATA_FOLDER",
    os.path.join(BASE_DIR, "data"),
)

ENCODING_FOLDER = os.getenv(
    "ENCODING_FOLDER",
    os.path.join(DATA_FOLDER, "encodings"),
)

UPLOAD_FOLDER = os.getenv(
    "UPLOAD_FOLDER",
    os.path.join(DATA_FOLDER, "uploads"),
)

# This should point to the PUBLIC .NET images endpoint.
# Local example: https://localhost:7183/images
# Production example: https://your-dotnet-api.example.com/images
PUBLIC_IMAGE_BASE_URL = os.getenv(
    "PUBLIC_IMAGE_BASE_URL",
    "https://localhost:7183/images",
).rstrip("/")

DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING")

# Local Windows fallback only.
LOCAL_DB_DRIVER = os.getenv("LOCAL_DB_DRIVER", "{SQL Server}")
LOCAL_DB_SERVER = os.getenv(
    "LOCAL_DB_SERVER",
    r"DESKTOP-CLQGA5Q\SQLEXPRESS",
)
LOCAL_DB_NAME = os.getenv(
    "LOCAL_DB_NAME",
    "PatientSystemDB",
)

MATCH_TOLERANCE = float(
    os.getenv("MATCH_TOLERANCE", "0.5")
)

PORT = int(os.getenv("PORT", "5000"))

os.makedirs(ENCODING_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# =========================================================
# Helpers
# =========================================================
def get_filename_from_path(value):
    if not value:
        return None

    normalized = str(value).replace("\\", "/")
    return normalized.rstrip("/").split("/")[-1]


def build_face_image_url(face_img):
    if not face_img:
        return None

    face_img_text = str(face_img).strip()

    if face_img_text.startswith(("http://", "https://")):
        return face_img_text

    filename = get_filename_from_path(face_img_text)
    if not filename:
        return None

    return f"{PUBLIC_IMAGE_BASE_URL}/{filename}"


def allowed_image_filename(filename):
    if not filename or "." not in filename:
        return False

    extension = filename.rsplit(".", 1)[1].lower()
    return extension in {"jpg", "jpeg", "png", "webp", "bmp"}


# =========================================================
# Face Recognition
# =========================================================
class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_patient_ids = []
        self.patient_data = {}
        self.knn = None
        self.lock = threading.Lock()

    def connect_to_database(self):
        try:
            if DB_CONNECTION_STRING:
                return pyodbc.connect(
                    DB_CONNECTION_STRING,
                    timeout=30,
                )

            logger.warning(
                "DB_CONNECTION_STRING is not set. "
                "Using local SQL Server trusted connection."
            )

            local_connection_string = (
                f"DRIVER={LOCAL_DB_DRIVER};"
                f"SERVER={LOCAL_DB_SERVER};"
                f"DATABASE={LOCAL_DB_NAME};"
                "Trusted_Connection=yes;"
                "TrustServerCertificate=yes;"
            )

            return pyodbc.connect(
                local_connection_string,
                timeout=30,
            )

        except Exception as exc:
            logger.exception("Database connection error: %s", exc)
            return None

    def load_encoding_images(self):
        logger.info("Loading face encodings...")

        connection = self.connect_to_database()
        if connection is None:
            return False

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
            patient_ids = []
            metadata = {}

            for patient in patients:
                (
                    patient_id,
                    name,
                    dob,
                    mobile_no,
                    national_no,
                    face_img,
                ) = patient

                encoding_file_path = os.path.join(
                    ENCODING_FOLDER,
                    f"{patient_id}_encoding.dat",
                )

                if not os.path.exists(encoding_file_path):
                    logger.warning(
                        "Encoding file missing for patient %s: %s",
                        patient_id,
                        encoding_file_path,
                    )
                    continue

                try:
                    with open(encoding_file_path, "rb") as encoding_file:
                        encoding = pickle.load(encoding_file)
                except Exception as exc:
                    logger.warning(
                        "Could not load encoding for patient %s: %s",
                        patient_id,
                        exc,
                    )
                    continue

                encodings.append(encoding)
                patient_ids.append(patient_id)
                metadata[patient_id] = {
                    "id": patient_id,
                    "name": name,
                    "dob": dob,
                    "mobileno": mobile_no,
                    "nationalno": national_no,
                    "faceImgUrl": build_face_image_url(face_img),
                }

            with self.lock:
                self.known_face_encodings = encodings
                self.known_patient_ids = patient_ids
                self.patient_data = metadata

                if encodings:
                    self.knn = NearestNeighbors(
                        n_neighbors=1,
                        algorithm="ball_tree",
                    )
                    self.knn.fit(encodings)
                    logger.info("Loaded %s face encodings.", len(encodings))
                else:
                    self.knn = None
                    logger.warning("No face encodings were loaded.")

            return True

        except Exception as exc:
            logger.exception("Error loading encodings: %s", exc)
            return False

        finally:
            connection.close()

    def compare_faces(self, unknown_image, tolerance=MATCH_TOLERANCE):
        try:
            small_image = cv2.resize(unknown_image, (320, 240))
            encodings = face_recognition.face_encodings(small_image)

            if not encodings:
                logger.warning("No face found in submitted image.")
                return None

            unknown_encoding = encodings[0]

            with self.lock:
                if self.knn is None:
                    logger.warning("KNN model is not initialized.")
                    return None

                distances, indices = self.knn.kneighbors(
                    [unknown_encoding],
                    n_neighbors=1,
                )

                distance = float(distances[0][0])
                logger.info("Match distance: %s", distance)

                if distance >= tolerance:
                    return None

                return self.known_patient_ids[indices[0][0]]

        except Exception as exc:
            logger.exception("Error comparing faces: %s", exc)
            return None

    def generate_encoding_file(self, patient_id, face_image_path):
        try:
            if not os.path.exists(face_image_path):
                logger.error("Image not found: %s", face_image_path)
                return None

            face_image = face_recognition.load_image_file(face_image_path)
            face_encodings = face_recognition.face_encodings(face_image)

            if len(face_encodings) != 1:
                logger.error(
                    "Expected exactly one face for patient %s, found %s.",
                    patient_id,
                    len(face_encodings),
                )
                return None

            encoding_file_path = os.path.join(
                ENCODING_FOLDER,
                f"{patient_id}_encoding.dat",
            )

            with open(encoding_file_path, "wb") as encoding_file:
                pickle.dump(face_encodings[0], encoding_file)

            logger.info(
                "Encoding saved for patient %s to %s",
                patient_id,
                encoding_file_path,
            )

            return encoding_file_path

        except Exception as exc:
            logger.exception(
                "Error generating encoding for patient %s: %s",
                patient_id,
                exc,
            )
            return None


facerec = SimpleFacerec()

_initialization_lock = threading.Lock()
_encodings_initialized = False


def ensure_encodings_initialized():
    global _encodings_initialized

    if _encodings_initialized:
        return

    with _initialization_lock:
        if _encodings_initialized:
            return

        _encodings_initialized = facerec.load_encoding_images()


@app.before_request
def initialize_before_request():
    ensure_encodings_initialized()


# =========================================================
# API Routes
# =========================================================
@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "healthy",
            "loadedEncodings": len(facerec.known_face_encodings),
        }
    ), 200


@app.route("/detectAndFind", methods=["POST"])
def detect_and_find():
    uploaded_file = request.files.get("file")

    if uploaded_file is None or uploaded_file.filename == "":
        return jsonify(
            {
                "status": "error",
                "message": "Face image file is required.",
            }
        ), 400

    if not allowed_image_filename(uploaded_file.filename):
        return jsonify(
            {
                "status": "error",
                "message": "Unsupported image type.",
            }
        ), 400

    safe_name = secure_filename(uploaded_file.filename)
    temp_path = os.path.join(
        UPLOAD_FOLDER,
        f"{uuid4().hex}_{safe_name}",
    )

    uploaded_file.save(temp_path)

    try:
        unknown_image = face_recognition.load_image_file(temp_path)
        patient_id = facerec.compare_faces(unknown_image)

        if patient_id is None:
            return jsonify(
                {
                    "status": "success",
                    "isMatch": False,
                    "patientName": "Unknown",
                    "patientData": {},
                }
            ), 200

        patient_data = facerec.patient_data.get(patient_id, {})

        return jsonify(
            {
                "status": "success",
                "isMatch": True,
                "patientName": patient_data.get("name"),
                "patientData": patient_data,
            }
        ), 200

    except Exception as exc:
        logger.exception("Error in /detectAndFind: %s", exc)
        return jsonify(
            {
                "status": "error",
                "message": "Face detection failed.",
                "details": str(exc),
            }
        ), 500

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.route("/generate_encoding", methods=["POST"])
def generate_encoding():
    patient_id = request.form.get("patientId")
    uploaded_file = request.files.get("file")

    if not patient_id:
        return jsonify(
            {
                "status": "error",
                "message": "patientId is required.",
            }
        ), 400

    try:
        patient_id = int(patient_id)
    except (TypeError, ValueError):
        return jsonify(
            {
                "status": "error",
                "message": "patientId must be an integer.",
            }
        ), 400

    if uploaded_file is None or uploaded_file.filename == "":
        return jsonify(
            {
                "status": "error",
                "message": "Face image file is required.",
            }
        ), 400

    if not allowed_image_filename(uploaded_file.filename):
        return jsonify(
            {
                "status": "error",
                "message": "Unsupported image type.",
            }
        ), 400

    safe_name = secure_filename(uploaded_file.filename)
    temp_path = os.path.join(
        UPLOAD_FOLDER,
        f"{uuid4().hex}_{safe_name}",
    )

    uploaded_file.save(temp_path)

    try:
        encoding_file_path = facerec.generate_encoding_file(
            patient_id,
            temp_path,
        )

        if not encoding_file_path:
            return jsonify(
                {
                    "status": "error",
                    "message": "Exactly one clear face is required.",
                }
            ), 422

        return jsonify(
            {
                "status": "success",
                "encodingFile": os.path.basename(encoding_file_path),
            }
        ), 200

    except Exception as exc:
        logger.exception("Error in /generate_encoding: %s", exc)
        return jsonify(
            {
                "status": "error",
                "message": "Failed to generate encoding.",
                "details": str(exc),
            }
        ), 500

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.route("/reload_encodings", methods=["GET"])
def reload_encodings():
    global _encodings_initialized

    success = facerec.load_encoding_images()
    _encodings_initialized = success

    if not success:
        return jsonify(
            {
                "status": "error",
                "message": "Failed to reload encodings.",
            }
        ), 500

    return jsonify(
        {
            "status": "success",
            "message": "Encodings reloaded successfully.",
            "loadedEncodings": len(facerec.known_face_encodings),
        }
    ), 200


# =========================================================
# Local development only
# In production use Gunicorn, for example:
# gunicorn --workers 1 --bind 0.0.0.0:5000 Script:app
# =========================================================
if __name__ == "__main__":
    ensure_encodings_initialized()

    app.run(
        host="0.0.0.0",
        port=PORT,
        debug=False,
        use_reloader=False,
        threaded=True,
    )
