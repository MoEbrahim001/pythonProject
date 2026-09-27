import os
import pickle
import pyodbc
import face_recognition


IMAGE_FOLDER = r"D:\Face-Recognition\PatientSystem.WebApi\images"
ENCODING_FOLDER = r"D:\Face-Recognition\PatientSystem.WebApi\EncodingFile"


DATABASE_CONFIG = {
    'DRIVER': '{SQL Server}',
    'SERVER': r'DESKTOP-CLQGA5Q\SQLEXPRESS',
    'DATABASE': 'PatientSystemDB',
    'Trusted_Connection': 'yes'
}


def main():

    os.makedirs(
        ENCODING_FOLDER,
        exist_ok=True
    )

    connection = pyodbc.connect(
        **DATABASE_CONFIG
    )

    cursor = connection.cursor()

   cursor.execute("""
    SELECT Id, Name, Dob, Mobileno, Nationalno, FaceImg
    FROM dbo.Patients
    WHERE EncodingFile IS NOT NULL
      AND LTRIM(RTRIM(EncodingFile)) <> ''
""")
    patients = cursor.fetchall()

    total = len(patients)

    success = 0
    missing_image = 0
    no_face = 0
    multiple_faces = 0
    errors = 0

    print(f"Patients to process: {total}")
    print("=" * 60)


    for index, patient in enumerate(
        patients,
        start=1
    ):

        patient_id = patient.Id
        patient_name = patient.Name
        face_img = patient.FaceImg

        image_file_name = os.path.basename(
            str(face_img).replace("\\", "/")
        )

        image_path = os.path.join(
            IMAGE_FOLDER,
            image_file_name
        )

        encoding_path = os.path.join(
            ENCODING_FOLDER,
            f"{patient_id}_encoding.dat"
        )

        print(
            f"[{index}/{total}] "
            f"{patient_id} - {patient_name}"
        )


        if not os.path.exists(image_path):

            print(
                f"   IMAGE NOT FOUND: {image_path}"
            )

            missing_image += 1
            continue


        try:

            image = face_recognition.load_image_file(
                image_path
            )

            encodings = face_recognition.face_encodings(
                image
            )


            if len(encodings) == 0:

                print(
                    "   NO FACE DETECTED"
                )

                no_face += 1
                continue


            if len(encodings) > 1:

                print(
                    f"   MULTIPLE FACES: {len(encodings)}"
                )

                multiple_faces += 1
                continue


            with open(
                encoding_path,
                "wb"
            ) as file:

                pickle.dump(
                    encodings[0],
                    file
                )


            cursor.execute(
                """
                UPDATE dbo.Patients
                SET EncodingFile = ?
                WHERE Id = ?
                """,
                encoding_path,
                patient_id
            )


            success += 1

            print(
                f"   SUCCESS: {encoding_path}"
            )


            if success % 50 == 0:

                connection.commit()

                print(
                    f"--- committed {success} encodings ---"
                )


        except Exception as e:

            errors += 1

            print(
                f"   ERROR: {e}"
            )


    connection.commit()
    connection.close()


    print()
    print("=" * 60)
    print("REBUILD FINISHED")
    print("=" * 60)

    print(f"Total          : {total}")
    print(f"Success        : {success}")
    print(f"Missing images : {missing_image}")
    print(f"No face        : {no_face}")
    print(f"Multiple faces : {multiple_faces}")
    print(f"Errors         : {errors}")


if __name__ == "__main__":
    main()