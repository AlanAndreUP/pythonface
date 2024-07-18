import cv2
import face_recognition
import time
import requests
import numpy as np
from datetime import datetime

# Constantes
TIME_LIMIT = 3 * 60
FACE_MATCH_THRESHOLD = 0.6
EXIT_TIME_THRESHOLD = 10
API_ENDPOINT = "https://api.rutasegura.xyz/pasajeros"

def send_passenger_count(num_passengers):
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    payload = {
        "fecha": current_date,
        "cantidad": num_passengers
    }

    try:
        response = requests.post(API_ENDPOINT, json=payload)
        response.raise_for_status()
        print(f"Datos enviados exitosamente: {response.status_code} {payload}")
    except requests.exceptions.RequestException as e:
        print(f"Error al enviar datos: {e}")

def main():
    video_capture = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

    if not video_capture.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return

    face_encodings_last_seen = {}

    while True:
        ret, frame = video_capture.read()

        if not ret:
            print("No se pudo capturar el frame.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        current_time = time.time()

        for encoding in list(face_encodings_last_seen.keys()):
            last_seen_time = face_encodings_last_seen[encoding][1]
            if current_time - last_seen_time > TIME_LIMIT or current_time - last_seen_time > EXIT_TIME_THRESHOLD:
                del face_encodings_last_seen[encoding]

        for encoding in face_encodings:
            encoding_tuple = tuple(encoding)
            match_found = False

            for known_encoding, (known_face_encoding, _) in face_encodings_last_seen.items():
                distance = face_recognition.face_distance([known_face_encoding], encoding)
                if distance[0] < FACE_MATCH_THRESHOLD:
                    face_encodings_last_seen[known_encoding] = (known_face_encoding, current_time)
                    match_found = True
                    break

            if not match_found:
                face_encodings_last_seen[encoding_tuple] = (encoding, current_time)

        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        cv2.putText(frame, f'Passengers: {len(face_encodings_last_seen)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Enviar el número de pasajeros detectados cada 10 segundos
        if int(current_time) % 10 == 0:
            send_passenger_count(len(face_encodings_last_seen))

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
