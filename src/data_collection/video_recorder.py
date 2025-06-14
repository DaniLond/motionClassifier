import cv2
import os
from datetime import datetime


class VideoRecorder:
    def __init__(self, output_dir="data/raw"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def record_activity_session(self, person_id, activity_type):
        """
        Graba una sesión de actividad específica
        Activities: walk_forward, walk_back, turn, sit, stand
        """
        cap = cv2.VideoCapture(0)

        # Configuración del video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{person_id}_{activity_type}_{timestamp}.mp4"
        filepath = os.path.join(self.output_dir, filename)

        fps = 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = cv2.VideoWriter(filepath, fourcc, fps, (width, height))

        print(f"Grabando {activity_type} para persona {person_id}")
        print("Presiona 'q' para detener la grabación")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            cv2.putText(frame, f"Persona: {person_id}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Actividad: {activity_type}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Grabación', frame)
            out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        return filepath
