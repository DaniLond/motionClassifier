import cv2
import json
import os
import re
from datetime import datetime


class VideoAnnotator:
    def __init__(self):
        self.activities = {
            '1': 'walk_forward',
            '2': 'walk_back',
            '3': 'turn',
            '4': 'sit',
            '5': 'stand',
            '0': 'idle'
        }
        self.annotations = []

    def annotate_video_automatically(self, video_path):
        """
        Anotación automática basada en el nombre del archivo"
        """
        filename = os.path.basename(video_path)
        activity = self._extract_activity_from_filename(filename)

        if not activity:
            print(f"No se pudo determinar la actividad para {filename}")
            return self.annotate_video_manual(video_path)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        cap.release()

        self.annotations = [{
            'activity': activity,
            'start_frame': 0,
            'start_time': 0.0,
            'end_frame': total_frames - 1,
            'end_time': duration,
            'confidence': 1.0,
            'method': 'automatic_filename'
        }]

        annotation_file = video_path.replace('.mp4', '_annotations.json')
        with open(annotation_file, 'w') as f:
            json.dump(self.annotations, f, indent=2)

        print(f"Anotación automática creada para {filename}: {activity}")
        return self.annotations

    def _extract_activity_from_filename(self, filename):
        """
        Extrae la actividad del nombre del archivo usando patrones comunes
        """
        filename_lower = filename.lower()

        activity_patterns = {
            'walk_forward': ['walk_forward', 'walkforward', 'walk_front', 'forward'],
            'walk_back': ['walk_back', 'walkback', 'walk_backward', 'backward'],
            'turn': ['turn', 'rotate', 'spin'],
            'sit': ['sit', 'sitting', 'sit_down'],
            'stand': ['stand', 'standing', 'stand_up'],
            'idle': ['idle', 'rest', 'static']
        }

        for activity, patterns in activity_patterns.items():
            for pattern in patterns:
                if pattern in filename_lower:
                    return activity

        return None

    def annotate_video_semi_automatic(self, video_path, segments=None):
        """
        Anotación semi-automática: divide el video en segmentos y asigna actividades
        
        Args:
            video_path: Ruta del video
            segments: Lista de tuplas (start_time, end_time, activity)
                     Si es None, divide el video en segmentos iguales
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        cap.release()

        if segments is None:
            activity = self._extract_activity_from_filename(
                os.path.basename(video_path))
            if activity:
                segments = [(0, duration, activity)]
            else:
                segment_duration = 5.0
                segments = []
                current_time = 0
                while current_time < duration:
                    end_time = min(current_time + segment_duration, duration)
                    segments.append((current_time, end_time, 'idle'))
                    current_time = end_time

        self.annotations = []
        for start_time, end_time, activity in segments:
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)

            self.annotations.append({
                'activity': activity,
                'start_frame': start_frame,
                'start_time': start_time,
                'end_frame': end_frame,
                'end_time': end_time,
                'confidence': 0.8,
                'method': 'semi_automatic'
            })

        annotation_file = video_path.replace('.mp4', '_annotations.json')
        with open(annotation_file, 'w') as f:
            json.dump(self.annotations, f, indent=2)

        print(
            f"Anotación semi-automática creada con {len(segments)} segmentos")
        return self.annotations

    def annotate_video_manual(self, video_path):
        """Herramienta para anotar videos manualmente"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print("Teclas de anotación:")
        for key, activity in self.activities.items():
            print(f"  {key}: {activity}")
        print("  'q': Salir y guardar")
        print("  'space': Pausar/Reanudar")

        frame_idx = 0
        current_activity = 'idle'
        paused = False
        self.annotations = []

        while frame_idx < total_frames:
            if not paused:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break

            timestamp = frame_idx / fps
            cv2.putText(frame, f"Frame: {frame_idx}/{total_frames}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Time: {timestamp:.2f}s", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Activity: {current_activity}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('Video Annotator', frame)

            key = cv2.waitKey(30) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
            elif chr(key) in self.activities:
                new_activity = self.activities[chr(key)]
                if new_activity != current_activity:
                    if self.annotations:
                        self.annotations[-1]['end_frame'] = frame_idx
                        self.annotations[-1]['end_time'] = timestamp

                    self.annotations.append({
                        'activity': new_activity,
                        'start_frame': frame_idx,
                        'start_time': timestamp,
                        'end_frame': None,
                        'end_time': None,
                        'method': 'manual'
                    })
                    current_activity = new_activity

            if not paused:
                frame_idx += 1

        if self.annotations and self.annotations[-1]['end_frame'] is None:
            self.annotations[-1]['end_frame'] = frame_idx
            self.annotations[-1]['end_time'] = frame_idx / fps

        cap.release()
        cv2.destroyAllWindows()

        annotation_file = video_path.replace('.mp4', '_annotations.json')
        with open(annotation_file, 'w') as f:
            json.dump(self.annotations, f, indent=2)

        return self.annotations

    def annotate_video(self, video_path, method='auto'):
        """
        Método principal que elige el tipo de anotación
        
        Args:
            video_path: Ruta del video
            method: 'auto', 'semi', 'manual'
        """
        if method == 'auto':
            return self.annotate_video_automatically(video_path)
        elif method == 'semi':
            return self.annotate_video_semi_automatic(video_path)
        else:
            return self.annotate_video_manual(video_path)


if __name__ == "__main__":
    annotator = VideoAnnotator()

    video_path = "data/raw/person_01_rep1_walk_forward.mp4"
    annotations = annotator.annotate_video(video_path, method='auto')

    segments = [
        (0, 3, 'idle'),
        (3, 8, 'walk_forward'),
        (8, 10, 'idle')
    ]
    annotations = annotator.annotate_video_semi_automatic(video_path, segments)
