import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import json
import os

class PoseExtractor:
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def extract_landmarks_from_video(self, video_path):
        """Extrae landmarks de pose de un video completo"""
        cap = cv2.VideoCapture(video_path)
        landmarks_data = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convertir BGR a RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)

            if results.pose_landmarks:
                # Extraer coordenadas de landmarks clave
                landmarks = self._extract_key_landmarks(results.pose_landmarks)
                landmarks['frame'] = frame_count
                landmarks['timestamp'] = frame_count / \
                    cap.get(cv2.CAP_PROP_FPS)
                landmarks_data.append(landmarks)

            frame_count += 1

        cap.release()
        return pd.DataFrame(landmarks_data)

    def _extract_key_landmarks(self, pose_landmarks):
        """Extrae landmarks clave y calcula características derivadas"""
        landmarks = {}

        key_points = {
            'nose': 0,
            'left_shoulder': 11, 'right_shoulder': 12,
            'left_hip': 23, 'right_hip': 24,
            'left_knee': 25, 'right_knee': 26,
            'left_ankle': 27, 'right_ankle': 28,
            'left_wrist': 15, 'right_wrist': 16
        }

        for name, idx in key_points.items():
            landmark = pose_landmarks.landmark[idx]
            landmarks[f'{name}_x'] = landmark.x
            landmarks[f'{name}_y'] = landmark.y
            landmarks[f'{name}_z'] = landmark.z
            landmarks[f'{name}_visibility'] = landmark.visibility

        landmarks.update(self._calculate_derived_features(landmarks))

        return landmarks

    def _calculate_derived_features(self, landmarks):
        """Calcula características derivadas importantes"""
        features = {}

        # 1. Inclinación lateral del tronco
        left_shoulder_y = landmarks['left_shoulder_y']
        right_shoulder_y = landmarks['right_shoulder_y']
        features['trunk_lateral_inclination'] = abs(
            left_shoulder_y - right_shoulder_y)

        # 2. Centro de masa aproximado
        features['center_mass_x'] = (
            landmarks['left_hip_x'] + landmarks['right_hip_x']) / 2
        features['center_mass_y'] = (
            landmarks['left_hip_y'] + landmarks['right_hip_y']) / 2

        # 3. Ángulos articulares aproximados
        # Ángulo de rodilla izquierda (hip-knee-ankle)
        features['left_knee_angle'] = self._calculate_angle(
            (landmarks['left_hip_x'], landmarks['left_hip_y']),
            (landmarks['left_knee_x'], landmarks['left_knee_y']),
            (landmarks['left_ankle_x'], landmarks['left_ankle_y'])
        )

        # Ángulo de rodilla derecha
        features['right_knee_angle'] = self._calculate_angle(
            (landmarks['right_hip_x'], landmarks['right_hip_y']),
            (landmarks['right_knee_x'], landmarks['right_knee_y']),
            (landmarks['right_ankle_x'], landmarks['right_ankle_y'])
        )

        # 4. Altura aproximada de la persona
        features['person_height'] = abs(landmarks['nose_y'] -
                                        min(landmarks['left_ankle_y'], landmarks['right_ankle_y']))

        return features

    def _calculate_angle(self, p1, p2, p3):
        """Calcula el ángulo entre tres puntos"""
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)

        return np.degrees(angle)
