import mediapipe as mp
import cv2
import numpy as np

class PoseDetector:
    def __init__(self, static_image_mode=False, model_complexity=2, 
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Inicializar el detector de poses.
        
        Args:
            static_image_mode (bool): Si es True, el detector se ejecuta en modo imagen estática
            model_complexity (int): Complejidad del modelo (0, 1, o 2)
            min_detection_confidence (float): Confianza mínima para la detección
            min_tracking_confidence (float): Confianza mínima para el seguimiento
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
    
    def detect_pose(self, frame):
        """
        Detectar poses en un frame.
        
        Args:
            frame: Frame de imagen en formato BGR
            
        Returns:
            tuple: (frame procesado, landmarks detectados)
        """
        # Convertir a RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Procesar con MediaPipe
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            # Dibujar landmarks
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            return frame, results.pose_landmarks
        
        return frame, None
    
    def get_landmarks_dict(self, landmarks):
        """
        Obtener un diccionario con las coordenadas de los landmarks.
        
        Args:
            landmarks: Objeto landmarks de MediaPipe
            
        Returns:
            dict: Diccionario con las coordenadas de los landmarks
        """
        if not landmarks:
            return {}
            
        return {
            'left_shoulder': (landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                            landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y),
            'right_shoulder': (landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                             landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y),
            'left_elbow': (landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW].x,
                          landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW].y),
            'right_elbow': (landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                           landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW].y),
            'left_wrist': (landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST].x,
                          landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST].y),
            'right_wrist': (landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST].x,
                           landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST].y)
        } 