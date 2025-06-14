import cv2
import numpy as np
import pandas as pd
from collections import deque
import joblib
from src.feature_extraction.pose_extractor import PoseExtractor
from src.preprocessing.data_processor import DataProcessor


class RealTimeActivityClassifier:
    def __init__(self, model_path, scaler_path=None):
       
        self.model_data = joblib.load(model_path)
        self.model = self.model_data['model']
        self.feature_columns = self.model_data['feature_columns']

       
        self.pose_extractor = PoseExtractor()
        self.data_processor = DataProcessor()

       
        self.frame_buffer = deque(maxlen=30)  
        self.prediction_history = deque(maxlen=10)

        
        self.current_activity = "idle"
        self.confidence = 0.0

    def process_frame(self, frame):
        """Procesa un frame y retorna la actividad detectada"""
       
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose_extractor.pose.process(rgb_frame)

        if results.pose_landmarks:
           
            landmarks = self.pose_extractor._extract_key_landmarks(
                results.pose_landmarks)
            self.frame_buffer.append(landmarks)

            
            if len(self.frame_buffer) >= 10:
                prediction, confidence = self._predict_activity()
                self.prediction_history.append(prediction)

                
                self.current_activity = self._smooth_predictions()
                self.confidence = confidence

          
            annotated_frame = self._draw_pose_and_info(
                frame, results.pose_landmarks)
            return annotated_frame, self.current_activity, self.confidence

        else:
            
            return frame, "no_person_detected", 0.0

    def _predict_activity(self):
        """Hace predicción basada en el buffer de frames"""
        if len(self.frame_buffer) < 10:
            return "idle", 0.0

       
        recent_frames = list(self.frame_buffer)[-10:]  
        df = pd.DataFrame(recent_frames)

      
        movement_features = self._calculate_realtime_features(df)

        
        try:
            
            feature_data = []
            for col in self.feature_columns:
                if col in movement_features:
                    feature_data.append(movement_features[col])
                else:
                    feature_data.append(0.0)  

          
            X = np.array(feature_data).reshape(1, -1)
            prediction = self.model.predict(X)[0]

            
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X)[0]
                confidence = max(probabilities)
            else:
                confidence = 0.8  

            return prediction, confidence

        except Exception as e:
            print(f"Error en predicción: {e}")
            return "idle", 0.0

    def _calculate_realtime_features(self, df):
        """Calcula características en tiempo real"""
        features = {}

        if len(df) < 2:
            return features

        
        last_frame = df.iloc[-1]
        features.update({
            'center_mass_x': last_frame.get('center_mass_x', 0),
            'center_mass_y': last_frame.get('center_mass_y', 0),
            'left_knee_angle': last_frame.get('left_knee_angle', 90),
            'right_knee_angle': last_frame.get('right_knee_angle', 90),
            'trunk_lateral_inclination': last_frame.get('trunk_lateral_inclination', 0),
            'person_height': last_frame.get('person_height', 1.0)
        })

       
        if 'center_mass_x' in df.columns and 'center_mass_y' in df.columns:
            features['movement_variance'] = (
                df['center_mass_x'].var() + df['center_mass_y'].var()) / 2
            features['vertical_movement'] = df['center_mass_y'].std()
        else:
            features['movement_variance'] = 0
            features['vertical_movement'] = 0

       
        if 'left_knee_angle' in df.columns and 'right_knee_angle' in df.columns:
            features['avg_knee_angle'] = (
                df['left_knee_angle'].mean() + df['right_knee_angle'].mean()) / 2
        else:
            features['avg_knee_angle'] = 90

        if 'trunk_lateral_inclination' in df.columns:
            features['trunk_stability'] = df['trunk_lateral_inclination'].std()
        else:
            features['trunk_stability'] = 0

        return features

    def _smooth_predictions(self):
        """Suaviza predicciones usando votación mayoritaria"""
        if len(self.prediction_history) < 3:
            return self.current_activity

      
        recent_predictions = list(self.prediction_history)[-5:]
        prediction_counts = {}
        for pred in recent_predictions:
           prediction_counts[pred] = prediction_counts.get(pred, 0) + 1
       
       
        return max(prediction_counts, key=prediction_counts.get)
   
    def _draw_pose_and_info(self, frame, pose_landmarks):
       """Dibuja la pose y información en el frame"""
      
       annotated_frame = frame.copy()
       self.pose_extractor.mp_drawing.draw_landmarks(
           annotated_frame, 
           pose_landmarks, 
           self.pose_extractor.mp_pose.POSE_CONNECTIONS
       )
       
       
       cv2.putText(annotated_frame, f"Actividad: {self.current_activity}", 
                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
       cv2.putText(annotated_frame, f"Confianza: {self.confidence:.2f}", 
                  (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
       
      
       if len(self.frame_buffer) > 0:
           last_features = self.frame_buffer[-1]
           cv2.putText(annotated_frame, f"Inclinacion: {last_features.get('trunk_lateral_inclination', 0):.3f}", 
                      (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
           cv2.putText(annotated_frame, f"Rodilla Izq: {last_features.get('left_knee_angle', 0):.1f}°", 
                      (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
           cv2.putText(annotated_frame, f"Rodilla Der: {last_features.get('right_knee_angle', 0):.1f}°", 
                      (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
       
       return annotated_frame

    def run_real_time_detection(self):
       """Ejecuta detección en tiempo real desde la cámara"""
       cap = cv2.VideoCapture(0)
       
       print("Sistema de detección de actividades iniciado")
       print("Presiona 'q' para salir")
       
       while True:
           ret, frame = cap.read()
           if not ret:
               break
           
           
           annotated_frame, activity, confidence = self.process_frame(frame)
           
          
           cv2.imshow('Detección de Actividades', annotated_frame)
           
         
           if cv2.waitKey(1) & 0xFF == ord('q'):
               break
       
       cap.release()
       cv2.destroyAllWindows()
