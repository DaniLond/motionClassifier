import cv2
import time
import os
from models.pose_detector import PoseDetector
from utils.angle_utils import calculate_shoulder_angle, calculate_elbow_angle
import mediapipe as mp
from mediapipe.python.solutions import pose as mp_pose

def normalize_lighting(frame):
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

class TennisTracker:
    def __init__(self):
        self.pose_detector = PoseDetector(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.cap = cv2.VideoCapture(0)
        cv2.namedWindow('Tennis Motion Tracker', cv2.WINDOW_NORMAL)
        
        self.prev_time = 0
        self.current_time = 0
        
        self.landmarks_dict = {}
        
    
    def process_frame(self, frame):
        """Procesar un frame y detectar poses"""
        normalized_frame = normalize_lighting(frame)
        
        processed_frame, landmarks = self.pose_detector.detect_pose(normalized_frame)
        
        if landmarks:
            self.landmarks_dict = self.pose_detector.get_landmarks_dict(landmarks)
            
            if len(self.landmarks_dict) >= 6:
                right_shoulder_angle = calculate_shoulder_angle(
                    self.landmarks_dict['right_shoulder'],
                    self.landmarks_dict['right_elbow'],
                    self.landmarks_dict['right_wrist']
                )
                
                right_elbow_angle = calculate_elbow_angle(
                    self.landmarks_dict['right_shoulder'],
                    self.landmarks_dict['right_elbow'],
                    self.landmarks_dict['right_wrist']
                )
                
                cv2.putText(processed_frame, f'Shoulder Angle: {int(right_shoulder_angle)}', 
                           (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(processed_frame, f'Elbow Angle: {int(right_elbow_angle)}', 
                           (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return processed_frame
    
    def run(self):
        """Ejecutar el sistema de seguimiento"""
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            normalized_frame = normalize_lighting(frame)
            processed_frame = self.process_frame(normalized_frame)
            
            self.current_time = time.time()
            fps = 1/(self.current_time - self.prev_time)
            self.prev_time = self.current_time
            
            cv2.putText(processed_frame, f'FPS: {int(fps)}', (20, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.putText(processed_frame, 'Presiona "q" para salir', (20, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Tennis Motion Tracker', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = TennisTracker()
    tracker.run() 
    
    