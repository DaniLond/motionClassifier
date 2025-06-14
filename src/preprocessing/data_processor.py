import pandas as pd
import numpy as np
from scipy import signal
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import json
import os



class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = None

    def load_and_merge_data(self, processed_dir, annotation_dir):
        """Carga y combina datos de landmarks con anotaciones"""
        all_data = []

        # CORRECCIÓN: Iterar sobre archivos CSV de landmarks, no MP4
        for landmark_file in os.listdir(processed_dir):
            if not landmark_file.endswith('_landmarks.csv'):
                continue

            # Obtener el nombre base del video
            video_base_name = landmark_file.replace('_landmarks.csv', '')
            video_file = video_base_name + '.mp4'

            # Cargar landmarks
            landmark_path = os.path.join(processed_dir, landmark_file)
            if not os.path.exists(landmark_path):
                continue

            print(f"Cargando landmarks: {landmark_file}")
            landmarks_df = pd.read_csv(landmark_path)

            # Cargar anotaciones
            annotation_file = video_base_name + '_annotations.json'
            annotation_path = os.path.join(annotation_dir, annotation_file)

            if os.path.exists(annotation_path):
                print(f"Cargando anotaciones: {annotation_file}")
                with open(annotation_path, 'r') as f:
                    annotations = json.load(f)

                # Agregar etiquetas a landmarks
                landmarks_df['activity'] = 'idle'  # Etiqueta por defecto

                for ann in annotations:
                    mask = ((landmarks_df['frame'] >= ann['start_frame']) &
                            (landmarks_df['frame'] <= ann['end_frame']))
                    landmarks_df.loc[mask, 'activity'] = ann['activity']
            else:
                print(f"⚠️  No se encontraron anotaciones para {video_base_name}")
                # Si no hay anotaciones, marcar todo como 'idle'
                landmarks_df['activity'] = 'idle'

            # Agregar metadatos
            landmarks_df['video_file'] = video_file
            landmarks_df['person_id'] = video_base_name.split('_')[0]

            all_data.append(landmarks_df)
            print(f"✅ Procesado: {landmark_file} ({len(landmarks_df)} frames)")

        # Verificar que tengamos datos antes de concatenar
        if not all_data:
            raise ValueError("No se encontraron archivos de landmarks para procesar. "
                            "Asegúrate de que existen archivos *_landmarks.csv en el directorio processed/")

        print(f"📊 Total de archivos procesados: {len(all_data)}")
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"📊 Total de frames combinados: {len(combined_df)}")

        return combined_df

    def smooth_landmarks(self, df, window_size=5):
        """Aplica suavizado a las coordenadas de landmarks"""
        smoothed_df = df.copy()

        # Columnas de coordenadas a suavizar
        coord_columns = [
            col for col in df.columns if col.endswith(('_x', '_y', '_z'))]

        for col in coord_columns:
            smoothed_df[col] = signal.savgol_filter(df[col], window_size, 3)

        return smoothed_df

    def extract_movement_features(self, df):
        """Extrae características de movimiento"""
        feature_df = df.copy()

        # Velocidades
        for coord in ['_x', '_y', '_z']:
            coord_cols = [col for col in df.columns if col.endswith(coord)]
            for col in coord_cols:
                velocity_col = col.replace(coord, f'_velocity{coord}')
                feature_df[velocity_col] = df[col].diff()

        # Aceleraciones
        velocity_cols = [
            col for col in feature_df.columns if 'velocity' in col]
        for col in velocity_cols:
            acc_col = col.replace('velocity', 'acceleration')
            feature_df[acc_col] = feature_df[col].diff()

        # Características agregadas por ventana de tiempo
        window_features = self._calculate_window_features(feature_df)

        return pd.concat([feature_df, window_features], axis=1)

    def _calculate_window_features(self, df, window_size=30):
        """Calcula características agregadas en ventanas de tiempo"""
        features = []

        for i in range(window_size, len(df)):
            window_data = df.iloc[i-window_size:i]

            window_features = {
                'window_end_frame': df.iloc[i]['frame'],
                'movement_variance': window_data[['center_mass_x', 'center_mass_y']].var().mean(),
                'avg_knee_angle': (window_data['left_knee_angle'].mean() +
                                   window_data['right_knee_angle'].mean()) / 2,
                'trunk_stability': window_data['trunk_lateral_inclination'].std(),
                'vertical_movement': window_data['center_mass_y'].std()
            }

            features.append(window_features)

        return pd.DataFrame(features)

    def prepare_training_data(self, df):
        """Prepara datos para entrenamiento"""
        # Seleccionar características relevantes
        feature_columns = [
            # Posiciones normalizadas
            'center_mass_x', 'center_mass_y',
            'left_knee_angle', 'right_knee_angle',
            'trunk_lateral_inclination',
            'person_height',

            # Características de movimiento
            'movement_variance',
            'avg_knee_angle',
            'trunk_stability',
            'vertical_movement'
        ]

        # Filtrar filas con datos completos
        valid_rows = df[feature_columns + ['activity']].dropna()

        X = valid_rows[feature_columns]
        y = valid_rows['activity']

        # Normalizar características
        X_scaled = self.scaler.fit_transform(X)

        self.feature_columns = feature_columns

        return X_scaled, y, feature_columns
