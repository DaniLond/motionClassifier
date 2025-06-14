import os
import pandas as pd
import numpy as np
from data_collection.video_recorder import VideoRecorder
from feature_extraction.pose_extractor import PoseExtractor
from data_collection.annotator import VideoAnnotator
from preprocessing.data_processor import DataProcessor
from models.model_trainer import ActivityClassifierTrainer


def main():
    print("=== PIPELINE DE ENTRENAMIENTO DE MODELO ===")

    
    raw_video_dir = "data/raw"
    processed_dir = "data/processed"
    annotations_dir = "data/annotations"
    models_dir = "models"

   
    for dir_path in [raw_video_dir, processed_dir, annotations_dir, models_dir]:
        os.makedirs(dir_path, exist_ok=True)

   
    print("\n1. RECOLECCIÓN DE DATOS")
    collect_new_data = input(
        "¿Deseas grabar nuevos videos? (y/n): ").lower() == 'y'

    if collect_new_data:
        recorder = VideoRecorder(raw_video_dir)
        activities = ['walk_forward', 'walk_back', 'turn', 'sit', 'stand']

        num_people = int(input("¿Cuántas personas participarán? "))

        for person_id in range(1, num_people + 1):
            person_name = f"person_{person_id:02d}"
            print(f"\nGrabando para {person_name}")

            for activity in activities:
                num_repetitions = int(
                    input(f"¿Cuántas repeticiones de '{activity}' para {person_name}? "))

                for rep in range(num_repetitions):
                    print(
                        f"Preparándose para grabar {activity} - repetición {rep + 1}")
                    input("Presiona Enter cuando estés listo...")

                    video_path = recorder.record_activity_session(
                        f"{person_name}_rep{rep + 1}",
                        activity
                    )
                    print(f"Video guardado: {video_path}")

    print("\n2. EXTRACCIÓN DE CARACTERÍSTICAS")
    pose_extractor = PoseExtractor()

    for video_file in os.listdir(raw_video_dir):
        if not video_file.endswith('.mp4'):
            continue

        video_path = os.path.join(raw_video_dir, video_file)
        landmark_file = video_file.replace('.mp4', '_landmarks.csv')
        landmark_path = os.path.join(processed_dir, landmark_file)

        
        if os.path.exists(landmark_path):
            print(f"Saltando {video_file} - ya procesado")
            continue

        print(f"Procesando {video_file}...")
        landmarks_df = pose_extractor.extract_landmarks_from_video(video_path)
        landmarks_df.to_csv(landmark_path, index=False)
        print(f"Características guardadas: {landmark_path}")

    
    print("\n3. ANOTACIÓN DE VIDEOS")
    annotate_videos = input(
        "¿Deseas anotar videos? (y/n): ").lower() == 'y'

    if annotate_videos:
        annotator = VideoAnnotator()

       
        method = input("Método de anotación (auto/semi/manual): ").lower()
        if method not in ['auto', 'semi', 'manual']:
            method = 'manual'

        for video_file in os.listdir(raw_video_dir):
            if not video_file.endswith('.mp4'):
                continue

            annotation_file = video_file.replace('.mp4', '_annotations.json')
            annotation_path = os.path.join(annotations_dir, annotation_file)

            if os.path.exists(annotation_path):
                print(f"Saltando {video_file} - ya anotado")
                continue

            video_path = os.path.join(raw_video_dir, video_file)
            print(f"Anotando {video_file} usando método {method}...")

            try:
                    
                annotations = annotator.annotate_video(video_path, method=method)

               
                if annotations:  
                   
                    import json
                    with open(annotation_path, 'w') as f:
                        json.dump(annotations, f, indent=2)
                    print(f"✅ Anotaciones guardadas en: {annotation_path}")
                    print(f"   Número de anotaciones: {len(annotations)}")
                else:
                    print(f"⚠️  No se generaron anotaciones para {video_file}")

            except Exception as e:
                print(f"❌ Error anotando {video_file}: {e}")
                continue

 
    print("\n4. PREPROCESAMIENTO DE DATOS")
    processor = DataProcessor()

    
    print("Cargando y combinando datos...")
    combined_df = processor.load_and_merge_data(processed_dir, annotations_dir)
    print(f"Datos combinados: {len(combined_df)} frames")

   
    print("Aplicando suavizado...")
    smoothed_df = processor.smooth_landmarks(combined_df)

    
    print("Extrayendo características de movimiento...")
    feature_df = processor.extract_movement_features(smoothed_df)

    
    print("Preparando datos para entrenamiento...")
    X, y, feature_columns = processor.prepare_training_data(feature_df)

    print(
        f"Datos preparados: {X.shape[0]} muestras, {X.shape[1]} características")
    print(f"Distribución de clases: {pd.Series(y).value_counts().to_dict()}")

    
    print("\n5. ENTRENAMIENTO DE MODELOS")
    trainer = ActivityClassifierTrainer()
    trainer.feature_columns = feature_columns

    
    print("Entrenando múltiples modelos...")
    results = trainer.train_multiple_models(X, y)

   
    print("\n=== RESULTADOS DE ENTRENAMIENTO ===")
    for model_name, model_data in results.items():
        print(f"\n{model_name}:")
        print(f"  Accuracy: {model_data['accuracy']:.4f}")
        print(
            f"  CV Score: {model_data['cv_mean']:.4f} (+/- {model_data['cv_std']:.4f})")

    
    model_path = os.path.join(models_dir, "best_activity_classifier.pkl")
    trainer.save_best_model(model_path)

    print(f"\n✅ Entrenamiento completado!")
    print(f"Mejor modelo guardado en: {model_path}")
    print(
        f"Mejor modelo: {trainer.best_model_name} con accuracy: {trainer.best_score:.4f}")


if __name__ == "__main__":
    main()
