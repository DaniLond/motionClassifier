import cv2
import argparse

from real_time.real_time_classifer import RealTimeActivityClassifier



def run_demo(model_path):
    """Ejecuta demo del sistema de detección usando OpenCV"""
    print("Iniciando sistema de detección de actividades...")

    classifier = RealTimeActivityClassifier(model_path)

    classifier.run_real_time_detection()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Sistema de Detección de Actividades Humanas - Versión OpenCV')
    parser.add_argument('--model', required=True,
                        help='Ruta al modelo entrenado (.pkl)')
    parser.add_argument('--camera', type=int, default=0,
                        help='Índice de la cámara a usar (por defecto: 0)')

    args = parser.parse_args()
    run_demo(args.model)
