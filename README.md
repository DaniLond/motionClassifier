# Integrantes
- Leidy Daniela Londoño
- Danna Valentina 
- Isabella Huila

# Sistema de Detección de Actividades Humanas

Este proyecto es un sistema completo para la detección y clasificación de actividades humanas en videos, utilizando técnicas de visión por computadora y aprendizaje automático. El sistema incluye módulos para la recolección de datos, extracción de características, anotación de videos, preprocesamiento de datos, entrenamiento de modelos y una interfaz para detección en tiempo real.

## Características principales

- **Grabación de videos**: Captura de videos de actividades humanas con metadatos integrados.
- **Anotación flexible**: Soporte para anotación.
- **Extracción de poses**: Detección de landmarks corporales usando MediaPipe.
- **Procesamiento de datos**: Suavizado de landmarks y extracción de características de movimiento.
- **Modelos de ML**: Entrenamiento de múltiples modelos (Random Forest, SVM, XGBoost) con selección automática del mejor.

## Componentes del sistema

1. **`VideoRecorder`**: Graba videos de actividades con etiquetas integradas.
2. **`VideoAnnotator`**: Herramienta para anotar videos con diferentes métodos.
3. **`PoseExtractor`**: Extrae landmarks corporales y características derivadas.
4. **`DataProcessor`**: Preprocesa y prepara los datos para entrenamiento.
5. **`ActivityClassifierTrainer`**: Entrena y evalúa modelos de clasificación.
6. **`RealTimeActivityClassifier`**: Clasifica actividades en tiempo real.

## Instalación

1. Clona el repositorio:
   ```bash
   https://github.com/DaniLond/motionClassifier.git

2. Instala las dependencias:
    ```bash
   pip install -r requirements.txt

### Uso

1. Pipeline completo
Ejecuta el pipeline completo (recolección de datos, procesamiento, entrenamiento):
    ```bash
    python main.py

2. Demo en tiempo real
    ```bash
    python src/demo.py --model models/best_activity_classifier.pkl

## Actividades soportadas

El sistema puede detectar las siguientes actividades:

- walk_forward: Caminar hacia adelante

- walk_back: Caminar hacia atrás

- turn: Girar

- sit: Sentarse

- stand: Pararse
