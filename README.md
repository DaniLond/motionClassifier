# Tennis Motion Tracker

Sistema de seguimiento de movimientos de tenis utilizando MediaPipe para la detección de poses y análisis de movimientos.

## Características

- Detección de poses en tiempo real
- Seguimiento de articulaciones clave para movimientos de tenis
- Cálculo de ángulos de hombros y codos
- Interfaz gráfica con visualización de FPS y ángulos

## Estructura del Proyecto

```
motionClassifier/
├── src/
│   ├── models/
│   │   └── pose_detector.py
│   ├── utils/
│   │   └── angle_utils.py
│   └── tennis_tracker.py
├── data/
│   ├── raw/
│   └── processed/
├── requirements.txt
└── README.md
```

## Requisitos

- Python 3.10 o superior
- Webcam funcional
- Dependencias listadas en `requirements.txt`

## Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/DaniLond/motionClassifier.git
cd motionClassifier
```

2. Crear un entorno virtual (opcional pero recomendado):
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Uso

1. Ejecutar el programa principal:
```bash
python src/tennis_tracker.py
```

2. Controles:
- Presionar 'q' para salir del programa

## Características Técnicas

- Utiliza MediaPipe para la detección de poses
- Calcula ángulos de articulaciones en tiempo real
- Muestra FPS y ángulos en la interfaz
- Preparado para futura integración con clasificación de movimientos

## Próximas Características

- Clasificación de movimientos de tenis (saque, derecha, revés, volea, smash)
- Almacenamiento de datos para entrenamiento
- Interfaz mejorada con más información
- Análisis de técnica y retroalimentación 