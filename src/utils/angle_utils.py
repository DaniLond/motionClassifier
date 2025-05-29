import numpy as np

def calculate_angle(a, b, c):
    """
    Calcular el ángulo entre tres puntos.
    
    Args:
        a (tuple): Coordenadas del primer punto (x, y)
        b (tuple): Coordenadas del punto central (x, y)
        c (tuple): Coordenadas del último punto (x, y)
    
    Returns:
        float: Ángulo en grados
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
    return angle

def calculate_shoulder_angle(shoulder, elbow, wrist):
    """
    Calcular el ángulo del hombro.
    
    Args:
        shoulder (tuple): Coordenadas del hombro (x, y)
        elbow (tuple): Coordenadas del codo (x, y)
        wrist (tuple): Coordenadas de la muñeca (x, y)
    
    Returns:
        float: Ángulo del hombro en grados
    """
    return calculate_angle(shoulder, elbow, wrist)

def calculate_elbow_angle(shoulder, elbow, wrist):
    """
    Calcular el ángulo del codo.
    
    Args:
        shoulder (tuple): Coordenadas del hombro (x, y)
        elbow (tuple): Coordenadas del codo (x, y)
        wrist (tuple): Coordenadas de la muñeca (x, y)
    
    Returns:
        float: Ángulo del codo en grados
    """
    return calculate_angle(shoulder, elbow, wrist) 