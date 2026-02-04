import numpy as np
from scipy.signal import find_peaks

def analizar_senal_ecg(datos_leidos, frecuencia_muestreo=1000):
    """
    Analiza una señal de ECG/Holter para detectar arritmias
    basándose en la variabilidad de los intervalos R-R.
    """
    
    # 1. Simular señal si no hay datos reales (Para pruebas)
    # Si los datos vienen vacíos, generamos un ritmo sinusal perfecto
    if len(datos_leidos) < 100:
        t = np.linspace(0, 10, 10000)
        datos_leidos = np.sin(2 * np.pi * 1.2 * t) * 1000 # 72 LPM simulados
    
    senal = np.array(datos_leidos)

    # 2. Detectar Picos R (Los latidos)
    # Buscamos picos que sean altos (prominence) y estén separados (distance)
    # Ajustamos la altura mínima basada en la señal (60% del máximo)
    altura_minima = np.max(senal) * 0.5
    distancia_minima = frecuencia_muestreo * 0.3 # Al menos 300ms entre latidos (max 200 LPM)
    
    picos, _ = find_peaks(senal, height=altura_minima, distance=distancia_minima)
    
    # 3. Calcular Frecuencia Cardíaca
    num_latidos = len(picos)
    duracion_segundos = len(senal) / frecuencia_muestreo
    bpm = (num_latidos / duracion_segundos) * 60
    
    # 4. Análisis de Ritmo (La parte Inteligente 🧠)
    diagnostico = "Ritmo No Determinado"
    color_alerta = "gray"
    
    if num_latidos < 2:
        diagnostico = "Señal Insuficiente"
    else:
        # Calcular distancias entre picos (Intervalos R-R)
        intervalos_rr = np.diff(picos) # Diferencia entre latido 1 y 2, 2 y 3...
        
        # Calcular la desviación estándar (Qué tan "locos" están los latidos)
        # Una desviación alta significa caos (Arritmia)
        desviacion_rr = np.std(intervalos_rr)
        variabilidad_ms = (desviacion_rr / frecuencia_muestreo) * 1000
        
        # Criterios Diagnósticos Simplificados
        if bpm > 100:
            diagnostico = f"Taquicardia ({int(bpm)} LPM)"
            color_alerta = "red"
        elif bpm < 60:
            diagnostico = f"Bradicardia ({int(bpm)} LPM)"
            color_alerta = "orange"
        elif variabilidad_ms > 120: # Más de 120ms de variación es sospechoso
            diagnostico = "Posible Fibrilación Auricular (Irregular)"
            color_alerta = "red"
        else:
            diagnostico = "Ritmo Sinusal Normal"
            color_alerta = "green"

    # 5. Empaquetar resultados
    resultado = {
        "latidos_detectados": int(num_latidos),
        "frecuencia_cardiaca": int(bpm),
        "diagnostico_texto": diagnostico,
        "alerta_color": color_alerta,
        "senal_grafica": senal.tolist() # Enviamos los datos para pintar
    }
    
    return resultado