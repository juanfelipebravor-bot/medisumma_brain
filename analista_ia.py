import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

print("--- INICIANDO ANÁLISIS AUTOMÁTICO ---")

# 1. Cargar al Paciente (El archivo que creamos antes)
archivo = "holter_prueba.dat"
fs = 500 # Frecuencia de muestreo (Hz)

try:
    datos = np.fromfile(archivo, dtype=np.int16)
    
    # Tomamos solo 5 segundos para visualizar bien (2500 muestras)
    limite = 5 * fs 
    segmento = datos[:limite]
    tiempo = np.arange(len(segmento)) / fs

    print("🔎 Buscando complejos QRS...")

    # 2. EL CEREBRO DE LA IA (Algoritmo de Detección)
    # Buscamos picos que sean altos (prominencia) y estén separados (distancia)
    # height=500: Ignora el ruido de fondo, solo mira picos altos
    # distance=150: Evita contar la onda T como un nuevo latido (Periodo refractario)
    picos, _ = find_peaks(segmento, height=500, distance=150)

    num_latidos = len(picos)
    fc_estimada = (num_latidos / 5) * 60  # Regla de tres simple para sacar LPM
    
    print(f"✅ Detección finalizada.")
    print(f"❤️ Latidos detectados en 5 seg: {num_latidos}")
    print(f"🩺 Frecuencia Cardiaca Instantánea: ~{int(fc_estimada)} LPM")

    # 3. VISUALIZACIÓN DIAGNÓSTICA
    plt.figure(figsize=(12, 5), facecolor='black')
    ax = plt.gca()
    ax.set_facecolor('black')

    # La Señal (Cian)
    plt.plot(tiempo, segmento, color='#00FFFF', label='Señal Raw', alpha=0.8)
    
    # La IA (Puntos Rojos sobre los latidos)
    plt.plot(tiempo[picos], segmento[picos], "ro", markersize=8, label='Detección IA')

    plt.title(f"Análisis IA: Frecuencia ~{int(fc_estimada)} LPM", color='white', fontsize=14)
    plt.xlabel("Tiempo (s)", color='gray')
    plt.legend(loc='upper right')
    plt.grid(color='white', alpha=0.1)
    plt.tick_params(colors='gray')
    
    plt.show()

except Exception as e:
    print(f"❌ Error: {e}")