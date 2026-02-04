import numpy as np
import matplotlib.pyplot as plt

print("--- INICIANDO LECTURA DE HOLTER ---")

archivo = "holter_prueba.dat"

try:
    # 1. LEER EL ARCHIVO BINARIO
    # Le decimos a Python: "Lee este archivo asumiendo que son enteros de 16 bits (int16)"
    # Esto es CRÍTICO: Si nos equivocamos de formato (ej. int32), la señal saldrá deforme.
    datos_crudos = np.fromfile(archivo, dtype=np.int16)
    
    print(f"✅ Archivo cargado exitosamente.")
    print(f"📊 Muestras totales recuperadas: {len(datos_crudos)}")
    
    # Calcular duración real basada en la frecuencia (500 Hz)
    fs = 500
    duracion_minutos = (len(datos_crudos) / fs) / 60
    print(f"⏱️ Duración estimada del estudio: {duracion_minutos:.1f} minutos")

    # 2. VISUALIZAR (Haremos un Zoom)
    # No vamos a graficar todo el minuto porque se vería muy apretado.
    # Vamos a ver solo los primeros 3 segundos (1500 muestras).
    muestras_zoom = 1500
    zoom_senal = datos_crudos[:muestras_zoom]
    tiempo = np.arange(muestras_zoom) / fs

    print("📈 Generando telemetría...")
    
    plt.figure(figsize=(12, 5), facecolor='black')
    ax = plt.gca()
    ax.set_facecolor('black')
    
    # Graficamos en Cian (Cyan) estilo futurista
    plt.plot(tiempo, zoom_senal, color='#00FFFF', linewidth=1.5)
    
    plt.title(f"Visualización de Datos Crudos: {archivo}", color='white')
    plt.xlabel("Segundos", color='gray')
    plt.ylabel("Amplitud (Digital)", color='gray')
    plt.grid(color='#00FFFF', linestyle=':', alpha=0.2)
    plt.tick_params(colors='gray')
    
    plt.show()

except FileNotFoundError:
    print("❌ ERROR: No encuentro el archivo 'holter_prueba.dat'.")
    print("Asegúrate de haber ejecutado el paso anterior primero.")
    