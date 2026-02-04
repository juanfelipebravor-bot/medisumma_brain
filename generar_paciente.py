import numpy as np
import struct

print("--- GRABANDO PACIENTE: TAQUICARDIA (140 LPM) ---")

def crear_complejo_qrs(t):
    # (El mismo latido de siempre)
    p = 0.15 * np.exp(-((t - 0.2)**2) / (2 * 0.005))
    q = -0.1 * np.exp(-((t - 0.38)**2) / (2 * 0.0005))
    r = 1.0 * np.exp(-((t - 0.40)**2) / (2 * 0.001)) 
    s = -0.2 * np.exp(-((t - 0.42)**2) / (2 * 0.0005))
    t_wave = 0.25 * np.exp(-((t - 0.65)**2) / (2 * 0.01))
    return p + q + r + s + t_wave

# Configuración acelerada
fs = 500
segundos = 10 
t_total = np.linspace(0, segundos, segundos * fs)
senal_limpia = np.zeros_like(t_total)

# SIMULACIÓN DE 140 LPM
# 140 latidos por minuto = 2.33 latidos por segundo
# Significa un latido cada ~0.42 segundos (aprox 214 muestras)
espacio_entre_latidos = int(fs * (60 / 140)) 

print(f"⚡ Inyectando latidos cada {espacio_entre_latidos} muestras...")

# Bucle manual para insertar latidos rápidos
for i in range(0, len(senal_limpia) - fs, espacio_entre_latidos):
    t_local = np.linspace(0, 1, fs)
    latido = crear_complejo_qrs(t_local)
    
    # Sumar el latido a la señal base (Superposición)
    # Solo sumamos los primeros 0.4 seg del latido para que no se solapen feo
    corte = min(espacio_entre_latidos, fs)
    senal_limpia[i:i+corte] += latido[:corte]

# Ruido y Guardado
ruido = 0.05 * np.random.normal(0, 1, len(senal_limpia))
senal_final = senal_limpia + ruido
senal_digitalizada = (senal_final * 1000).astype(np.int16)

archivo = "paciente_taquicardia.dat" # <--- CAMBIAMOS EL NOMBRE
with open(archivo, "wb") as f:
    f.write(senal_digitalizada.tobytes())

print(f"⚠️ ¡Paciente crítico generado en '{archivo}'!")