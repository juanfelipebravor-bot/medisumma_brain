import numpy as np
import matplotlib.pyplot as plt

print("--- GENERADOR DE ECG CLÍNICO ---")

def crear_latido(t):
    """Crea un complejo P-QRS-T sintético"""
    # Onda P (Auricular)
    p_wave = 0.15 * np.exp(-((t - 0.2)**2) / (2 * 0.005))
    # Complejo QRS (Ventricular - El "Pico")
    q_wave = -0.1 * np.exp(-((t - 0.38)**2) / (2 * 0.0005))
    r_wave = 1.0 * np.exp(-((t - 0.40)**2) / (2 * 0.001))
    s_wave = -0.2 * np.exp(-((t - 0.42)**2) / (2 * 0.0005))
    # Onda T (Repolarización)
    t_wave = 0.25 * np.exp(-((t - 0.65)**2) / (2 * 0.01))
    
    return p_wave + q_wave + r_wave + s_wave + t_wave

# Configuración del Monitor
fs = 500        # 500 Hz (Estándar médico)
segundos = 3    # Duración de la tira
t_total = np.linspace(0, segundos, segundos * fs)
ecg_completo = np.zeros_like(t_total)

# Generar latidos (Simulando 60 LPM)
print("🫀 Inyectando latidos fisiológicos...")
for i in range(segundos):
    # Crear un latido de 1 segundo
    t_local = np.linspace(0, 1, fs) 
    latido = crear_latido(t_local)
    
    # Insertarlo en la línea de tiempo
    inicio = i * fs
    fin = inicio + fs
    if fin <= len(ecg_completo):
        ecg_completo[inicio:fin] = latido

# Añadir "Ruido Eléctrico" (Para realismo)
ruido = 0.02 * np.random.normal(0, 1, len(ecg_completo))
senal_final = ecg_completo + ruido

# Visualización Tipo "Monitor UCI"
print("🖥️  Renderizando monitor...")
plt.figure(figsize=(12, 5), facecolor='black') # Marco negro
ax = plt.gca()
ax.set_facecolor('black') # Fondo negro
plt.plot(t_total, senal_final, color='#00ff00', linewidth=1.5) # Verde Fósforo
plt.title("Derivación II - Ritmo Sinusal (60 LPM)", color='white', fontsize=14)
plt.xlabel("Tiempo (s)", color='gray')
plt.ylabel("mV", color='gray')
plt.grid(color='green', linestyle=':', alpha=0.3)
plt.tick_params(colors='gray')
plt.show()
