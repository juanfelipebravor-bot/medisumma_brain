from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from scipy.signal import find_peaks
import io

# 1. INICIAR EL HOSPITAL (APP)
app = FastAPI(title="Cerebro MediSumma", version="1.0")

# 2. PERMISOS DE SEGURIDAD (CORS)
# Esto es vital: Le damos permiso a tu App de Flutter (que vive en otro puerto)
# para que hable con este servidor Python. Sin esto, el navegador bloquea la conexión.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción pondremos la URL real, para desarrollo usamos "*" (Todos)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("--- 🏥 SERVIDOR MEDISUMMA INICIADO ---")

@app.get("/")
def home():
    """Endpoint de prueba para ver si el servidor respira"""
    return {"estado": "En línea", "mensaje": "Listo para recibir Holters"}

@app.post("/analizar_holter")
async def analizar_holter(file: UploadFile = File(...)):
    """
    Recibe un archivo .DAT o .ECG, lo decodifica y devuelve el diagnóstico.
    """
    print(f"📥 Recibiendo archivo: {file.filename}")
    
    # A. LEER LOS BYTES (Extracción de sangre)
    contenido = await file.read()
    
    # B. CONVERTIR A SEÑAL NUMÉRICA (Procesamiento)
    # Asumimos int16 como en tus pruebas anteriores
    try:
        senal = np.frombuffer(contenido, dtype=np.int16)
    except Exception as e:
        return {"error": f"No se pudo leer el formato: {str(e)}"}

    # C. CORTAR UN SEGMENTO (Biopsia)
    # Analizamos solo los primeros 10 segundos para respuesta rápida
    fs = 500
    limite = 10 * fs
    segmento = senal[:limite]

    # D. DIAGNÓSTICO CON IA (Tu algoritmo de picos)
    # Usamos height=500 como umbral base, ajustado a tu simulación
    picos, _ = find_peaks(segmento, height=500, distance=150)
    
    num_latidos = len(picos)
    # Calcular Frecuencia Cardiaca: (Latidos / 10 seg) * 60 seg
    fc_estimada = int((num_latidos / 10) * 60)

    print(f"✅ Diagnóstico: {fc_estimada} LPM detectados.")

    # E. EMPAQUETAR RESULTADOS (Informe Médico)
    # Convertimos los numpy arrays a listas normales de Python para poder enviarlas por internet
    # Enviamos solo 1000 puntos para graficar rápido en el frontend (2 segundos)
    datos_para_grafica = segmento[:1000].tolist() 

    return {
        "filename": file.filename,
        "duracion_analizada": "10 segundos",
        "latidos_detectados": num_latidos,
        "frecuencia_cardiaca": fc_estimada,
        "diagnostico_texto": "Ritmo Sinusal" if 60 <= fc_estimada <= 100 else "Posible Arritmia",
        "senal_grafica": datos_para_grafica # Esto es lo que dibujará Flutter
    }