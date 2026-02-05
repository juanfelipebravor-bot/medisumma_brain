from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import scipy.signal as signal
import cv2
import os

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Cerebro MediSumma v7.0: OCR de Electrocardiograma ACTIVO 👁️📉"

# --- FUNCIÓN AUXILIAR 1: PROCESAMIENTO DE IMAGEN (OCR) ---
def extraer_senal_imagen(path):
    # 1. Leer imagen
    img = cv2.imread(path)
    if img is None: return []

    # Redimensionar para estandarizar (ancho 1000px)
    aspect_ratio = img.shape[0] / img.shape[1]
    target_width = 1000
    target_height = int(target_width * aspect_ratio)
    img = cv2.resize(img, (target_width, target_height))

    # 2. Escala de Grises y Umbralizado
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Invertimos: Tinta negra se vuelve Blanca (255), Papel blanco se vuelve Negro (0)
    # Usamos un umbral (90) para intentar ignorar la cuadrícula rosada clara
    _, binary = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)

    # 3. Escaneo Columna por Columna (Digitalización)
    senal = []
    height, width = binary.shape

    for x in range(width):
        col = binary[:, x]
        # Buscamos índices donde hay "tinta" (píxeles blancos en negativo)
        indices = np.where(col > 0)[0]
        
        if len(indices) > 0:
            # Promedio de la posición Y de la tinta en esa columna
            y_avg = np.mean(indices)
            # Invertimos eje Y (en imagen 0 es arriba, en gráfica 0 es abajo)
            val = height - y_avg
            senal.append(val)
        else:
            # Si no hay tinta en esta columna (ruido o hueco), repetimos el último valor conocido
            senal.append(senal[-1] if len(senal) > 0 else height/2)

    # 4. Normalización (Centrar en 0)
    senal = np.array(senal)
    senal = senal - np.mean(senal)
    
    # 5. Amplificación (Ganancia visual para la app)
    senal = senal * 5 
    
    return senal.tolist()

# --- FUNCIÓN AUXILIAR 2: MOTOR DE DIAGNÓSTICO COMPARTIDO ---
def motor_diagnostico(ecg_signal, fs):
    # Detección QRS
    distance = int(fs * 0.4) 
    # Umbral adaptativo (40% del máximo)
    height_threshold = np.max(ecg_signal) * 0.4 
    peaks, _ = signal.find_peaks(ecg_signal, height=height_threshold, distance=distance)
    
    if len(peaks) < 2:
        return 0, "TRAZADO NO LEÍBLE / RUIDO", "grey", 0, False

    rr_intervals = np.diff(peaks)
    mean_rr = np.mean(rr_intervals)
    std_rr = np.std(rr_intervals)
    bpm = int(60000 / (mean_rr / fs * 1000))
    cv_rr = std_rr / mean_rr 

    # Diagnóstico
    diagnostico = "Ritmo Sinusal"
    color = "green"
    
    if cv_rr > 0.15:
        diagnostico = "POSIBLE FIBRILACIÓN AURICULAR (Irregular)"
        color = "red"
    else:
        if bpm > 100:
            diagnostico = "TAQUICARDIA"
            color = "orange"
        elif bpm < 60:
            diagnostico = "BRADICARDIA"
            color = "green"
        else:
            diagnostico = "RITMO SINUSAL NORMAL"
            color = "green"
            
    return bpm, diagnostico, color, cv_rr, True

# --- ENDPOINT 1: ANÁLISIS DE FOTO ---
@app.route('/analizar_ecg_foto', methods=['POST'])
def analizar_ecg_foto():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file"}), 400
        
        file = request.files['file']
        filepath = f"/tmp/{file.filename}"
        file.save(filepath)

        # 1. Extracción de señal (OCR)
        senal_grafica = extraer_senal_imagen(filepath)
        
        if len(senal_grafica) < 100:
             return jsonify({
                "status": "success",
                "grid_detected": False,
                "mensaje": "No se pudo extraer señal clara.",
                "diagnostico_texto": "ERROR DE LECTURA",
                "alerta_color": "grey",
                "frecuencia_cardiaca": 0,
                "senal_grafica": []
            })

        # 2. Análisis
        # Estimamos FS heurística: 1000px ancho / ~3 segundos = 333 Hz
        fs_estimada = 333.0 
        
        bpm, dx, color, cv, exito = motor_diagnostico(np.array(senal_grafica), fs_estimada)
        
        # Limpieza
        try: os.remove(filepath)
        except: pass

        return jsonify({
            "status": "success",
            "grid_detected": True,
            "mensaje": "Digitalización Completada",
            "senal_grafica": senal_grafica, # La señal extraída
            "frecuencia_cardiaca": bpm,
            "diagnostico_texto": dx,
            "alerta_color": color,
            "detalles": f"Fuente: CÁMARA | CV: {cv:.2f}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- ENDPOINT 2: ANÁLISIS DE HOLTER (.DAT) ---
@app.route('/analizar_holter', methods=['POST'])
def analizar_holter():
    try:
        if 'file' not in request.files: return jsonify({"error": "No file"}), 400
        file = request.files['file']
        filepath = f"/tmp/{file.filename}"
        file.save(filepath)
        
        raw_data = np.fromfile(filepath, dtype=np.int16)
        # Limitamos a 5000 muestras para análisis rápido
        ecg_signal = raw_data[:5000] if len(raw_data) > 5000 else raw_data
        
        bpm, dx, color, cv, exito = motor_diagnostico(ecg_signal, 500.0) # FS 500Hz
        
        try: os.remove(filepath)
        except: pass

        return jsonify({
            "frecuencia_cardiaca": bpm,
            "diagnostico_texto": dx,
            "alerta_color": color,
            "senal_grafica": ecg_signal.tolist(),
            "detalles": f"Fuente: HOLTER DIGITAL | CV: {cv:.2f}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)