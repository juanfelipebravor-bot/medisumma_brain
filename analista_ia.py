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
    return "Cerebro MediSumma v7.1: Filtro de Cuadrícula AVANZADO 📉🧼"

# --- NUEVO MOTOR DE EXTRACCIÓN LIMPIA ---
def extraer_senal_imagen(path):
    # 1. Leer imagen
    img = cv2.imread(path)
    if img is None: return []

    # 2. ELIMINACIÓN DE CUADRÍCULA (El paso clave que faltaba)
    # Convertimos a HSV para detectar el ROJO/ROSADO del papel
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Rangos para detectar tinta roja/rosa (grid)
    lower_red1 = np.array([0, 20, 20])
    upper_red1 = np.array([20, 255, 255])
    lower_red2 = np.array([160, 20, 20])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    grid_mask = mask1 + mask2
    
    # "Borramos" la cuadrícula pintándola de blanco en la imagen original
    img_clean = img.copy()
    img_clean[grid_mask > 0] = [255, 255, 255] # Blanco

    # 3. ENFOQUE QUIRÚRGICO (Recorte Automático)
    # En lugar de leer toda la hoja (12 derivadas mezcladas), cortamos la franja central
    # Asumimos que en el centro hay una derivada limpia (ej. V2/V5)
    height, width, _ = img_clean.shape
    start_y = int(height * 0.40) # Empezar al 40% de la altura
    end_y = int(height * 0.60)   # Terminar al 60%
    roi = img_clean[start_y:end_y, :] # Recorte central
    
    # 4. Extracción de Tinta Negra
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Umbralizado más agresivo para quedarnos solo con lo MUY negro (tinta)
    _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

    # 5. Escaneo (Digitalización)
    senal = []
    h_roi, w_roi = binary.shape
    
    for x in range(w_roi):
        col = binary[:, x]
        indices = np.where(col > 0)[0]
        
        if len(indices) > 0:
            # Tomamos la mediana para evitar ruido outlier
            y_avg = np.median(indices)
            val = h_roi - y_avg
            senal.append(val)
        else:
            # Mantener valor anterior si hay hueco
            senal.append(senal[-1] if len(senal) > 0 else h_roi/2)

    # Procesamiento de señal final
    senal = np.array(senal)
    senal = senal - np.mean(senal) # Centrar
    senal = signal.medfilt(senal, kernel_size=5) # Suavizado final de picos de ruido
    senal = senal * 5 # Ganancia
    
    return senal.tolist()

# --- MOTOR DIAGNÓSTICO (Igual que antes) ---
def motor_diagnostico(ecg_signal, fs):
    distance = int(fs * 0.4) 
    height_threshold = np.max(ecg_signal) * 0.35 
    peaks, _ = signal.find_peaks(ecg_signal, height=height_threshold, distance=distance)
    
    if len(peaks) < 2:
        return 0, "RUIDO / NO LEÍBLE", "grey", 0, False

    rr_intervals = np.diff(peaks)
    mean_rr = np.mean(rr_intervals)
    std_rr = np.std(rr_intervals)
    bpm = int(60000 / (mean_rr / fs * 1000))
    cv_rr = std_rr / mean_rr 

    diagnostico = "Ritmo Sinusal"
    color = "green"
    
    if cv_rr > 0.15:
        diagnostico = "POSIBLE FA / RITMO IRREGULAR"
        color = "orange" # Bajamos a naranja por precaución en fotos
    else:
        if bpm > 100: diagnostico = "TAQUICARDIA"; color = "orange"
        elif bpm < 60: diagnostico = "BRADICARDIA"; color = "green"
        else: diagnostico = "RITMO SINUSAL NORMAL"; color = "green"
            
    return bpm, diagnostico, color, cv_rr, True

@app.route('/analizar_ecg_foto', methods=['POST'])
def analizar_ecg_foto():
    try:
        if 'file' not in request.files: return jsonify({"error": "No file"}), 400
        file = request.files['file']
        filepath = f"/tmp/{file.filename}"
        file.save(filepath)

        senal_grafica = extraer_senal_imagen(filepath)
        
        if len(senal_grafica) < 50: # Si quedó vacío tras borrar cuadrícula
             return jsonify({"status": "error", "mensaje": "No se detectó trazo negro claro."})

        # FS Estimada para foto panorámica
        bpm, dx, color, cv, exito = motor_diagnostico(np.array(senal_grafica), 250.0) 
        
        try: os.remove(filepath)
        except: pass

        return jsonify({
            "status": "success", "grid_detected": True, "mensaje": "Digitalización Completada",
            "senal_grafica": senal_grafica, "frecuencia_cardiaca": bpm,
            "diagnostico_texto": dx, "alerta_color": color, "detalles": f"Filtro Grid: ACTIVO | CV: {cv:.2f}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/analizar_holter', methods=['POST'])
def analizar_holter():
    # ... (Mismo código Holter de siempre) ...
    try:
        file = request.files['file']
        filepath = f"/tmp/{file.filename}"
        file.save(filepath)
        raw_data = np.fromfile(filepath, dtype=np.int16)
        ecg_signal = raw_data[:5000] if len(raw_data) > 5000 else raw_data
        bpm, dx, color, cv, exito = motor_diagnostico(ecg_signal, 500.0)
        try: os.remove(filepath)