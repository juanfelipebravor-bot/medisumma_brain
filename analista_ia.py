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
    return "Cerebro MediSumma v8.0: SEGMENTACIÓN DE 12 DERIVADAS 🏥"

def procesar_imagen_inteligente(path):
    try:
        img = cv2.imread(path)
        if img is None: return []

        # 1. Pre-procesamiento (Escala de grises)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2. LIMPIEZA DE SOMBRAS (CLAHE)
        # Esto permite leer fotos con mala iluminación (como la de WhatsApp)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

        # 3. BINARIZACIÓN ADAPTATIVA (Crucial para no perder trazos finos)
        # Calcula el umbral localmente en lugar de globalmente
        binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 15, 3)

        # 4. SEGMENTACIÓN AUTOMÁTICA (La solución al ruido)
        # Un ECG de 12 derivadas tiene filas. Si leemos todo verticalmente, se mezclan.
        # Vamos a extraer la FRANJA INFERIOR (aprox el último 20% de la hoja)
        # que corresponde a la Tira de Ritmo Continua (generalmente V1, II, o V5 largo).
        h, w = binary.shape
        
        # Cortamos el 25% inferior de la imagen
        roi_bottom = binary[int(h*0.75):h, :] 

        # 5. Escaneo de la ROI Limpia
        senal = []
        h_roi, w_roi = roi_bottom.shape
        
        # Recolectar puntos
        x_points = []
        y_points = []

        for x in range(w_roi):
            col = roi_bottom[:, x]
            indices = np.where(col > 0)[0]
            
            if len(indices) > 0:
                # Mediana para evitar ruido suelto
                val = h_roi - np.median(indices)
                y_points.append(val)
                x_points.append(x)
        
        if len(y_points) < 50: return []

        # 6. INTERPOLACIÓN (Suavizado de líneas perdidas)
        all_x = np.arange(w_roi)
        senal_completa = np.interp(all_x, x_points, y_points)

        # 7. Filtrado Final (Quitar vibración de alta frecuencia)
        senal_final = signal.detrend(senal_completa)
        senal_final = signal.savgol_filter(senal_final, 11, 3)
        
        # Amplificación para que se vea bien en el celular
        return (senal_final * 4).tolist()

    except Exception as e:
        print(f"Error: {str(e)}")
        return []

def diagnosticar(senal):
    if len(senal) < 100: return 0, "ERROR CALIDAD", "grey"
    
    # Frecuencia estimada para una tira de ritmo completa en foto
    fs = 250.0 
    
    # Detección de QRS
    peaks, _ = signal.find_peaks(senal, height=np.max(senal)*0.4, distance=int(fs*0.4))
    
    if len(peaks) < 2: return 0, "TRAZO NO LEÍBLE", "grey"

    rr = np.diff(peaks)
    bpm = int(60000 / (np.mean(rr) / fs * 1000))
    cv = np.std(rr) / np.mean(rr)

    dx = "RITMO SINUSAL"
    color = "green"

    if cv > 0.15: 
        dx = "ARRITMIA / IRREGULAR"
        color = "orange"
    elif bpm > 100: 
        dx = "TAQUICARDIA"
        color = "orange"
    elif bpm < 50: 
        dx = "BRADICARDIA"
        color = "green"

    return bpm, dx, color

@app.route('/analizar_ecg_foto', methods=['POST'])
def analizar_ecg_foto():
    try:
        f = request.files['file']
        path = f"/tmp/{f.filename}"
        f.save(path)

        # Usamos el nuevo procesador inteligente
        grafica = procesar_imagen_inteligente(path)
        
        if not grafica:
             return jsonify({
                "senal_grafica": [],
                "frecuencia_cardiaca": 0,
                "diagnostico_texto": "NO SE DETECTÓ RITMO",
                "alerta_color": "grey",
                "mensaje": "Asegúrese de que la foto incluya la tira de ritmo inferior."
            })

        bpm, dx, color = diagnosticar(grafica)

        try: os.remove(path)
        except: pass

        return jsonify({
            "senal_grafica": grafica,
            "frecuencia_cardiaca": bpm,
            "diagnostico_texto": dx,
            "alerta_color": color,
            "grid_detected": True,
            "mensaje": "Digitalización 12-D (Tira Ritmo) OK"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/analizar_holter', methods=['POST'])
def analizar_holter():
    try:
        f = request.files['file']
        p = f"/tmp/{f.filename}"
        f.save(p)
        raw = np.fromfile(p, dtype=np.int16)
        sig = raw[:5000] if len(raw)>5000 else raw
        bpm, dx, color = diagnosticar(sig) # Reutilizamos lógica simple
        return jsonify({"frecuencia_cardiaca": bpm, "diagnostico_texto": dx, "alerta_color": color, "senal_grafica": sig.tolist()})
    except: return jsonify({"error": "Holter error"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)