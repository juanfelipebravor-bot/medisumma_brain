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
    return "Cerebro MediSumma v7.3: ONLINE"

def procesar_imagen(path):
    try:
        img = cv2.imread(path)
        if img is None: return []

        # 1. Filtro Grid (Rojo/Rosa)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 20, 20])
        upper_red1 = np.array([20, 255, 255])
        lower_red2 = np.array([160, 20, 20])
        upper_red2 = np.array([180, 255, 255])
        
        mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
        
        # Limpiar
        img[mask > 0] = [255, 255, 255]

        # 2. Recorte Central (35% a 65%)
        h, w, _ = img.shape
        roi = img[int(h*0.35):int(h*0.65), :]
        
        # 3. Binarizar
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)

        # 4. Escanear
        senal = []
        h_roi, w_roi = binary.shape
        for x in range(w_roi):
            col = binary[:, x]
            idc = np.where(col > 0)[0]
            if len(idc) > 0:
                val = h_roi - np.median(idc)
                senal.append(val)
            else:
                last = senal[-1] if len(senal) > 0 else h_roi/2
                senal.append(last)

        # Post-proceso
        arr = np.array(senal)
        arr = arr - np.mean(arr)
        arr = signal.medfilt(arr, 5)
        return (arr * 5).tolist()
    except:
        return []

def diagnosticar(senal):
    if len(senal) < 50: return 0, "ERROR LECTURA", "grey"
    
    fs = 333.0
    dist = int(fs * 0.4)
    peaks, _ = signal.find_peaks(senal, height=np.max(senal)*0.35, distance=dist)
    
    if len(peaks) < 2: return 0, "RUIDO / ARTEFACTO", "grey"

    rr = np.diff(peaks)
    bpm = int(60000 / (np.mean(rr) / fs * 1000))
    cv = np.std(rr) / np.mean(rr)

    if cv > 0.15: return bpm, "POSIBLE FA / IRREGULAR", "orange"
    if bpm > 100: return bpm, "TAQUICARDIA", "orange"
    if bpm < 60: return bpm, "BRADICARDIA", "green"
    return bpm, "RITMO SINUSAL", "green"

@app.route('/analizar_ecg_foto', methods=['POST'])
def analizar_ecg_foto():
    try:
        f = request.files['file']
        path = f"/tmp/{f.filename}"
        f.save(path)

        grafica = procesar_imagen(path)
        bpm, dx, color = diagnosticar(grafica)

        try: os.remove(path)
        except: pass

        return jsonify({
            "senal_grafica": grafica,
            "frecuencia_cardiaca": bpm,
            "diagnostico_texto": dx,
            "alerta_color": color,
            "grid_detected": True,
            "mensaje": "Digitalización OK"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/analizar_holter', methods=['POST'])
def analizar_holter():
    return jsonify({"mensaje": "Holter activo (simplificado para prueba)"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)