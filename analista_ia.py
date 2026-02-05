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
    return "Cerebro MediSumma v9.0: INK HUNTER (Aislamiento de Tinta) 🖤"

def procesar_tira_ritmo(path):
    try:
        img = cv2.imread(path)
        if img is None: return []

        # 1. RECORTE DE SEGURIDAD (TIRA DE RITMO)
        # Cortamos el 15% inferior de la imagen. 
        # En su foto, ahí está el DII largo. Esto elimina todo el ruido de arriba.
        h, w, _ = img.shape
        roi = img[int(h*0.85):h, :] 

        # 2. AISLAMIENTO DE TINTA (La clave v9.0)
        # Convertimos a HSV (Matiz, Saturación, Valor)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # DEFINICIÓN DE "TINTA NEGRA":
        # - Saturation (S): Bajo (0 a 80) -> No tiene color (evita la cuadrícula rosada/roja)
        # - Value (V): Bajo (0 a 130) -> Es oscuro (evita el papel blanco y sombras claras)
        
        lower_ink = np.array([0, 0, 0])      # Negro absoluto
        upper_ink = np.array([180, 80, 130]) # Gris oscuro sin color
        
        mask = cv2.inRange(hsv, lower_ink, upper_ink)

        # 3. LIMPIEZA MORFOLÓGICA
        # Conectamos los puntos. Si la tinta se cortó, esto la une.
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.erode(mask, kernel, iterations=1)

        # 4. DIGITALIZACIÓN ROBUSTA
        # En lugar de escanear columna por columna ciegamente, buscamos el "centro de masa" de la tinta
        senal = []
        h_roi, w_roi = mask.shape
        
        x_points = []
        y_points = []

        for x in range(w_roi):
            col = mask[:, x]
            indices = np.where(col > 0)[0] # Píxeles de tinta
            
            if len(indices) > 0:
                # Tomamos la mediana vertical de la tinta en esa columna
                # Invertimos el eje Y para que coincida con la gráfica
                val = h_roi - np.median(indices)
                y_points.append(val)
                x_points.append(x)
        
        if len(y_points) < 50: return [] # No se encontró tinta

        # 5. INTERPOLACIÓN LINEAL (Rellena huecos donde la tinta falló)
        all_x = np.arange(w_roi)
        if len(x_points) > 1:
            senal_completa = np.interp(all_x, x_points, y_points)
        else:
            return []

        # 6. FILTRADO FINAL (Suavizado médico)
        # Elimina la línea base errante
        senal_final = signal.detrend(senal_completa)
        # Filtro Savitzky-Golay para suavizar los bordes pixelados
        try:
            senal_final = signal.savgol_filter(senal_final, 21, 3)
        except:
            pass # Si falla el filtro (muy corta señal), se deja cruda
        
        return (senal_final * 5).tolist()

    except Exception as e:
        print(f"Error v9: {str(e)}")
        return []

def diagnosticar(senal):
    if len(senal) < 100: return 0, "ERROR: IMAGEN MUY BORROSA", "grey"
    
    fs = 250.0 # Estimado estándar para fotos
    
    # Buscamos picos QRS prominentes
    peaks, _ = signal.find_peaks(senal, height=np.max(senal)*0.45, distance=int(fs*0.4))
    
    if len(peaks) < 2: return 0, "RUIDO / NO LEÍBLE", "grey"

    rr = np.diff(peaks)
    bpm = int(60000 / (np.mean(rr) / fs * 1000))
    cv = np.std(rr) / np.mean(rr)

    dx = "RITMO SINUSAL"
    color = "green"

    # Lógica simple y efectiva
    if cv > 0.15: 
        dx = "ARRITMIA (Irregular)"
        color = "orange"
    elif bpm > 100: 
        dx = "TAQUICARDIA"
        color = "orange"
    elif bpm < 60: 
        dx = "BRADICARDIA"
        color = "green"

    return bpm, dx, color

@app.route('/analizar_ecg_foto', methods=['POST'])
def analizar_ecg_foto():
    try:
        f = request.files['file']
        path = f"/tmp/{f.filename}"
        f.save(path)

        grafica = procesar_tira_ritmo(path)
        
        if not grafica:
             # Si falla, devolvemos error controlado
             return jsonify({
                "senal_grafica": [],
                "frecuencia_cardiaca": 0,
                "diagnostico_texto": "NO SE DETECTÓ TINTA NEGRA",
                "alerta_color": "grey",
                "mensaje": "Intente con mejor iluminación o enfoque."
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
            "mensaje": "Lectura DII Largo OK"
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
        bpm, dx, color = diagnosticar(sig)
        return jsonify({"frecuencia_cardiaca": bpm, "diagnostico_texto": dx, "alerta_color": color, "senal_grafica": sig.tolist()})
    except: return jsonify({"error": "Holter error"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)