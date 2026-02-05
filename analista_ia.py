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
    return "Cerebro MediSumma: OIDOS (Holter) y OJOS (Vision) ACTIVOS v6.0 👁️🫀"

# --- MÓDULO 1: VISION ARTIFICIAL (NUEVO) ---
@app.route('/analizar_ecg_foto', methods=['POST'])
def analizar_ecg_foto():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No se envió imagen"}), 400
        
        file = request.files['file']
        filename = file.filename
        filepath = f"/tmp/{filename}"
        file.save(filepath)

        # 1. Leer la imagen con OpenCV
        img = cv2.imread(filepath)
        
        if img is None:
            return jsonify({"error": "El archivo no es una imagen válida"}), 400

        # 2. Pre-procesamiento: Detectar si es papel ECG (Busqueda de tonos Rojos/Rosados)
        # Convertimos a formato HSV (Matiz, Saturación, Valor) para filtrar colores
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Definir rango de color rosado/rojo (típico de la cuadrícula)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_grid = mask1 + mask2

        # Contamos cuántos píxeles de "cuadrícula" encontramos
        grid_pixels = cv2.countNonZero(mask_grid)
        total_pixels = img.shape[0] * img.shape[1]
        grid_ratio = grid_pixels / total_pixels

        # 3. Lógica Inicial de Validación
        calidad = "Imagen recibida correctamente"
        papel_detectado = False

        if grid_ratio > 0.01: # Si más del 1% es rojizo/rosado
            calidad = "Papel Milimetrado Detectado. Calidad: Aceptable."
            papel_detectado = True
        else:
            calidad = "ADVERTENCIA: No se detecta cuadrícula estándar. ¿Es una foto B/N?"

        # Limpieza
        try:
            os.remove(filepath)
        except:
            pass

        # Por ahora, devolvemos el análisis técnico de la imagen
        return jsonify({
            "status": "success",
            "mensaje": calidad,
            "grid_detected": papel_detectado,
            "dimensiones": f"{img.shape[1]}x{img.shape[0]} px",
            "diagnostico_preliminar": "Módulo de Digitalización listo para extracción de señal."
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- MÓDULO 2: HOLTER DIGITAL (YA CALIBRADO) ---
@app.route('/analizar_holter', methods=['POST'])
def analizar_holter():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No se envió archivo"}), 400
        
        file = request.files['file']
        filename = file.filename
        filepath = f"/tmp/{filename}"
        file.save(filepath)

        fs = 500.0 
        raw_data = np.fromfile(filepath, dtype=np.int16)
        ecg_signal = raw_data[:5000] if len(raw_data) > 5000 else raw_data

        distance = int(fs * 0.4) 
        height_threshold = np.max(ecg_signal) * 0.5
        peaks, _ = signal.find_peaks(ecg_signal, height=height_threshold, distance=distance)
        
        if len(peaks) < 2:
            return jsonify({"diagnostico_texto": "TRAZADO INSUFICIENTE", "alerta_color": "grey", "frecuencia_cardiaca": 0, "senal_grafica": []})

        rr_intervals = np.diff(peaks)
        mean_rr = np.mean(rr_intervals)
        std_rr = np.std(rr_intervals)
        bpm = int(60000 / (mean_rr / fs * 1000))
        cv_rr = std_rr / mean_rr 

        p_window = int(0.20 * fs) 
        buffer_r = int(0.04 * fs)
        p_waves_detected = 0
        total_beats_checked = 0

        for r_idx in peaks:
            if r_idx > p_window:
                segmento_p = ecg_signal[r_idx - p_window : r_idx - buffer_r]
                if len(segmento_p) > 0:
                    pico_p = np.max(segmento_p) - np.min(segmento_p)
                    umbral_p = (np.max(ecg_signal) * 0.03) 
                    if pico_p > umbral_p:
                        p_waves_detected += 1
                total_beats_checked += 1
        
        tiene_onda_p = (p_waves_detected / total_beats_checked > 0.5) if total_beats_checked > 0 else False

        diagnostico = "Ritmo Sinusal Normal"
        color = "green"
        es_irregular = cv_rr > 0.15 

        if es_irregular:
            if not tiene_onda_p:
                diagnostico = "FIBRILACIÓN AURICULAR"
                color = "red"
            else:
                diagnostico = "ARRITMIA SINUSAL"
                color = "green"
        else:
            if bpm > 150 and not tiene_onda_p:
                 diagnostico = "TAQUICARDIA SUPRAVENTRICULAR"
                 color = "red"
            elif bpm > 100:
                 diagnostico = "TAQUICARDIA SINUSAL" if tiene_onda_p else "TAQUICARDIA (Posible Reentrada)"
                 color = "orange"
            elif bpm < 60:
                 diagnostico = "BRADICARDIA SINUSAL"
                 color = "green"
            else:
                 diagnostico = "RITMO SINUSAL NORMAL" if tiene_onda_p else "RITMO DE LA UNIÓN"
                 color = "green"

        senal_grafica = ecg_signal.tolist()

        return jsonify({
            "frecuencia_cardiaca": bpm,
            "diagnostico_texto": diagnostico,
            "alerta_color": color,
            "senal_grafica": senal_grafica,
            "detalles": f"P-Wave: {'SI' if tiene_onda_p else 'NO'} | CV: {cv_rr:.2f}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)