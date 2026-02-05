from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import scipy.signal as signal
import os

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Cerebro Electrofisiólogo V5 (Hi-Res 500Hz) 🫀"

@app.route('/analizar_holter', methods=['POST'])
def analizar_holter():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No se envió archivo"}), 400
        
        file = request.files['file']
        filename = file.filename
        filepath = f"/tmp/{filename}"
        file.save(filepath)

        # 1. LECTURA (CORRECCIÓN CLÍNICA)
        # Ajustado a 500 Hz para coincidir con la realidad de 5 cuadros = 60 lpm
        fs = 500.0 
        raw_data = np.fromfile(filepath, dtype=np.int16)
        
        # Analizamos 10 segundos para mayor precisión
        ecg_signal = raw_data[:5000] if len(raw_data) > 5000 else raw_data

        # 2. DETECCIÓN QRS
        distance = int(fs * 0.4) 
        height_threshold = np.max(ecg_signal) * 0.5
        peaks, _ = signal.find_peaks(ecg_signal, height=height_threshold, distance=distance)
        
        # 3. ANÁLISIS R-R
        if len(peaks) < 2:
            return jsonify({"diagnostico_texto": "TRAZADO INSUFICIENTE", "alerta_color": "grey", "frecuencia_cardiaca": 0, "senal_grafica": []})

        rr_intervals = np.diff(peaks)
        mean_rr = np.mean(rr_intervals)
        std_rr = np.std(rr_intervals)
        
        # Cálculo BPM corregido
        bpm = int(60000 / (mean_rr / fs * 1000))
        cv_rr = std_rr / mean_rr 

        # 4. BUSQUEDA DE ONDA P (Ventana ajustada a 500Hz)
        # 200ms = 100 puntos (a 500Hz)
        p_window = int(0.20 * fs) 
        buffer_r = int(0.04 * fs) # 20 puntos
        
        p_waves_detected = 0
        total_beats_checked = 0

        for r_idx in peaks:
            if r_idx > p_window:
                segmento_p = ecg_signal[r_idx - p_window : r_idx - buffer_r]
                if len(segmento_p) > 0:
                    pico_p = np.max(segmento_p) - np.min(segmento_p)
                    # Umbral más sensible para detectar P pequeñas
                    umbral_p = (np.max(ecg_signal) * 0.03) 
                    
                    if pico_p > umbral_p:
                        p_waves_detected += 1
                total_beats_checked += 1
        
        tiene_onda_p = (p_waves_detected / total_beats_checked > 0.5) if total_beats_checked > 0 else False

        # 5. DIAGNÓSTICO INTEGRAL
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
                 color = "green" # Verde porque suele ser fisiológica en atletas
            else:
                 diagnostico = "RITMO SINUSAL NORMAL" if tiene_onda_p else "RITMO DE LA UNIÓN"
                 color = "green"

        # IMPORTANTE: Enviamos la señal COMPLETA (sin '::2') para que se vea la Onda P
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