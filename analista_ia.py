from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import scipy.signal as signal
import os

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Cerebro Electrofisiólogo V4 (P-Wave Detect) 🫀"

@app.route('/analizar_holter', methods=['POST'])
def analizar_holter():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No se envió archivo"}), 400
        
        file = request.files['file']
        filename = file.filename
        filepath = f"/tmp/{filename}"
        file.save(filepath)

        # 1. LECTURA
        fs = 250.0  # Hz
        raw_data = np.fromfile(filepath, dtype=np.int16)
        # Analizar hasta 10 segundos
        ecg_signal = raw_data[:5000] if len(raw_data) > 5000 else raw_data

        # 2. DETECCIÓN QRS
        distance = int(fs * 0.4) 
        height_threshold = np.max(ecg_signal) * 0.5
        peaks, _ = signal.find_peaks(ecg_signal, height=height_threshold, distance=distance)
        
        # 3. ANÁLISIS R-R (Regularidad)
        if len(peaks) < 2:
            return jsonify({"diagnostico_texto": "TRAZADO INSUFICIENTE", "alerta_color": "grey", "frecuencia_cardiaca": 0, "senal_grafica": []})

        rr_intervals = np.diff(peaks)
        mean_rr = np.mean(rr_intervals)
        std_rr = np.std(rr_intervals)
        bpm = int(60000 / (mean_rr / fs * 1000))
        cv_rr = std_rr / mean_rr # Coeficiente de Variación

        # 4. BUSQUEDA DE ONDA P (Nuevo Algoritmo) 🔎
        # Miramos una ventana de 200ms ANTES del QRS (y dejamos 40ms de buffer para no confundir la subida R)
        p_window = int(0.20 * fs) # 200ms
        buffer_r = int(0.04 * fs)  # 40ms
        
        p_waves_detected = 0
        total_beats_checked = 0

        for r_idx in peaks:
            if r_idx > p_window: # Solo si hay espacio atrás
                # Extraemos el segmento anterior al QRS
                segmento_p = ecg_signal[r_idx - p_window : r_idx - buffer_r]
                
                # Criterio: ¿Hay algún pico en ese segmento que sea al menos 5-10% del tamaño del QRS?
                # Y que no sea ruido plano
                if len(segmento_p) > 0:
                    pico_p = np.max(segmento_p) - np.min(segmento_p) # Amplitud local
                    umbral_p = (np.max(ecg_signal) * 0.05) # 5% del voltaje máximo global
                    
                    if pico_p > umbral_p:
                        p_waves_detected += 1
                
                total_beats_checked += 1
        
        # Ratio de Ondas P (Si aparece en más del 60% de latidos, asumimos que existe)
        tiene_onda_p = (p_waves_detected / total_beats_checked > 0.6) if total_beats_checked > 0 else False

        # 5. DIAGNÓSTICO INTEGRAL (Criterios AHA)
        diagnostico = "Ritmo Sinusal Normal"
        color = "green"
        es_irregular = cv_rr > 0.15 # 15% de tolerancia

        # --- ARBOL DE DECISIÓN MÉDICA ---
        
        if es_irregular:
            # R-R Irregular
            if not tiene_onda_p:
                diagnostico = "FIBRILACIÓN AURICULAR"
                color = "red"
            else:
                diagnostico = "ARRITMIA SINUSAL (P Presente)"
                color = "green"
        else:
            # R-R Regular
            if bpm > 150 and not tiene_onda_p:
                 diagnostico = "TAQUICARDIA SUPRAVENTRICULAR / FLUTTER 1:1"
                 color = "red"
            elif bpm > 100:
                 diagnostico = "TAQUICARDIA SINUSAL" if tiene_onda_p else "TAQUICARDIA (Posible Reentrada)"
                 color = "orange"
            elif bpm < 60:
                 diagnostico = "BRADICARDIA SINUSAL" if tiene_onda_p else "RITMO DE ESCAPE / JUNCTIONAL"
                 color = "orange"
            else:
                 if not tiene_onda_p:
                     diagnostico = "RITMO NODAL / AUSENCIA DE P"
                     color = "orange"
                 else:
                     diagnostico = "RITMO SINUSAL NORMAL"
                     color = "green"

        senal_grafica = ecg_signal[::2].tolist()

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