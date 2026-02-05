from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import scipy.signal as signal
import os

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Cerebro Electrofisiólogo MediSumma: ACTIVO v3.0 🫀"

@app.route('/analizar_holter', methods=['POST'])
def analizar_holter():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No se envió archivo"}), 400
        
        file = request.files['file']
        filename = file.filename
        filepath = f"/tmp/{filename}"
        file.save(filepath)

        # 1. LECTURA DE SEÑAL
        # Frecuencia de muestreo típica de Holter: 250 Hz
        fs = 250.0 
        raw_data = np.fromfile(filepath, dtype=np.int16)
        
        # Analizamos un tramo representativo (10 segundos = 2500 muestras)
        # o toda la señal si es corta, para tener precisión estadística
        window_size = 5000 
        ecg_signal = raw_data[:window_size] if len(raw_data) > window_size else raw_data

        # 2. DETECCIÓN DE COMPLEJOS QRS (Algoritmo de Pan-Tompkins simplificado)
        # Usamos find_peaks con una distancia mínima para evitar contar onda T como R
        # Distancia 150 muestras (aprox 600ms) evita doble conteo en ritmos normales
        distance = int(fs * 0.4) 
        height_threshold = np.max(ecg_signal) * 0.5
        
        peaks, _ = signal.find_peaks(ecg_signal, height=height_threshold, distance=distance)
        
        # 3. CÁLCULO DE INTERVALOS R-R
        # Convertimos diferencias de índices a milisegundos
        rr_intervals_samples = np.diff(peaks)
        rr_intervals_ms = (rr_intervals_samples / fs) * 1000

        # 4. ANÁLISIS ESTADÍSTICO (Criterios Médicos)
        if len(rr_intervals_ms) < 2:
            bpm = 0
            cv_rr = 0
        else:
            mean_rr = np.mean(rr_intervals_ms)
            std_rr = np.std(rr_intervals_ms)
            bpm = int(60000 / mean_rr)
            
            # Coeficiente de Variación (CV): La medida clave para FA
            # Si CV > 0.10-0.15, es altamente sugestivo de "Irregularmente Irregular"
            cv_rr = std_rr / mean_rr

        # 5. MOTOR DE DIAGNÓSTICO
        diagnostico_texto = "Ritmo Sinusal Normal"
        alerta_color = "green"

        # Lógica de Diagnóstico
        if bpm == 0:
            diagnostico_texto = "SEÑAL INSUFICIENTE / ARTEFACTO"
            alerta_color = "grey"
        
        elif cv_rr > 0.12: # Umbral de irregularidad (12% de variación)
            # Es irregular. Ahora miramos la frecuencia para clasificar la FA
            if bpm > 100:
                diagnostico_texto = "FIBRILACIÓN AURICULAR (RVR)" # Respuesta Ventricular Rápida
                alerta_color = "red"
            elif bpm < 60:
                diagnostico_texto = "FIBRILACIÓN AURICULAR (Respuesta Lenta)"
                alerta_color = "orange"
            else:
                diagnostico_texto = "FA - RESPUESTA VENTRICULAR CONTROLADA"
                alerta_color = "orange"
        
        else:
            # Es regular. Miramos frecuencia.
            if bpm > 100:
                diagnostico_texto = "TAQUICARDIA SINUSAL"
                alerta_color = "orange"
            elif bpm < 60:
                diagnostico_texto = "BRADICARDIA SINUSAL"
                alerta_color = "green"
        
        # Preparamos datos para gráfica (diezmado para velocidad)
        senal_grafica = ecg_signal[::2].tolist() 

        # Limpieza
        try:
            os.remove(filepath)
        except:
            pass

        print(f"Dx: {diagnostico_texto} | BPM: {bpm} | CV R-R: {cv_rr:.3f}")

        return jsonify({
            "frecuencia_cardiaca": bpm,
            "diagnostico_texto": diagnostico_texto,
            "alerta_color": alerta_color,
            "senal_grafica": senal_grafica,
            "rr_stats": f"Variabilidad RR: {(cv_rr*100):.1f}%" # Dato extra para el médico
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)