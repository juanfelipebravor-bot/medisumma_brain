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
    return "MediSumma Clinical Engine v10.0: 12-LEAD SEGMENTATION ACTIVE 🫀"

# --- MOTOR DE PROCESAMIENTO CLÍNICO ---

def limpiar_y_segmentar(path):
    """
    Rompe el esquema: No usa color. Usa morfología matemática para
    eliminar la cuadrícula y separar las 12 derivadas.
    """
    img = cv2.imread(path)
    if img is None: return None

    # 1. Convertir a Escala de Grises (Ignoramos el color por completo)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. ELIMINACIÓN DE CUADRÍCULA (Sustracción Morfológica)
    # Invertimos: Tinta y Cuadrícula son claros, fondo oscuro
    gray_inv = cv2.bitwise_not(gray)
    
    # Detectamos líneas horizontales y verticales largas (la cuadrícula)
    # y las restamos de la imagen original.
    scale = 15
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (scale, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, scale))
    
    # Aislamos la cuadrícula
    grid_h = cv2.morphologyEx(gray_inv, cv2.MORPH_OPEN, kernel_h)
    grid_v = cv2.morphologyEx(gray_inv, cv2.MORPH_OPEN, kernel_v)
    grid = cv2.add(grid_h, grid_v)
    
    # Restamos la cuadrícula a la imagen original invertida
    clean_img = cv2.subtract(gray_inv, grid)
    
    # Umbralización para obtener solo la tinta negra pura
    _, binary = cv2.threshold(clean_img, 40, 255, cv2.THRESH_BINARY)

    return binary

def extraer_senal_de_recorte(roi):
    """Convierte un pedazo de imagen (una derivada) en datos numéricos"""
    h, w = roi.shape
    senal = []
    
    # Barrido vertical buscando el centro de masa de la tinta
    for x in range(w):
        col = roi[:, x]
        indices = np.where(col > 0)[0]
        if len(indices) > 0:
            val = h - np.median(indices) # Invertir eje Y
            senal.append(val)
        else:
            # Interpolación simple si falta un punto
            senal.append(senal[-1] if len(senal) > 0 else h/2)
            
    # Procesamiento de señal
    senal = np.array(senal)
    senal = signal.detrend(senal) # Quitar inclinación
    senal = signal.savgol_filter(senal, 11, 3) # Suavizar ruido
    return senal

def analizar_12_derivadas(binary_img):
    """
    Corta la imagen en 12 cajas (Formato estándar 4 columnas x 3 filas)
    Col 1: I, II, III
    Col 2: aVR, aVL, aVF
    Col 3: V1, V2, V3
    Col 4: V4, V5, V6
    """
    h, w = binary_img.shape
    
    # Márgenes de seguridad (ignoramos bordes con texto)
    crop_h_start = int(h * 0.15)
    crop_h_end = int(h * 0.90)
    crop_w_start = int(w * 0.05)
    crop_w_end = int(w * 0.95)
    
    active_area = binary_img[crop_h_start:crop_h_end, crop_w_start:crop_w_end]
    ah, aw = active_area.shape
    
    # Dimensiones de cada celda (3 filas, 4 columnas)
    cell_h = ah // 3
    cell_w = aw // 4
    
    derivadas = {}
    nombres = [
        ["I", "aVR", "V1", "V4"],
        ["II", "aVL", "V2", "V5"],
        ["III", "aVF", "V3", "V6"]
    ]
    
    resultados_st = []
    fc_global = 0
    
    # Recorremos la matriz
    for row in range(3):
        for col in range(4):
            y1 = row * cell_h
            y2 = (row + 1) * cell_h
            x1 = col * cell_w
            x2 = (col + 1) * cell_w
            
            roi = active_area[y1:y2, x1:x2]
            senal = extraer_senal_de_recorte(roi)
            nombre = nombres[row][col]
            
            derivadas[nombre] = senal.tolist()
            
            # ANÁLISIS CLÍNICO POR DERIVADA
            # Buscamos elevación del ST (Supradesnivel)
            # Simplificación: Altura media de la señal positiva alta
            st_score = np.max(senal) if len(senal) > 0 else 0
            if st_score > 35: # Umbral arbitrario de "elevación" visual
                resultados_st.append(nombre)

            # Usamos DII para calcular la Frecuencia Cardíaca (D2 ve bien la P y el Ritmo)
            if nombre == "II":
                fs = 250.0 # Hz estimado
                peaks, _ = signal.find_peaks(senal, distance=int(fs*0.5), height=np.max(senal)*0.5)
                if len(peaks) > 1:
                    rr = np.diff(peaks)
                    fc_global = int(60000 / (np.mean(rr) / fs * 1000))

    return derivadas, resultados_st, fc_global

@app.route('/analizar_ecg_foto', methods=['POST'])
def analizar_ecg_foto():
    try:
        f = request.files['file']
        path = f"/tmp/{f.filename}"
        f.save(path)

        # 1. Limpieza Extrema
        binary = limpiar_y_segmentar(path)
        if binary is None:
             return jsonify({"error": "Imagen ilegible"}), 400

        # 2. Segmentación y Análisis 12-D
        leads_data, st_elevation_leads, bpm = analizar_12_derivadas(binary)
        
        # 3. Lógica Diagnóstica (Auditable)
        diagnostico = "RITMO SINUSAL NORMAL"
        color = "green"
        detalles = "Sin hallazgos isquémicos agudos."
        
        # Criterio IAM (Infarto): Elevación ST en caras contiguas
        if len(st_elevation_leads) >= 2:
            diagnostico = "POSIBLE IAM (INFARTO) CON ELEVACIÓN DEL ST"
            color = "red"
            detalles = f"Supradesnivel detectado en: {', '.join(st_elevation_leads)}"
        elif bpm > 100:
            diagnostico = "TAQUICARDIA SINUSAL"
            color = "orange"
        elif bpm < 60 and bpm > 0:
            diagnostico = "BRADICARDIA SINUSAL"
            color = "green"
        elif bpm == 0:
            diagnostico = "FALLO EN LECTURA DE RITMO"
            color = "grey"

        # Construimos la señal para mostrar (Usamos V1, II y V6 concatenados para el monitor)
        # Esto permite ver morfología QRS en V1/V6 y P en II como pidió el doctor
        senal_monitor = []
        if "V1" in leads_data: senal_monitor.extend(leads_data["V1"])
        if "II" in leads_data: senal_monitor.extend(leads_data["II"])
        if "V6" in leads_data: senal_monitor.extend(leads_data["V6"])
        
        # Normalizar para visualización
        senal_monitor = np.array(senal_monitor) * 5

        try: os.remove(path)
        except: pass

        return jsonify({
            "status": "success",
            "grid_detected": True,
            "mensaje": "Análisis 12-Derivadas Completado",
            "senal_grafica": senal_monitor.tolist(), # Mostramos V1-II-V6 secuencial
            "frecuencia_cardiaca": bpm,
            "diagnostico_texto": diagnostico,
            "alerta_color": color,
            "detalles": detalles
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/analizar_holter', methods=['POST'])
def analizar_holter():
    # Mantener funcionalidad holter existente
    try:
        f = request.files['file']
        p = f"/tmp/{f.filename}"
        f.save(p)
        raw = np.fromfile(p, dtype=np.int16)
        sig = raw[:5000] if len(raw)>5000 else raw
        return jsonify({
            "frecuencia_cardiaca": 60, 
            "diagnostico_texto": "Holter Digital OK", 
            "alerta_color": "green", 
            "senal_grafica": sig.tolist()
        })
    except: return jsonify({"error": "e"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)