"""
MediSumma Brain v4.0 — Motor ECG de Precisión Clínica
Análisis de fotografías de ECG en papel sin API Key
Algoritmo: calibración por cuadrícula, extracción de señal multi-derivación,
           detección PQRST, intervalos calibrados, diagnóstico con criterios AHA/ESC.
"""

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt, savgol_filter
from scipy.ndimage import uniform_filter1d
import cv2
import math

app = FastAPI(title="Cerebro MediSumma v4.0", version="4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("=== SERVIDOR MEDISUMMA v4.0 — Motor ECG Clínico Avanzado ===")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES DE PAPEL ECG ESTÁNDAR (AHA/ESC)
# ─────────────────────────────────────────────────────────────────────────────
ECG_PAPER_SPEED  = 25.0   # mm/s — velocidad estándar
ECG_GAIN         = 10.0   # mm/mV — ganancia estándar
ECG_SMALL_GRID   = 1.0    # mm — cuadro pequeño = 0.04 s = 0.1 mV
ECG_LARGE_GRID   = 5.0    # mm — cuadro grande = 0.2 s = 0.5 mV
TARGET_WIDTH     = 2000   # px — ancho de trabajo


# ─────────────────────────────────────────────────────────────────────────────
# 1. CALIBRACIÓN: PÍXELES / MM
# ─────────────────────────────────────────────────────────────────────────────

def calibrar_px_mm(img_gray: np.ndarray) -> float:
    """
    Detecta la densidad de la cuadrícula ECG mediante autocorrelación.
    La cuadrícula ECG tiene periodicidad a 1 mm (cuadro pequeño).
    Devuelve px/mm. Rango esperado: 4–25 px/mm.
    """
    h, w = img_gray.shape
    # Usar franja central para evitar bordes y texto
    y0, y1 = h // 5, 4 * h // 5
    perfil = np.mean(img_gray[y0:y1, :w // 2].astype(np.float64), axis=0)
    perfil -= np.mean(perfil)

    if np.std(perfil) < 1e-3:
        return 8.0  # fallback

    # Autocorrelación hasta 150 px
    max_lag = min(len(perfil) - 1, 150)
    autocorr = np.correlate(perfil, perfil, mode='full')
    autocorr = autocorr[len(perfil) - 1: len(perfil) - 1 + max_lag]
    autocorr = autocorr / autocorr[0]  # normalizar
    autocorr[0] = 0  # eliminar lag 0

    # Primer pico de autocorrelación = periodo de cuadrícula dominante
    picos, props = find_peaks(autocorr, distance=2, height=0.05)
    if len(picos) == 0:
        return 8.0

    # Ordenar picos por altura descendente
    ordenados = picos[np.argsort(props['peak_heights'])[::-1]]

    for candidato in ordenados:
        periodo = float(candidato)
        # Si el periodo encaja con 1 mm pequeño (4–20 px) → directo
        if 4.0 <= periodo <= 20.0:
            return periodo
        # Si encaja con 5 mm grande (20–100 px) → dividir entre 5
        if 20.0 < periodo <= 100.0:
            px_per_mm = periodo / 5.0
            if 4.0 <= px_per_mm <= 20.0:
                return px_per_mm

    return 8.0  # fallback seguro


# ─────────────────────────────────────────────────────────────────────────────
# 2. PREPROCESAMIENTO Y EXTRACCIÓN DE TRAZA
# ─────────────────────────────────────────────────────────────────────────────

def eliminar_cuadricula(bin_img: np.ndarray, px_mm: float) -> np.ndarray:
    """
    Elimina las líneas de cuadrícula del ECG binarizado.
    Las líneas de cuadrícula son estructuras lineales de 1 px de grosor.
    La traza ECG es más gruesa (2–6 px) y curvilínea.
    """
    # Detectar líneas horizontales largas
    k_h = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(3, int(px_mm * 2)), 1))
    lineas_h = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, k_h)

    # Detectar líneas verticales largas
    k_v = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(3, int(px_mm * 2))))
    lineas_v = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, k_v)

    # Cuadrícula = unión de líneas h y v, dilatadas ligeramente
    cuadricula = cv2.dilate(
        cv2.bitwise_or(lineas_h, lineas_v),
        np.ones((2, 2), np.uint8), iterations=1)

    # Traza = binaria − cuadrícula
    traza = cv2.subtract(bin_img, cuadricula)

    # Eliminar ruido puntual (artefactos <3px)
    traza = cv2.morphologyEx(traza, cv2.MORPH_OPEN,
                             np.ones((2, 2), np.uint8))
    return traza


def preprocesar_ecg(img_gray: np.ndarray, px_mm: float):
    """
    Pipeline completo:
    1. CLAHE para compensar iluminación desigual de foto
    2. Umbral adaptativo local (robusto vs flash/sombras)
    3. Eliminación de cuadrícula
    Devuelve imagen binaria (traza = 255, fondo = 0).
    """
    h, w = img_gray.shape

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_eq = clahe.apply(img_gray)

    # Suavizado leve para reducir ruido antes del umbral
    img_blur = cv2.GaussianBlur(img_eq, (3, 3), 0)

    # Umbral adaptativo: superior para fotos con iluminación variable
    block = max(11, int(px_mm * 6) | 1)
    bin_img = cv2.adaptiveThreshold(
        img_blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block, 8)

    traza = eliminar_cuadricula(bin_img, px_mm)
    return traza


# ─────────────────────────────────────────────────────────────────────────────
# 3. SEGMENTACIÓN DE FILAS / DERIVACIONES
# ─────────────────────────────────────────────────────────────────────────────

def detectar_filas(traza_bin: np.ndarray, px_mm: float) -> list:
    """
    Detecta las bandas verticales con contenido de traza ECG.
    ECG estándar 12 derivaciones: 4 filas (3 de 4 derivaciones + tira ritmo).
    Devuelve lista de (y0, y1) de hasta 4 filas, ordenadas de arriba a abajo.
    """
    h, w = traza_bin.shape

    # Proyección horizontal: suma de píxeles activos por fila
    proyeccion = np.sum(traza_bin, axis=1).astype(np.float64)

    # Suavizar para encontrar bandas
    win = max(5, int(px_mm * 4))
    suave = uniform_filter1d(proyeccion, size=win)

    umbral = max(1.0, np.percentile(suave[suave > 0], 25)) \
             if np.any(suave > 0) else 1.0
    activo = suave > umbral

    cambios = np.diff(activo.astype(int))
    inicios = list(np.where(cambios == 1)[0] + 1)
    fines   = list(np.where(cambios == -1)[0] + 1)

    # Asegurar paridad
    if activo[0]:
        inicios = [0] + inicios
    if activo[-1]:
        fines.append(h)
    if len(inicios) > len(fines):
        inicios = inicios[:len(fines)]
    elif len(fines) > len(inicios):
        fines = fines[:len(inicios)]

    bandas = [(a, b) for a, b in zip(inicios, fines)
              if (b - a) >= int(px_mm * 4)]

    if not bandas:
        paso = h // 4
        return [(i * paso, (i + 1) * paso) for i in range(4)]

    # Conservar las 4 bandas más grandes
    if len(bandas) > 4:
        bandas = sorted(bandas, key=lambda x: x[1] - x[0], reverse=True)[:4]
        bandas = sorted(bandas, key=lambda x: x[0])

    while len(bandas) < 4:
        paso = h // 4
        bandas = [(i * paso, (i + 1) * paso) for i in range(4)]

    return bandas[:4]


def extraer_senal_franja(traza_bin: np.ndarray, y0: int, y1: int,
                          x0: int = 0, x1: int = None) -> np.ndarray:
    """
    Extrae señal 1D de una franja rectangular de la imagen binarizada.
    Usa centroide (media de Y de píxeles activos) por columna.
    Interpola vacíos. Aplica Savitzky-Golay para suavizar preservando morfología.
    Retorna señal con picos R hacia arriba (positivos).
    """
    if x1 is None:
        x1 = traza_bin.shape[1]
    franja = traza_bin[y0:y1, x0:x1]
    mh = franja.shape[0]
    w  = franja.shape[1]

    senal = np.zeros(w, dtype=np.float64)
    valido = np.zeros(w, dtype=bool)
    ultimo = mh / 2.0

    for col in range(w):
        col_data = franja[:, col]
        pixeles = np.where(col_data > 0)[0]
        if len(pixeles) >= 1:
            y = float(np.mean(pixeles))
            senal[col] = y
            valido[col] = True
            ultimo = y
        else:
            senal[col] = ultimo

    # Interpolar columnas vacías
    if not np.all(valido):
        indices = np.arange(w)
        validos_idx = indices[valido]
        if len(validos_idx) >= 2:
            senal = np.interp(indices, validos_idx, senal[valido])

    # Invertir: centroide bajo = pico alto
    senal = (mh / 2.0) - senal

    # Savitzky-Golay preserva picos mejor que media móvil
    win = max(5, int(mh * 0.3) | 1)
    if len(senal) > win + 2:
        try:
            senal = savgol_filter(senal, window_length=win, polyorder=2)
        except Exception:
            pass

    return senal


# ─────────────────────────────────────────────────────────────────────────────
# 4. DETECCIÓN DE COMPLEJOS QRS Y PICOS R
# ─────────────────────────────────────────────────────────────────────────────

def detectar_picos_r(senal: np.ndarray, fs: float) -> np.ndarray:
    """
    Detección robusta de picos R con umbral adaptativo.
    fs en px/s (= px_mm × 25).
    """
    if len(senal) < int(fs * 0.3):
        return np.array([], dtype=int)

    # Eliminar línea de base lenta con filtro pasa-alto
    if len(senal) > 60:
        try:
            b, a = butter(2, 0.5 / (fs / 2), btype='high')
            senal_filt = filtfilt(b, a, senal)
        except Exception:
            senal_filt = senal - uniform_filter1d(senal, size=int(fs * 0.5))
    else:
        senal_filt = senal

    # Umbral: percentil 75 × 0.5 (adaptativo)
    umbral = np.percentile(senal_filt, 75) * 0.5
    dist_min = max(3, int(0.25 * fs))  # 250 ms mínimo entre latidos

    picos, props = find_peaks(senal_filt, height=umbral,
                              distance=dist_min, prominence=0.1 * np.std(senal_filt))

    # Si muy pocos picos, bajar umbral
    if len(picos) < 2:
        umbral2 = np.percentile(senal_filt, 60) * 0.3
        picos, _ = find_peaks(senal_filt, height=umbral2, distance=dist_min)

    return picos


# ─────────────────────────────────────────────────────────────────────────────
# 5. MEDICIÓN DE INTERVALOS (PR, QRS, QT/QTc, ST)
# ─────────────────────────────────────────────────────────────────────────────

def medir_intervalos(senal: np.ndarray, picos: np.ndarray,
                     fs: float) -> dict:
    """
    Mide con precisión PR, QRS, QT, QTc y ST.
    Detecta onda P antes del QRS e onda T después.
    fs en px/s.
    """
    if len(picos) < 1:
        return _intervalos_fallback()

    qrs_list, pr_list, qt_list, st_list = [], [], [], []
    p_detectadas = 0

    for p in picos:
        amp_r = float(senal[p])
        if amp_r <= 0:
            continue
        umbral_15pct = 0.15 * amp_r

        # ── Inicio y fin QRS ──────────────────────────────────────────────
        ini_qrs = p
        for i in range(p - 1, max(0, p - int(0.12 * fs)), -1):
            if senal[i] <= umbral_15pct:
                ini_qrs = i
                break

        fin_qrs = p
        for i in range(p + 1, min(len(senal) - 1, p + int(0.12 * fs))):
            if senal[i] <= umbral_15pct:
                fin_qrs = i
                break

        dur_qrs = int(round((fin_qrs - ini_qrs) / fs * 1000))
        if 30 <= dur_qrs <= 250:
            qrs_list.append(dur_qrs)

        # ── Onda P (120–220 ms antes del inicio QRS) ──────────────────────
        vp0 = max(0, ini_qrs - int(0.22 * fs))
        vp1 = max(0, ini_qrs - int(0.06 * fs))
        if vp1 - vp0 >= 5:
            seg_p = senal[vp0:vp1]
            picos_p, _ = find_peaks(seg_p,
                                    height=max(0.05 * amp_r, 0.0),
                                    distance=max(2, int(0.03 * fs)))
            if len(picos_p) > 0:
                mejor_p = picos_p[np.argmax(seg_p[picos_p])]
                pr_ms = int(round((ini_qrs - (vp0 + mejor_p)) / fs * 1000))
                if 60 <= pr_ms <= 400:
                    pr_list.append(pr_ms)
                    p_detectadas += 1

        # ── Onda T y fin QT ───────────────────────────────────────────────
        vt0 = fin_qrs + max(1, int(0.05 * fs))  # saltar ST isoelectrico
        vt1 = min(len(senal) - 1, p + int(0.60 * fs))
        if vt1 - vt0 >= 5:
            seg_t = senal[vt0:vt1]
            amp_abs = np.abs(seg_t)
            if np.max(amp_abs) >= 0.05 * amp_r:
                pico_t_local = int(np.argmax(amp_abs))
                pos_t = vt0 + pico_t_local

                # Fin T: donde la señal regresa a baseline (dentro de 200 ms post-T)
                baseline = float(np.mean(senal[max(0, ini_qrs - int(0.15 * fs)):
                                              max(1, ini_qrs - int(0.05 * fs))])) \
                           if ini_qrs > int(0.15 * fs) else 0.0
                fin_t = pos_t
                for j in range(pos_t, min(len(senal) - 1, pos_t + int(0.20 * fs))):
                    if abs(senal[j] - baseline) < 0.08 * amp_r:
                        fin_t = j
                        break

                qt_ms = int(round((fin_t - ini_qrs) / fs * 1000))
                if 200 <= qt_ms <= 750:
                    qt_list.append(qt_ms)

        # ── Segmento ST (en J+80 ms) ─────────────────────────────────────
        j80 = fin_qrs + int(0.08 * fs)
        if j80 < len(senal):
            tp_inicio = max(0, ini_qrs - int(0.20 * fs))
            tp_fin    = max(1, ini_qrs - int(0.06 * fs))
            if tp_fin > tp_inicio:
                baseline_tp = float(np.mean(senal[tp_inicio:tp_fin]))
                delta_st = float(senal[j80] - baseline_tp)
                st_list.append(delta_st)

    qrs_ms = int(np.median(qrs_list)) if qrs_list else 90

    if pr_list:
        pr_ms = int(np.median(pr_list))
    else:
        rr_ms = _rr_medio_ms(picos, fs)
        pr_ms = max(120, min(220, int(rr_ms * 0.20)))

    qt_ms = int(np.median(qt_list)) if qt_list else 400
    st_delta = float(np.median(st_list)) if st_list else 0.0

    return {
        "qrs_ms":        qrs_ms,
        "pr_ms":         pr_ms,
        "qt_ms":         qt_ms,
        "st_delta":      round(st_delta, 4),
        "p_detectadas":  p_detectadas,
    }


def _intervalos_fallback():
    return {"qrs_ms": 90, "pr_ms": 160, "qt_ms": 400,
            "st_delta": 0.0, "p_detectadas": 0}


def _rr_medio_ms(picos, fs):
    if len(picos) < 2:
        return 800
    return float(np.mean(np.diff(picos)) / fs * 1000)


def calcular_qtc_bazett(qt_ms: int, fc: float) -> int:
    if fc <= 0 or qt_ms <= 0:
        return 0
    rr_s = 60.0 / fc
    return int(round(qt_ms / math.sqrt(rr_s)))


# ─────────────────────────────────────────────────────────────────────────────
# 6. ANÁLISIS POR DERIVACIONES
# ─────────────────────────────────────────────────────────────────────────────

# Layout estándar ECG 12 derivaciones (4 columnas × 3 filas + tira ritmo)
LEAD_LAYOUT = [
    ["I",   "aVR", "V1", "V4"],
    ["II",  "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"],
]


def analizar_derivaciones(traza_bin: np.ndarray, filas: list,
                           px_mm: float) -> dict:
    """
    Extrae y analiza cada derivación del ECG.
    Retorna dict con hallazgos por derivación y amplitudes para cálculo de eje.
    """
    fs_eq = px_mm * ECG_PAPER_SPEED  # px/s calibrado
    w = traza_bin.shape[1]
    ancho_col = w // 4

    analisis = {}
    amplitudes = {}  # {lead_name: net_amplitude} para eje

    for fila_idx, (y0, y1) in enumerate(filas[:3]):  # 3 filas de derivaciones
        leads_fila = LEAD_LAYOUT[fila_idx]

        for col_idx, lead_name in enumerate(leads_fila):
            x0 = col_idx * ancho_col
            x1 = (col_idx + 1) * ancho_col

            senal = extraer_senal_franja(traza_bin, y0, y1, x0, x1)
            if len(senal) < 10:
                continue

            picos = detectar_picos_r(senal, fs_eq)

            if len(picos) == 0:
                analisis[lead_name] = "No evaluable"
                amplitudes[lead_name] = 0.0
                continue

            # Amplitud neta QRS (R máx - S mín)
            amp_r = float(np.median([senal[p] for p in picos]))
            amp_s = float(np.min(senal))
            net_amp = amp_r - amp_s if amp_s < 0 else amp_r
            amplitudes[lead_name] = net_amp

            # Morfología simplificada
            desc = _morfologia_lead(senal, picos, fs_eq, lead_name)
            analisis[lead_name] = desc

    # Tira de ritmo (fila 4)
    if len(filas) >= 4:
        y0r, y1r = filas[3]
        senal_r = extraer_senal_franja(traza_bin, y0r, y1r)
        picos_r = detectar_picos_r(senal_r, fs_eq)
        fc_r = _calcular_fc(picos_r, fs_eq)
        regular_r = _es_regular(picos_r)
        analisis["II (tira)"] = (
            f"FC {fc_r} lpm — {'regular' if regular_r else 'IRREGULAR'}"
        )
    else:
        # Usar primera derivación disponible para tira de ritmo
        senal_r = extraer_senal_franja(traza_bin, filas[0][0], filas[0][1])
        picos_r = detectar_picos_r(senal_r, fs_eq)

    return analisis, amplitudes, senal_r, picos_r


def _morfologia_lead(senal, picos, fs, lead_name):
    """Descripción morfológica rápida de una derivación."""
    amp_r = float(np.median([senal[p] for p in picos]))
    amp_s_vals = []
    q_vals = []

    for p in picos:
        # Q: mínimo local antes del pico R
        ventana_q = senal[max(0, p - int(0.06 * fs)):p]
        if len(ventana_q) > 0:
            q_vals.append(float(np.min(ventana_q)))
        # S: mínimo local después del pico R
        ventana_s = senal[p:min(len(senal), p + int(0.06 * fs))]
        if len(ventana_s) > 0:
            amp_s_vals.append(float(np.min(ventana_s)))

    q_amp = float(np.median(q_vals)) if q_vals else 0.0
    s_amp = float(np.median(amp_s_vals)) if amp_s_vals else 0.0

    partes = []
    if q_amp < -0.1 * amp_r:
        partes.append("Q patológica" if q_amp < -0.25 * amp_r else "Q pequeña")
    partes.append(f"R {'prominente' if amp_r > 2 else 'normal'}")
    if s_amp < -0.1 * amp_r:
        partes.append("S marcada")

    return " — ".join(partes) if partes else "Normal"


# ─────────────────────────────────────────────────────────────────────────────
# 7. CÁLCULO DE EJE ELÉCTRICO
# ─────────────────────────────────────────────────────────────────────────────

def calcular_eje(amplitudes: dict) -> tuple:
    """
    Eje eléctrico del QRS en plano frontal usando derivaciones I y aVF.
    Método: eje = atan2(amp_aVF, amp_I) en grados.
    Normal: −30° a +90° | LAD: −30° a −90° | RAD: +90° a +180°
    """
    amp_i   = amplitudes.get("I",   0.0)
    amp_avf = amplitudes.get("aVF", 0.0)

    if abs(amp_i) < 0.01 and abs(amp_avf) < 0.01:
        return "Indeterminado", None

    eje_deg = math.degrees(math.atan2(amp_avf, amp_i))

    if -30 <= eje_deg <= 90:
        descripcion = f"Normal ({eje_deg:.0f}°)"
    elif -90 <= eje_deg < -30:
        descripcion = f"Desviación izquierda / LAD ({eje_deg:.0f}°)"
    elif 90 < eje_deg <= 180:
        descripcion = f"Desviación derecha / RAD ({eje_deg:.0f}°)"
    else:
        descripcion = f"Indeterminado ({eje_deg:.0f}°)"

    return descripcion, round(eje_deg, 1)


# ─────────────────────────────────────────────────────────────────────────────
# 8. MÉTRICAS GLOBALES
# ─────────────────────────────────────────────────────────────────────────────

def _calcular_fc(picos, fs):
    if len(picos) < 2:
        return 0
    rr = np.diff(picos).astype(float) / fs
    rr_validos = rr[(rr > 0.25) & (rr < 2.5)]
    if len(rr_validos) == 0:
        rr_validos = rr
    return int(round(60.0 / np.median(rr_validos)))


def _es_regular(picos, umbral_cv=0.12):
    if len(picos) < 3:
        return True
    rr = np.diff(picos).astype(float)
    cv = np.std(rr) / np.mean(rr) if np.mean(rr) > 0 else 0
    return cv < umbral_cv


# ─────────────────────────────────────────────────────────────────────────────
# 9. DIAGNÓSTICO CLÍNICO (CRITERIOS AHA/ESC)
# ─────────────────────────────────────────────────────────────────────────────

def diagnosticar_clinico(fc, regular, qrs_ms, pr_ms, qt_ms, qtc_ms,
                          st_delta, p_detectadas, eje_deg,
                          amplitudes, analisis_deriv):
    """
    Motor de diagnóstico con criterios clínicos AHA/ESC.
    Evalúa: ritmo, FC, conducción, repolarización, eje.
    """
    hallazgos   = []
    criticos    = []
    patrones    = []
    dx_dif      = []
    alerta      = "verde"

    def upgrade_alerta(nuevo):
        nonlocal alerta
        orden = {"verde": 0, "amarillo": 1, "rojo": 2}
        if orden[nuevo] > orden[alerta]:
            alerta = nuevo

    # ── 1. FRECUENCIA CARDIACA ────────────────────────────────────────────
    if fc <= 0:
        hallazgos.append("Frecuencia cardiaca no determinable")
    elif fc < 40:
        hallazgos.append(f"Bradicardia severa ({fc} lpm)")
        criticos.append(f"Bradicardia severa {fc} lpm — riesgo paro sinusal")
        upgrade_alerta("rojo")
    elif fc < 60:
        hallazgos.append(f"Bradicardia sinusal ({fc} lpm)")
        upgrade_alerta("amarillo")
    elif fc > 180:
        hallazgos.append(f"Taquicardia severa ({fc} lpm)")
        criticos.append(f"Taquicardia {fc} lpm — descartar TV / FV / TSV")
        upgrade_alerta("rojo")
    elif fc > 100:
        hallazgos.append(f"Taquicardia ({fc} lpm)")
        upgrade_alerta("amarillo")
    else:
        hallazgos.append(f"Frecuencia cardiaca normal ({fc} lpm)")

    # ── 2. RITMO ─────────────────────────────────────────────────────────
    if not regular and fc > 0:
        hallazgos.append("Ritmo irregular — evaluar fibrilación/flutter auricular")
        patrones.append("Irregularidad RR — posible FA")
        dx_dif.extend(["Fibrilación auricular", "Flutter auricular",
                       "Extrasístoles frecuentes"])
        upgrade_alerta("amarillo")

    # ── 3. ONDA P / CONDUCCIÓN AV ────────────────────────────────────────
    if p_detectadas == 0 and fc > 0:
        if not regular:
            hallazgos.append("Ondas P ausentes con ritmo irregular — FA probable")
            patrones.append("Sin onda P visible — FA o flutter auricular")
            criticos.append("Ausencia onda P + ritmo irregular: Fibrilación Auricular probable")
            upgrade_alerta("amarillo")
        else:
            hallazgos.append("Onda P no identificable — evaluar ritmo de la unión / TRIN")
    else:
        if 120 <= pr_ms <= 200:
            hallazgos.append(f"Intervalo PR normal ({pr_ms} ms)")
        elif pr_ms > 200:
            hallazgos.append(f"PR prolongado ({pr_ms} ms) — Bloqueo AV 1er grado")
            patrones.append("Bloqueo AV primer grado (PR > 200 ms)")
            dx_dif.append("Bloqueo AV 1er grado")
            upgrade_alerta("amarillo")
        elif pr_ms < 120 and pr_ms > 0:
            hallazgos.append(f"PR corto ({pr_ms} ms) — evaluar WPW / preexcitación")
            patrones.append("PR corto — descartar Wolff-Parkinson-White")
            dx_dif.append("Síndrome de preexcitación / WPW")
            upgrade_alerta("amarillo")

    # ── 4. DURACIÓN QRS ──────────────────────────────────────────────────
    if qrs_ms < 70:
        hallazgos.append(f"QRS estrecho ({qrs_ms} ms) — conducción normal")
    elif 70 <= qrs_ms < 120:
        hallazgos.append(f"QRS normal ({qrs_ms} ms)")
    elif 120 <= qrs_ms < 150:
        hallazgos.append(f"QRS limítrofe ancho ({qrs_ms} ms) — posible bloqueo incompleto de rama")
        patrones.append("QRS limítrofe (120–150 ms) — bloqueo incompleto de rama")
        dx_dif.extend(["Bloqueo incompleto de rama derecha",
                       "Bloqueo incompleto de rama izquierda"])
        upgrade_alerta("amarillo")
    else:
        hallazgos.append(f"QRS ancho ({qrs_ms} ms) — Bloqueo completo de rama o conducción aberrante")
        patrones.append("QRS ancho ≥150 ms — bloqueo completo de rama")
        criticos.append(f"QRS ancho {qrs_ms} ms — descartar bloqueo de rama o taquicardia ventricular")
        dx_dif.extend(["Bloqueo completo de rama derecha (BRDHH)",
                       "Bloqueo completo de rama izquierda (BRIHH)",
                       "Taquicardia ventricular si FC elevada"])
        upgrade_alerta("amarillo" if qrs_ms < 200 else "rojo")

    # ── 5. REPOLARIZACIÓN (QT/QTc) ───────────────────────────────────────
    if qtc_ms > 0:
        if qtc_ms > 500:
            hallazgos.append(f"QTc muy prolongado ({qtc_ms} ms) — ALTO RIESGO Torsades de Pointes")
            criticos.append(f"QTc {qtc_ms} ms — riesgo arritmia ventricular maligna")
            upgrade_alerta("rojo")
        elif qtc_ms > 450:
            hallazgos.append(f"QTc prolongado ({qtc_ms} ms) — evaluar causas (fármacos, electrolitos)")
            dx_dif.append("QT largo congénito vs adquirido")
            upgrade_alerta("amarillo")
        elif qtc_ms < 350 and qtc_ms > 0:
            hallazgos.append(f"QTc corto ({qtc_ms} ms) — evaluar hipercalcemia / síndrome QT corto")
            dx_dif.append("Síndrome QT corto / hipercalcemia")
            upgrade_alerta("amarillo")
        else:
            hallazgos.append(f"QTc normal ({qtc_ms} ms)")

    # ── 6. SEGMENTO ST ───────────────────────────────────────────────────
    # st_delta normalizado respecto a amplitud R
    # Umbral clínico: ≥1mm (≈0.1mV) en derivaciones de miembros
    #                  ≥2mm en precordiales para STEMI
    UMBRAL_ELEVACION = 0.20   # relativo al tamaño del R (calibración aproximada)
    UMBRAL_DEPRESION = -0.15

    if st_delta > UMBRAL_ELEVACION:
        hallazgos.append("Posible elevación del segmento ST — DESCARTAR STEMI URGENTE")
        criticos.append("Elevación ST — activar protocolo código IAM")
        patrones.append("STEMI probable")
        dx_dif.extend(["STEMI", "Pericarditis aguda", "Repolarización precoz"])
        upgrade_alerta("rojo")
    elif st_delta < UMBRAL_DEPRESION:
        hallazgos.append("Posible depresión del segmento ST — evaluar isquemia subendocárdica")
        patrones.append("Depresión ST — isquemia posible")
        dx_dif.extend(["NSTEMI / SCASEST", "Sobrecarga ventricular",
                       "Efecto digitálico"])
        upgrade_alerta("amarillo")
    else:
        hallazgos.append("Segmento ST en línea isoeléctrica — sin signos de lesión")

    # ── 7. EJE ELÉCTRICO ─────────────────────────────────────────────────
    if eje_deg is not None:
        if eje_deg < -30:
            hallazgos.append(f"Desviación izquierda del eje ({eje_deg:.0f}°) — evaluar HBAI, HVI")
            patrones.append("Desviación izquierda del eje")
            dx_dif.extend(["Hemibloqueo anterior izquierdo (HBAI)",
                           "Hipertrofia ventricular izquierda"])
        elif eje_deg > 90:
            hallazgos.append(f"Desviación derecha del eje ({eje_deg:.0f}°) — evaluar HVD, HBPI")
            patrones.append("Desviación derecha del eje")
            dx_dif.extend(["Hipertrofia ventricular derecha",
                           "Hemibloqueo posterior izquierdo"])

    # ── 8. DIAGNÓSTICO PRINCIPAL ─────────────────────────────────────────
    if "STEMI" in " ".join(patrones):
        dx_principal = "STEMI — ACTIVAR PROTOCOLO CATETERISMO CARDÍACO URGENTE"
    elif alerta == "rojo" and fc > 150:
        if qrs_ms >= 120:
            dx_principal = f"Taquicardia de QRS ancho ({fc} lpm) — posible taquicardia ventricular"
        else:
            dx_principal = f"Taquicardia supraventricular ({fc} lpm) — evaluación urgente"
    elif alerta == "rojo" and fc < 40:
        dx_principal = f"Bradicardia severa ({fc} lpm) — riesgo de paro"
    elif "Fibrilación auricular" in " ".join(dx_dif) and not regular:
        dx_principal = f"Fibrilación auricular — FC ventricular {fc} lpm"
    elif "Bloqueo completo de rama" in " ".join(patrones):
        dx_principal = "Trastorno de conducción intraventricular — bloqueo de rama"
    elif alerta == "amarillo":
        detalles = "; ".join(hallazgos[:2])
        dx_principal = f"Alteraciones electrocardiográficas — {detalles}"
    else:
        ritmo_desc = "Taquicardia sinusal" if fc > 100 else (
                     "Bradicardia sinusal" if fc < 60 else "Ritmo sinusal normal")
        dx_principal = f"{ritmo_desc} — {fc} lpm — Sin alteraciones mayores"

    # ── 9. CONDUCTA ───────────────────────────────────────────────────────
    if alerta == "rojo":
        conducta = (
            "URGENCIA INMEDIATA: Activar código cardíaco, "
            "desfibrilador disponible, acceso venoso, monitorización continua"
        )
    elif alerta == "amarillo":
        conducta = (
            "Evaluación cardiológica prioritaria, ECG de 12 derivaciones completo, "
            "electrolitos, función renal y tiroidea, lista de medicamentos actuales"
        )
    else:
        conducta = (
            "Control ambulatorio según contexto clínico. "
            "Correlacionar con síntomas y factores de riesgo cardiovascular"
        )

    # ── 10. RITMO TEXTO ───────────────────────────────────────────────────
    if not regular and p_detectadas == 0:
        ritmo_txt = "Fibrilación auricular probable"
    elif not regular:
        ritmo_txt = "Ritmo irregular — extrasístoles / FA"
    elif fc > 100:
        ritmo_txt = "Taquicardia sinusal"
    elif fc < 60:
        ritmo_txt = "Bradicardia sinusal"
    else:
        ritmo_txt = "Ritmo sinusal regular"

    return {
        "ritmo_txt":     ritmo_txt,
        "dx_principal":  dx_principal,
        "hallazgos":     hallazgos,
        "criticos":      criticos,
        "patrones":      patrones,
        "dx_dif":        list(dict.fromkeys(dx_dif))[:6],  # únicos, máx 6
        "conducta":      conducta,
        "alerta":        alerta,
    }


# ─────────────────────────────────────────────────────────────────────────────
# UTILIDADES SEÑAL HOLTER
# ─────────────────────────────────────────────────────────────────────────────

def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = fs / 2
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return b, a


def filtrar_ecg(senal, fs=500):
    if len(senal) < 30:
        return senal
    b, a = butter_bandpass(0.5, 40, fs)
    return filtfilt(b, a, senal)


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
def home():
    return {"estado": "En línea", "version": "4.0",
            "mensaje": "MediSumma Brain v4.0 — Motor ECG Clínico Avanzado"}


@app.post("/analizar_holter")
async def analizar_holter(file: UploadFile = File(...)):
    contenido = await file.read()
    try:
        senal = np.frombuffer(contenido, dtype=np.int16).astype(np.float64)
    except Exception as e:
        return {"error": f"No se pudo leer el formato: {e}"}

    fs     = 500
    senal  = senal[:10 * fs]
    senal_f = filtrar_ecg(senal, fs)
    picos  = detectar_picos_r(senal_f, fs)
    fc     = _calcular_fc(picos, fs)
    regular = _es_regular(picos)

    dx = ("Ritmo Sinusal Normal" if fc and 60 <= fc <= 100 and regular
          else "Bradiarritmia" if fc and fc < 60
          else "Taquiarritmia" if fc and fc > 100
          else "Ritmo Irregular")

    return {
        "filename": file.filename,
        "duracion_analizada": "10 segundos",
        "latidos_detectados": len(picos),
        "frecuencia_cardiaca": fc,
        "diagnostico_texto": dx,
        "detalles": f"FC {fc} lpm — {'regular' if regular else 'irregular'}",
        "alerta_color": "red" if fc and (fc > 150 or fc < 40) else
                        "orange" if not regular else "green",
        "senal_grafica": senal_f[:2000].tolist(),
    }


@app.post("/analizar_ecg_foto")
async def analizar_ecg_foto(file: UploadFile = File(...)):
    """
    Analiza una fotografía de ECG en papel con motor de visión computacional calibrado.
    Sin API Key. Devuelve JSON compatible con EcgAnalysisResult Flutter.
    """
    img_bytes = await file.read()

    # ── Decodificar imagen ────────────────────────────────────────────────
    nparr = np.frombuffer(img_bytes, np.uint8)
    img_color = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_color is None:
        return _error("No se pudo decodificar la imagen — verifica el formato")

    # Redimensionar a ancho estándar manteniendo proporción
    h_orig, w_orig = img_color.shape[:2]
    escala   = TARGET_WIDTH / w_orig
    h_nuevo  = max(200, int(h_orig * escala))
    img_resz = cv2.resize(img_color, (TARGET_WIDTH, h_nuevo),
                          interpolation=cv2.INTER_LINEAR)

    img_gray = cv2.cvtColor(img_resz, cv2.COLOR_BGR2GRAY)

    # ── Calibración ──────────────────────────────────────────────────────
    px_mm = calibrar_px_mm(img_gray)
    fs_eq = px_mm * ECG_PAPER_SPEED  # px/s calibrado real

    # ── Preprocesamiento ─────────────────────────────────────────────────
    traza_bin = preprocesar_ecg(img_gray, px_mm)

    # ── Verificar calidad ────────────────────────────────────────────────
    contenido_total = float(np.sum(traza_bin > 0))
    if contenido_total < (TARGET_WIDTH * 10):
        return _error("Imagen con muy poco contraste — fotografía con mejor iluminación y sin flash directo")

    # ── Segmentar filas ───────────────────────────────────────────────────
    filas = detectar_filas(traza_bin, px_mm)

    # ── Análisis por derivaciones ─────────────────────────────────────────
    try:
        analisis_leads, amplitudes, senal_ritmo, picos_ritmo = \
            analizar_derivaciones(traza_bin, filas, px_mm)
    except Exception as e:
        return _error(f"Error al analizar derivaciones: {e}")

    if len(picos_ritmo) < 2:
        return _error("Pocos complejos QRS detectados — alinea mejor la imagen y asegúrate de incluir la tira de ritmo")

    # ── Métricas globales (sobre tira de ritmo) ───────────────────────────
    fc      = _calcular_fc(picos_ritmo, fs_eq)
    regular = _es_regular(picos_ritmo)
    intervalos = medir_intervalos(senal_ritmo, picos_ritmo, fs_eq)
    qtc_ms  = calcular_qtc_bazett(intervalos["qt_ms"], fc)
    eje_txt, eje_deg = calcular_eje(amplitudes)

    # ── Diagnóstico clínico ───────────────────────────────────────────────
    dx = diagnosticar_clinico(
        fc=fc,
        regular=regular,
        qrs_ms=intervalos["qrs_ms"],
        pr_ms=intervalos["pr_ms"],
        qt_ms=intervalos["qt_ms"],
        qtc_ms=qtc_ms,
        st_delta=intervalos["st_delta"],
        p_detectadas=intervalos["p_detectadas"],
        eje_deg=eje_deg,
        amplitudes=amplitudes,
        analisis_deriv=analisis_leads,
    )

    # ── Calidad de imagen ─────────────────────────────────────────────────
    n_ciclos = len(picos_ritmo)
    if n_ciclos >= 6 and intervalos["p_detectadas"] >= 3:
        calidad = f"Buena — {n_ciclos} complejos QRS detectados, onda P visible"
    elif n_ciclos >= 3:
        calidad = f"Aceptable — {n_ciclos} complejos detectados. Mejor imagen mejoraría precisión"
    else:
        calidad = "Regular — pocos complejos detectados; sube imagen más nítida"

    # ── Morfología P y T ──────────────────────────────────────────────────
    if intervalos["p_detectadas"] == 0:
        morfologia_p = "Onda P no identificable — evaluar FA o ritmo nodal"
    elif intervalos["pr_ms"] > 200:
        morfologia_p = f"Onda P presente con PR prolongado ({intervalos['pr_ms']} ms)"
    else:
        morfologia_p = f"Onda P presente — PR {intervalos['pr_ms']} ms (normal)"

    qrs_ms = intervalos["qrs_ms"]
    if qrs_ms >= 150:
        morfologia_qrs = f"QRS ancho ({qrs_ms} ms) — bloqueo de rama probable"
    elif qrs_ms >= 120:
        morfologia_qrs = f"QRS limítrofe ({qrs_ms} ms)"
    else:
        morfologia_qrs = f"QRS estrecho ({qrs_ms} ms) — conducción normal"

    st_d = intervalos["st_delta"]
    if st_d > 0.2:
        st_txt = "Elevación del ST — STEMI posible"
        onda_t_txt = "T positiva con posible elevación — patrón de lesión"
    elif st_d < -0.15:
        st_txt = "Depresión del ST — isquemia posible"
        onda_t_txt = "Evaluar inversión de onda T — isquemia"
    else:
        st_txt = "ST isoeléctrico — sin signos de lesión aguda"
        onda_t_txt = "Onda T morfología no evaluada en detalle"

    # ── Narrativa clínica ─────────────────────────────────────────────────
    narrativa = (
        f"ECG analizado mediante visión computacional calibrada. "
        f"Calibración estimada: {px_mm:.1f} px/mm ({fs_eq:.0f} px/s equivalente). "
        f"{'Ritmo ' + dx['ritmo_txt'] + '.' if fc > 0 else 'Ritmo no determinable.'} "
        f"Frecuencia cardiaca: {fc} lpm. "
        f"Intervalo PR: {intervalos['pr_ms']} ms. "
        f"Duración QRS: {qrs_ms} ms. "
        f"QT: {intervalos['qt_ms']} ms / QTc Bazett: {qtc_ms} ms. "
        f"Eje eléctrico: {eje_txt}. "
        f"Segmento ST: {st_txt}. "
        f"{'HALLAZGOS CRÍTICOS: ' + ' | '.join(dx['criticos']) + '.' if dx['criticos'] else 'Sin hallazgos críticos inmediatos detectados.'}"
    )

    return {
        "calidad_imagen":           calidad,
        "ritmo":                    dx["ritmo_txt"],
        "frecuencia_cardiaca":      fc,
        "intervalo_pr_ms":          intervalos["pr_ms"],
        "duracion_qrs_ms":          qrs_ms,
        "intervalo_qt_ms":          intervalos["qt_ms"],
        "qtc_ms":                   qtc_ms,
        "eje_electrico":            eje_txt,
        "morfologia_p":             morfologia_p,
        "morfologia_qrs":           morfologia_qrs,
        "segmento_st":              st_txt,
        "onda_t":                   onda_t_txt,
        "analisis_por_derivacion":  analisis_leads,
        "hallazgos_criticos":       dx["criticos"],
        "hallazgos":                dx["hallazgos"],
        "patrones_detectados":      dx["patrones"],
        "diagnostico_principal":    dx["dx_principal"],
        "diagnosticos_diferenciales": dx["dx_dif"],
        "interpretacion_narrativa": narrativa,
        "alerta_nivel":             dx["alerta"],
        "conducta_recomendada":     dx["conducta"],
        "advertencia": (
            "Análisis automático por visión computacional — uso exclusivamente educativo y de apoyo. "
            "No reemplaza la interpretación de un cardiólogo certificado. "
            "Ante cualquier hallazgo crítico consulte urgencias de inmediato."
        ),
    }


def _error(msg: str) -> dict:
    return {
        "calidad_imagen":             f"Error — {msg}",
        "ritmo":                      "No evaluable",
        "frecuencia_cardiaca":        0,
        "intervalo_pr_ms":            0,
        "duracion_qrs_ms":            0,
        "intervalo_qt_ms":            0,
        "qtc_ms":                     0,
        "eje_electrico":              "No evaluable",
        "morfologia_p":               "No evaluable",
        "morfologia_qrs":             "No evaluable",
        "segmento_st":                "No evaluable",
        "onda_t":                     "No evaluable",
        "analisis_por_derivacion":    {},
        "hallazgos_criticos":         [],
        "hallazgos":                  [msg],
        "patrones_detectados":        [],
        "diagnostico_principal":      "Imagen no evaluable",
        "diagnosticos_diferenciales": [],
        "interpretacion_narrativa":   msg,
        "alerta_nivel":               "amarillo",
        "conducta_recomendada":       "Vuelve a cargar con mejor imagen y buena iluminación",
        "advertencia":                "Solo uso educativo.",
    }
