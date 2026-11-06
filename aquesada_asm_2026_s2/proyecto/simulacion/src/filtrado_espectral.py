import numpy as np

from scipy import signal
from scipy import ndimage


# =====================================================
# CONFIGURACIÓN GENERAL
# =====================================================

FS = 16000

N = 1024
HOP = 512

REDUCCION = 1.0
GANANCIA_MIN = 0.08
PASO_BANDA_HZ = 1450.0
UMBRAL_PICO = 7.0


# =====================================================
# FILTRADO ESPECTRAL
# =====================================================

def filtrar_espectral(
    senal,
    fs=FS,
    n=N,
    reduccion=REDUCCION,
    ganancia_min=GANANCIA_MIN,
    paso_banda_hz=PASO_BANDA_HZ,
    umbral_pico=UMBRAL_PICO
):
    senal = np.asarray(
        senal,
        dtype=np.float64
    )

    if n <= 0:
        raise ValueError(
            "El tamaño de bloque debe ser mayor que cero."
        )

    if n % 2 != 0:
        raise ValueError(
            "El tamaño de bloque debe ser par."
        )

    if reduccion < 0:
        raise ValueError(
            "La reducción no puede ser negativa."
        )

    # =================================================
    # 1. STFT
    # =================================================

    frecuencias, tiempos, Z = signal.stft(
        senal,
        fs=fs,
        window="hann",
        nperseg=n,
        noverlap=n // 2,
        boundary="zeros",
        padded=True
    )

    magnitud = np.abs(Z)

    # =================================================
    # 2. ESTIMACIÓN DEL PISO ESPECTRAL
    # =================================================

    piso = ndimage.median_filter(
        magnitud,
        size=(31, 1),
        mode="nearest"
    )

    # =================================================
    # 3. PERFIL ESPECTRAL GLOBAL
    # =================================================

    perfil = np.median(
        magnitud,
        axis=1
    )

    perfil_fondo = ndimage.median_filter(
        perfil,
        size=31,
        mode="nearest"
    )

    razon = (
        perfil
        / (perfil_fondo + 1e-12)
    )

    # =================================================
    # 4. DETECCIÓN AUTOMÁTICA DE PICOS
    # =================================================

    picos, propiedades = signal.find_peaks(
        razon,
        height=umbral_pico,
        distance=3
    )

    # Solo considerar picos fuera de la banda útil.
    picos_validos = np.array(
        [
            p
            for p in picos
            if frecuencias[p] > paso_banda_hz
        ],
        dtype=int
    )

    # =================================================
    # 5. GANANCIA ESPECTRAL
    # =================================================

    ganancia = np.maximum(
        ganancia_min,
        1.0
        - reduccion
        * (
            piso
            / (magnitud + 1e-12)
        )
    )

    # Atenuación adicional alrededor
    # de componentes tonales detectadas.
    for pico in picos_validos:

        inicio = max(
            0,
            pico - 2
        )

        fin = min(
            len(frecuencias),
            pico + 3
        )

        ganancia[inicio:fin, :] *= (
            1.0
            / (
                1.0
                + 8.0 * reduccion
            )
        )

    # Nunca permitir una ganancia
    # menor que la mínima especificada.
    ganancia = np.maximum(
        ganancia,
        ganancia_min
    )

    # =================================================
    # 6. MODIFICACIÓN COMPLEJA
    # =================================================

    Z_filtrado = (
        Z
        * ganancia
    )

    # =================================================
    # 7. IFFT Y RECONSTRUCCIÓN
    # =================================================

    _, salida = signal.istft(
        Z_filtrado,
        fs=fs,
        window="hann",
        nperseg=n,
        noverlap=n // 2,
        input_onesided=True,
        boundary=True
    )

    # Conservar misma longitud que entrada.
    salida = salida[
        :len(senal)
    ]

    return salida


# =====================================================
# INFORMACIÓN DEL PROCESAMIENTO
# =====================================================

def obtener_parametros():
    return {
        "fs": FS,
        "n": N,
        "hop": HOP,
        "overlap_pct": 50,
        "reduccion": REDUCCION,
        "ganancia_min": GANANCIA_MIN,
        "paso_banda_hz": PASO_BANDA_HZ,
        "umbral_pico": UMBRAL_PICO,
    }