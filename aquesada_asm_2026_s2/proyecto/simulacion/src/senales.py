import numpy as np


# =====================================================
# CONFIGURACIÓN GENERAL
# =====================================================

FS = 16000
N_BLOQUE = 1024
DURACION = 6.0
SNR_OBJETIVO_DB = 5.0

SEMILLAS = {
    "P1": 101,
    "P2": 202,
    "P3": 303,
    "P4": 404,
    "P5": 505,
}


# =====================================================
# AJUSTAR RUIDO A SNR OBJETIVO
# =====================================================

def ajustar_ruido_a_snr(
    senal_limpia,
    ruido,
    snr_objetivo_db
):
    senal_limpia = np.asarray(
        senal_limpia,
        dtype=np.float64
    )

    ruido = np.asarray(
        ruido,
        dtype=np.float64
    )

    potencia_senal = np.mean(
        senal_limpia ** 2
    )

    potencia_ruido = np.mean(
        ruido ** 2
    )

    if potencia_ruido <= 0:
        raise ValueError(
            "La potencia del ruido debe ser mayor que cero."
        )

    snr_lineal = 10 ** (
        snr_objetivo_db / 10
    )

    potencia_ruido_objetivo = (
        potencia_senal / snr_lineal
    )

    escala = np.sqrt(
        potencia_ruido_objetivo
        / potencia_ruido
    )

    ruido_ajustado = ruido * escala

    contaminada = (
        senal_limpia
        + ruido_ajustado
    )

    return ruido_ajustado, contaminada


# =====================================================
# CALCULAR SNR
# =====================================================

def calcular_snr_db(
    referencia,
    contaminada
):
    referencia = np.asarray(
        referencia,
        dtype=np.float64
    )

    contaminada = np.asarray(
        contaminada,
        dtype=np.float64
    )

    error = contaminada - referencia

    energia_senal = np.sum(
        referencia ** 2
    )

    energia_error = np.sum(
        error ** 2
    )

    return 10 * np.log10(
        energia_senal
        / max(
            energia_error,
            1e-30
        )
    )


# =====================================================
# P1
# TONO + INTERFERENCIA SINUSOIDAL
# =====================================================

def generar_p1(semilla=None):
    t = np.arange(
        int(FS * DURACION)
    ) / FS

    limpia = np.sin(
        2 * np.pi * 1000 * t
    )

    # Sin semilla: reproducir exactamente el caso de referencia.
    # Con semilla: variar de forma reproducible la fase del ruido.
    fase_ruido = 0.0
    if semilla is not None:
        rng = np.random.default_rng(semilla)
        fase_ruido = rng.uniform(0.0, 2 * np.pi)

    ruido_base = np.sin(
        2 * np.pi * 2500 * t + fase_ruido
    )

    ruido, contaminada = ajustar_ruido_a_snr(
        limpia,
        ruido_base,
        SNR_OBJETIVO_DB
    )

    return limpia, ruido, contaminada


# =====================================================
# P2
# TRES TONOS + RUIDO DE ALTA FRECUENCIA
# =====================================================

def generar_p2(semilla=None):
    rng = np.random.default_rng(
        SEMILLAS["P2"] if semilla is None else semilla
    )

    t = np.arange(
        int(FS * DURACION)
    ) / FS

    limpia = (
        1.0 * np.sin(2*np.pi*500*t)
        + 0.7 * np.sin(2*np.pi*900*t)
        + 0.5 * np.sin(2*np.pi*1300*t)
    )

    ruido_blanco = rng.normal(
        0,
        1,
        len(t)
    )

    R = np.fft.rfft(
        ruido_blanco
    )

    f = np.fft.rfftfreq(
        len(ruido_blanco),
        1 / FS
    )

    R[f < 2000] = 0

    ruido_base = np.fft.irfft(
        R,
        n=len(ruido_blanco)
    )

    ruido, contaminada = ajustar_ruido_a_snr(
        limpia,
        ruido_base,
        SNR_OBJETIVO_DB
    )

    return limpia, ruido, contaminada


# =====================================================
# P3
# SEÑAL ARBITRARIA + RUIDO DE BANDA ANCHA
# =====================================================

def generar_p3(semilla=None):
    rng = np.random.default_rng(
        SEMILLAS["P3"] if semilla is None else semilla
    )

    t = np.arange(
        int(FS * DURACION)
    ) / FS

    envolvente = (
        0.7
        + 0.3 * np.sin(
            2 * np.pi * 0.8 * t
        )
    )

    limpia = envolvente * (
        0.8 * np.sin(2*np.pi*400*t)
        + 0.5 * np.sin(2*np.pi*750*t)
        + 0.3 * np.sin(2*np.pi*1200*t)
    )

    ruido_base = rng.normal(
        0,
        1,
        len(t)
    )

    ruido, contaminada = ajustar_ruido_a_snr(
        limpia,
        ruido_base,
        SNR_OBJETIVO_DB
    )

    return limpia, ruido, contaminada


# =====================================================
# P4
# SEÑAL TRANSITORIA + RUIDO IMPULSIVO
# =====================================================

def generar_p4(semilla=None):
    rng = np.random.default_rng(
        SEMILLAS["P4"] if semilla is None else semilla
    )

    t = np.arange(
        int(FS * DURACION)
    ) / FS

    centro = 2.0

    envolvente = np.exp(
        -35 * np.abs(
            t - centro
        )
    )

    limpia = envolvente * np.sin(
        2 * np.pi * 900 * t
    )

    ruido_base = np.zeros_like(
        limpia
    )

    num_impulsos = 80

    indices = rng.choice(
        len(ruido_base),
        size=num_impulsos,
        replace=False
    )

    ruido_base[indices] = rng.choice(
        [-1.0, 1.0],
        size=num_impulsos
    )

    ruido, contaminada = ajustar_ruido_a_snr(
        limpia,
        ruido_base,
        SNR_OBJETIVO_DB
    )

    return limpia, ruido, contaminada


# =====================================================
# P5
# SOLAPAMIENTO PARCIAL SEÑAL-RUIDO
# =====================================================

def generar_p5(semilla=None):
    rng = np.random.default_rng(
        SEMILLAS["P5"] if semilla is None else semilla
    )

    t = np.arange(
        int(FS * DURACION)
    ) / FS

    limpia = (
        1.0 * np.sin(2*np.pi*500*t)
        + 0.7 * np.sin(2*np.pi*900*t)
        + 0.5 * np.sin(2*np.pi*1300*t)
    )

    ruido_blanco = rng.normal(
        0,
        1,
        len(t)
    )

    R = np.fft.rfft(
        ruido_blanco
    )

    f = np.fft.rfftfreq(
        len(ruido_blanco),
        1 / FS
    )

    mascara = (
        (f >= 800)
        & (f <= 1500)
    )

    R[~mascara] = 0

    ruido_base = np.fft.irfft(
        R,
        n=len(ruido_blanco)
    )

    ruido, contaminada = ajustar_ruido_a_snr(
        limpia,
        ruido_base,
        SNR_OBJETIVO_DB
    )

    return limpia, ruido, contaminada