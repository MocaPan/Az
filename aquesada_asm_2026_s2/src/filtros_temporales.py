"""Filtro IIR causal en secciones de segundo orden para Tarea 1, apartado 2.3."""

import numpy as np
from scipy import signal


FS = 16000
ORDEN = 12
CORTE_HZ = 1500.0


def disenar_filtro(fs=FS, orden=ORDEN, corte_hz=CORTE_HZ):
    """Butterworth pasa bajas, implementado como SOS para estabilidad numérica."""
    return signal.butter(
        orden,
        corte_hz,
        btype="lowpass",
        fs=fs,
        output="sos",
    )


def ecuacion_diferencias(x, sos):
    """Aplica cada biquad por recurrencia Direct Form II Transposed, muestra a muestra.

    w0 = b0*x + z1; z1 = b1*x - a1*y + z2; z2 = b2*x - a2*y.
    Para cada etapa, el denominador se normaliza a0=1.
    """
    y = np.asarray(x, dtype=np.float64).copy()

    for sec in np.asarray(sos, dtype=np.float64):
        b0, b1, b2, a0, a1, a2 = sec
        b0, b1, b2, a1, a2 = b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0
        z1 = z2 = 0.0
        salida = np.empty_like(y)

        for i, muestra in enumerate(y):
            v = b0 * muestra + z1
            z1 = b1 * muestra - a1 * v + z2
            z2 = b2 * muestra - a2 * v
            salida[i] = v

        y = salida

    return y


def validar(x, sos, tolerancia=1e-10):
    propio = ecuacion_diferencias(x, sos)
    referencia = signal.sosfilt(sos, np.asarray(x, dtype=np.float64))
    error = float(np.max(np.abs(propio - referencia)))

    return error, bool(
        np.allclose(propio, referencia, atol=tolerancia, rtol=0)
    )


def propiedades(sos, fs=FS):
    z, p, k = signal.sos2zpk(sos)
    w, h = signal.sosfreqz(sos, worN=4096, fs=fs)
    ang = np.unwrap(np.angle(h))
    retardo = -np.gradient(ang, 2 * np.pi * w / fs)

    return {
        "ceros": z,
        "polos": p,
        "ganancia": k,
        "frecuencias": w,
        "magnitud_db": 20 * np.log10(np.maximum(abs(h), 1e-12)),
        "fase_rad": ang,
        "retardo_grupo_muestras": retardo,
        "estable": bool(np.all(np.abs(p) < 1)),
        "radio_polo_max": float(max(abs(p))),
    }


def cuantizacion(sos, tipo=np.float32):
    """Evaluación de redondeo de coeficientes en coma flotante."""
    return np.asarray(sos, dtype=tipo).astype(np.float64)
