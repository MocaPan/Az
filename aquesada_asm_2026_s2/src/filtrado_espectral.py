"""Filtrado espectral adaptativo mediante STFT/ISTFT y solapamiento (2.4)."""

import numpy as np
from scipy import ndimage, signal


def filtrar_espectral(
    x,
    fs=16000,
    N=1024,
    reduccion=1.0,
    paso_banda_hz=1450.0,
    umbral_pico=7.0,
    detalle=False,
):
    """Reduce ruido en bloques sin fijar frecuencias interferentes.

    Estima el fondo espectral mediante medianas locales, detecta picos
    prominentes fuera de la banda útil y aplica una ganancia espectral
    suave. La ISTFT reconstruye la señal mediante solapamiento y suma.
    """
    x = np.asarray(x, dtype=np.float64)

    # Validación de parámetros.
    if N < 32 or N & (N - 1):
        raise ValueError("N debe ser potencia de dos >=32")

    if reduccion < 0:
        raise ValueError("reduccion debe ser no negativa")

    # 1. Transformación de la señal al dominio tiempo-frecuencia.
    f, t, Z = signal.stft(
        x,
        fs=fs,
        window="hann",
        nperseg=N,
        noverlap=N // 2,
        boundary="zeros",
        padded=True,
    )

    # 2. Estimación del ruido de fondo en cada trama.
    mag = np.abs(Z)
    piso = ndimage.median_filter(mag, size=(31, 1), mode="nearest")

    # 3. Detección de interferencias tonales persistentes.
    perfil = np.median(mag, axis=1)
    perfil_fondo = ndimage.median_filter(perfil, size=31, mode="nearest")

    elegibles = (
        (f > paso_banda_hz)
        & (f < 0.95 * fs / 2)
        & (perfil > 0.01 * np.max(perfil))
    )

    razon = perfil / (perfil_fondo + 1e-12)
    picos, _ = signal.find_peaks(razon, height=umbral_pico, distance=3)
    picos = np.array([p for p in picos if elegibles[p]], dtype=int)

    # 4. Atenuación espectral suave según el nivel de reducción.
    ganancia = np.maximum(
        0.08,
        1 - reduccion * (piso / (mag + 1e-12)) ** 1.0,
    )

    # Rechazo adicional alrededor de los picos detectados automáticamente.
    for p in picos:
        ini = max(0, p - 2)
        fin = min(len(f), p + 3)
        ganancia[ini:fin, :] *= 1 / (1 + 8 * reduccion)

    # 5. Modificación espectral y reconstrucción con ISTFT.
    # La representación unilateral permite reconstruir una señal real.
    Y = Z * ganancia

    _, y = signal.istft(
        Y,
        fs=fs,
        window="hann",
        nperseg=N,
        noverlap=N // 2,
        input_onesided=True,
        boundary=True,
    )

    y = y[: len(x)]

    # 6. Entrega opcional de metadatos para las pruebas y el informe.
    if detalle:
        return y, {
            "frecuencias_picos_hz": f[picos].tolist(),
            "N": N,
            "salto": N // 2,
            "solapamiento": 0.5,
            "ganancia_minima": float(ganancia.min()),
            "error_imaginario_esperado": 0.0,
        }

    return y
