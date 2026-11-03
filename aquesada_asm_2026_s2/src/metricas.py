import time
import numpy as np


def mse(referencia, estimacion):
    """
    Error cuadrático medio entre una referencia y una estimación.
    """
    referencia = np.asarray(referencia, dtype=np.float64)
    estimacion = np.asarray(estimacion, dtype=np.float64)

    if referencia.shape != estimacion.shape:
        raise ValueError(
            "La referencia y la estimación deben tener la misma forma."
        )

    return float(
        np.mean(
            (referencia - estimacion) ** 2
        )
    )


def snr_db(referencia, estimacion):
    """
    SNR utilizando la señal limpia como referencia.

    SNR = 10 log10(
        energia_referencia /
        energia_error
    )
    """
    referencia = np.asarray(referencia, dtype=np.float64)
    estimacion = np.asarray(estimacion, dtype=np.float64)

    if referencia.shape != estimacion.shape:
        raise ValueError(
            "La referencia y la estimación deben tener la misma forma."
        )

    energia_senal = np.sum(
        referencia ** 2
    )

    energia_error = np.sum(
        (referencia - estimacion) ** 2
    )

    return float(
        10.0 * np.log10(
            energia_senal
            /
            max(energia_error, 1e-30)
        )
    )


def delta_snr_db(
    referencia,
    entrada,
    salida
):
    """
    Mejora de SNR respecto a la señal contaminada.
    """
    snr_entrada = snr_db(
        referencia,
        entrada
    )

    snr_salida = snr_db(
        referencia,
        salida
    )

    return float(
        snr_salida - snr_entrada
    )


def alinear_por_retardo(
    referencia,
    salida,
    max_retardo=64
):
    """
    Busca un retardo causal entero que minimice el MSE.

    Se utiliza únicamente para evaluación offline del filtro temporal.
    No modifica el algoritmo de filtrado real.

    Retorna:
        referencia_alineada,
        salida_alineada,
        retardo
    """
    referencia = np.asarray(
        referencia,
        dtype=np.float64
    )

    salida = np.asarray(
        salida,
        dtype=np.float64
    )

    mejor_retardo = 0
    mejor_error = np.inf

    for retardo in range(
        max_retardo + 1
    ):
        if retardo == 0:
            ref = referencia
            out = salida
        else:
            ref = referencia[:-retardo]
            out = salida[retardo:]

        error = np.mean(
            (ref - out) ** 2
        )

        if error < mejor_error:
            mejor_error = error
            mejor_retardo = retardo

    if mejor_retardo == 0:
        return (
            referencia,
            salida,
            0
        )

    return (
        referencia[:-mejor_retardo],
        salida[mejor_retardo:],
        mejor_retardo
    )


def energia_util_conservada_pct(
    referencia,
    salida
):
    """
    Estima cuánto de la componente alineada con la señal limpia
    permanece en la salida.

    Se calcula mediante la proyección de la salida sobre la referencia.

    100 % significa que la componente útil conserva aproximadamente
    la misma amplitud global que la señal limpia.
    """
    referencia = np.asarray(
        referencia,
        dtype=np.float64
    )

    salida = np.asarray(
        salida,
        dtype=np.float64
    )

    if referencia.shape != salida.shape:
        raise ValueError(
            "Referencia y salida deben tener la misma forma."
        )

    energia_ref = np.dot(
        referencia,
        referencia
    )

    if energia_ref <= 0:
        return 0.0

    ganancia_util = (
        np.dot(salida, referencia)
        /
        energia_ref
    )

    return float(
        100.0
        * ganancia_util ** 2
    )


def energia_banda(
    senal,
    fs,
    f_min,
    f_max
):
    """
    Calcula energía espectral dentro de una banda utilizando
    una FFT con ventana Hann.
    """
    senal = np.asarray(
        senal,
        dtype=np.float64
    )

    N = len(senal)

    ventana = np.hanning(N)

    espectro = np.fft.rfft(
        senal * ventana
    )

    frecuencias = np.fft.rfftfreq(
        N,
        d=1.0 / fs
    )

    mascara = (
        (frecuencias >= f_min)
        &
        (frecuencias <= f_max)
    )

    return float(
        np.sum(
            np.abs(
                espectro[mascara]
            ) ** 2
        )
    )


def atenuacion_interferencia_db(
    entrada,
    salida,
    fs,
    frecuencia,
    ancho_hz=20.0
):
    """
    Atenuación de una interferencia dominante alrededor de una
    frecuencia conocida o estimada.

    Valor positivo -> la interferencia fue reducida.
    """
    mitad = ancho_hz / 2.0

    energia_entrada = energia_banda(
        entrada,
        fs,
        frecuencia - mitad,
        frecuencia + mitad
    )

    energia_salida = energia_banda(
        salida,
        fs,
        frecuencia - mitad,
        frecuencia + mitad
    )

    return float(
        10.0 * np.log10(
            max(
                energia_entrada,
                1e-30
            )
            /
            max(
                energia_salida,
                1e-30
            )
        )
    )


def medir_tiempo_bloque(
    funcion,
    bloque,
    repeticiones=30
):
    """
    Mide el tiempo medio de ejecución de una función sobre un bloque.

    La primera ejecución se usa como calentamiento.
    """
    bloque = np.asarray(
        bloque,
        dtype=np.float64
    )

    funcion(bloque)

    tiempos = []

    for _ in range(repeticiones):
        inicio = time.perf_counter()

        funcion(bloque)

        fin = time.perf_counter()

        tiempos.append(
            fin - inicio
        )

    return float(
        1000.0
        * np.mean(tiempos)
    )


def memoria_temporal_bytes(
    numero_secciones,
    N=1024,
    bytes_float=4
):
    """
    Estimación aproximada para implementación float32 del IIR.

    Coeficientes SOS:
        6 coeficientes por sección.

    Estados:
        2 estados por sección.

    Bloque:
        N muestras.
    """
    coeficientes = (
        numero_secciones
        * 6
        * bytes_float
    )

    estados = (
        numero_secciones
        * 2
        * bytes_float
    )

    bloque = (
        N
        * bytes_float
    )

    return {
        "coeficientes": coeficientes,
        "estados": estados,
        "bloque": bloque,
        "total": (
            coeficientes
            + estados
            + bloque
        )
    }


def memoria_espectral_bytes(
    N=1024,
    bytes_float=4
):
    """
    Estimación aproximada para procesamiento espectral float32.

    Se consideran:
        - bloque temporal
        - ventana
        - espectro complejo
        - ganancia espectral
        - buffer de solapamiento

    Para una RFFT de N muestras existen N/2 + 1 bins complejos.
    Un complejo float32 requiere 8 bytes.
    """
    bins = (
        N // 2 + 1
    )

    bloque = (
        N
        * bytes_float
    )

    ventana = (
        N
        * bytes_float
    )

    espectro = (
        bins
        * 2
        * bytes_float
    )

    ganancia = (
        bins
        * bytes_float
    )

    solapamiento = (
        N // 2
        * bytes_float
    )

    total = (
        bloque
        + ventana
        + espectro
        + ganancia
        + solapamiento
    )

    return {
        "bloque": bloque,
        "ventana": ventana,
        "espectro": espectro,
        "ganancia": ganancia,
        "solapamiento": solapamiento,
        "total": total
    }