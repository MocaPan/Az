import numpy as np


# =====================================================
# MSE
# =====================================================

def calcular_mse(
    referencia,
    estimada
):
    referencia = np.asarray(
        referencia,
        dtype=np.float64
    )

    estimada = np.asarray(
        estimada,
        dtype=np.float64
    )

    if len(referencia) != len(estimada):
        raise ValueError(
            "Las señales deben tener la misma longitud."
        )

    error = referencia - estimada

    return np.mean(
        error ** 2
    )


# =====================================================
# SNR
# =====================================================

def calcular_snr_db(
    referencia,
    estimada
):
    referencia = np.asarray(
        referencia,
        dtype=np.float64
    )

    estimada = np.asarray(
        estimada,
        dtype=np.float64
    )

    if len(referencia) != len(estimada):
        raise ValueError(
            "Las señales deben tener la misma longitud."
        )

    error = referencia - estimada

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
# DELTA SNR
# =====================================================

def calcular_delta_snr_db(
    referencia,
    entrada,
    salida
):
    snr_entrada = calcular_snr_db(
        referencia,
        entrada
    )

    snr_salida = calcular_snr_db(
        referencia,
        salida
    )

    return (
        snr_salida
        - snr_entrada
    )


# =====================================================
# ENERGÍA
# =====================================================

def calcular_energia(
    senal
):
    senal = np.asarray(
        senal,
        dtype=np.float64
    )

    return np.sum(
        senal ** 2
    )


# =====================================================
# ENERGÍA ÚTIL CONSERVADA
# =====================================================

def energia_util_conservada_pct(
    referencia,
    salida
):
    referencia = np.asarray(
        referencia,
        dtype=np.float64
    )

    salida = np.asarray(
        salida,
        dtype=np.float64
    )

    if len(referencia) != len(salida):
        raise ValueError(
            "Las señales deben tener la misma longitud."
        )

    energia_ref = np.dot(
        referencia,
        referencia
    )

    if energia_ref <= 0:
        return 0.0

    # Proyección de la salida sobre
    # la referencia limpia.
    ganancia_util = (
        np.dot(
            salida,
            referencia
        )
        / energia_ref
    )

    return 100.0 * (
        ganancia_util ** 2
    )


# =====================================================
# ALINEACIÓN POR RETARDO
# =====================================================

def alinear_por_retardo(
    referencia,
    salida,
    max_retardo=64,
    min_retardo=0
):
    referencia = np.asarray(
        referencia,
        dtype=np.float64
    )

    salida = np.asarray(
        salida,
        dtype=np.float64
    )

    if referencia.ndim != 1 or salida.ndim != 1:
        raise ValueError("Las señales deben ser unidimensionales.")

    if len(referencia) != len(salida):
        raise ValueError("Las señales deben tener la misma longitud.")

    if not (0 <= min_retardo <= max_retardo < len(referencia)):
        raise ValueError("El intervalo de retardos no es válido.")

    mejor_retardo = min_retardo
    mejor_correlacion = -np.inf

    # Buscar únicamente los retardos permitidos
    for retardo in range(
        min_retardo,
        max_retardo + 1
    ):

        if retardo > 0:
            ref = referencia[:-retardo]
            out = salida[retardo:]
        else:
            ref = referencia
            out = salida

        # Correlación positiva normalizada
        energia_ref = np.dot(ref, ref)
        energia_out = np.dot(out, out)

        denominador = np.sqrt(
            energia_ref * energia_out
        )

        if denominador <= 0:
            continue

        correlacion = (
            np.dot(ref, out) / denominador
        )

        if correlacion > mejor_correlacion:
            mejor_correlacion = correlacion
            mejor_retardo = retardo

    # Alinear las señales
    if mejor_retardo > 0:
        referencia_alineada = referencia[:-mejor_retardo]
        salida_alineada = salida[mejor_retardo:]
    else:
        referencia_alineada = referencia
        salida_alineada = salida

    return (
        referencia_alineada,
        salida_alineada,
        mejor_retardo
    )
