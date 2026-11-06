import numpy as np

from scipy.signal import (
    kaiserord,
    firwin,
    butter,
    sosfilt,
    freqz,
    sosfreqz,
    group_delay,
)


# =====================================================
# CONFIGURACIÓN GENERAL
# =====================================================

FS = 16000

F_PASS = 1300
F_STOP = 1800

ATENUACION_FIR_DB = 40

ORDEN_IIR = 12
FC_IIR = 1500


# =====================================================
# DISEÑO FIR
# =====================================================

def disenar_fir(
    fs=FS,
    f_pass=F_PASS,
    f_stop=F_STOP,
    atenuacion_db=ATENUACION_FIR_DB
):
    ancho_transicion_hz = (
        f_stop - f_pass
    )

    ancho_normalizado = (
        ancho_transicion_hz
        / (fs / 2)
    )

    num_taps, beta = kaiserord(
        atenuacion_db,
        ancho_normalizado
    )

    # FIR tipo I:
    # número impar de coeficientes
    if num_taps % 2 == 0:
        num_taps += 1

    orden = num_taps - 1

    fc = (
        f_pass + f_stop
    ) / 2

    coeficientes = firwin(
        num_taps,
        fc,
        window=("kaiser", beta),
        fs=fs
    )

    return {
        "coeficientes": coeficientes,
        "num_taps": num_taps,
        "orden": orden,
        "beta": beta,
        "fc": fc,
    }


# =====================================================
# RESPUESTA FIR
# =====================================================

def respuesta_fir(
    coeficientes,
    fs=FS,
    worN=16384
):
    frecuencias, H = freqz(
        coeficientes,
        worN=worN,
        fs=fs
    )

    magnitud_db = 20 * np.log10(
        np.maximum(
            np.abs(H),
            1e-12
        )
    )

    fase = np.unwrap(
        np.angle(H)
    )

    return frecuencias, H, magnitud_db, fase


# =====================================================
# RETARDO DE GRUPO FIR
# =====================================================

def retardo_grupo_fir(
    coeficientes,
    fs=FS,
    worN=4096
):
    frecuencias, retardo = group_delay(
        (
            coeficientes,
            [1.0]
        ),
        w=worN,
        fs=fs
    )

    return frecuencias, retardo


# =====================================================
# DISEÑO IIR
# =====================================================

def disenar_iir(
    fs=FS,
    orden=ORDEN_IIR,
    fc=FC_IIR
):
    sos = butter(
        orden,
        fc,
        btype="low",
        fs=fs,
        output="sos"
    )

    return sos


# =====================================================
# RESPUESTA IIR
# =====================================================

def respuesta_iir(
    sos,
    fs=FS,
    worN=16384
):
    frecuencias, H = sosfreqz(
        sos,
        worN=worN,
        fs=fs
    )

    magnitud_db = 20 * np.log10(
        np.maximum(
            np.abs(H),
            1e-12
        )
    )

    fase = np.unwrap(
        np.angle(H)
    )

    return frecuencias, H, magnitud_db, fase


# =====================================================
# FILTRADO IIR
# =====================================================

def filtrar_iir(
    senal,
    sos=None
):
    if sos is None:
        sos = disenar_iir()

    senal = np.asarray(
        senal,
        dtype=np.float64
    )

    return sosfilt(
        sos,
        senal
    )


# =====================================================
# COSTO COMPUTACIONAL
# =====================================================

def estimar_costo_fir(
    num_taps,
    fs=FS
):
    multiplicaciones_muestra = num_taps
    sumas_muestra = num_taps - 1

    multiplicaciones_segundo = (
        multiplicaciones_muestra
        * fs
    )

    return {
        "multiplicaciones_muestra":
            multiplicaciones_muestra,

        "sumas_muestra":
            sumas_muestra,

        "multiplicaciones_segundo":
            multiplicaciones_segundo,
    }


def estimar_costo_iir(
    num_sos,
    fs=FS
):
    # Aproximadamente 5 multiplicaciones
    # por biquad en DF-II transpuesta.

    multiplicaciones_muestra = (
        num_sos * 5
    )

    multiplicaciones_segundo = (
        multiplicaciones_muestra
        * fs
    )

    return {
        "multiplicaciones_muestra":
            multiplicaciones_muestra,

        "multiplicaciones_segundo":
            multiplicaciones_segundo,
    }


# =====================================================
# MEMORIA FLOAT32
# =====================================================

def estimar_memoria_fir(
    num_taps
):
    bytes_float = 4

    coeficientes = (
        num_taps
        * bytes_float
    )

    estados = (
        (num_taps - 1)
        * bytes_float
    )

    return {
        "coeficientes_bytes":
            coeficientes,

        "estados_bytes":
            estados,

        "total_bytes":
            coeficientes + estados,
    }


def estimar_memoria_iir(
    num_sos
):
    bytes_float = 4

    coeficientes = (
        num_sos
        * 5
        * bytes_float
    )

    estados = (
        num_sos
        * 2
        * bytes_float
    )

    return {
        "coeficientes_bytes":
            coeficientes,

        "estados_bytes":
            estados,

        "total_bytes":
            coeficientes + estados,
    }