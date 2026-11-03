import numpy as np


def resolucion_frecuencial(fs, N):
    return fs / N


def duracion_bloque(fs, N):
    return N / fs


def memoria_bloque_bytes(N, bytes_por_muestra=4):
    return N * bytes_por_muestra


def generar_tono(frecuencia, fs, N):
    n = np.arange(N)

    return np.sin(
        2 * np.pi * frecuencia * n / fs
    )


def ventana_rectangular(N):
    return np.ones(N)


def ventana_hann(N):
    return np.hanning(N)


def aplicar_ventana(senal, ventana):
    return senal * ventana


def error_simetria_conjugada(X):
    N = len(X)

    errores = [
        np.abs(
            X[k]
            - np.conj(X[N - k])
        )
        for k in range(1, N)
    ]

    return np.max(errores)


def preservar_simetria_conjugada(X, k):
    X_mod = X.copy()
    N = len(X)

    if k == 0:
        X_mod[k] = np.real(X_mod[k])
        return X_mod

    if N % 2 == 0 and k == N // 2:
        X_mod[k] = np.real(X_mod[k])
        return X_mod

    X_mod[N - k] = np.conj(
        X_mod[k]
    )

    return X_mod


def energia_en_banda(
    espectro,
    frecuencias,
    f_min,
    f_max
):
    mascara = (
        (frecuencias >= f_min)
        &
        (frecuencias < f_max)
    )

    return np.sum(
        np.abs(espectro[mascara]) ** 2
    )


def distribucion_energia_bandas(
    espectro,
    frecuencias,
    bandas
):
    energia_total = np.sum(
        np.abs(espectro) ** 2
    )

    if energia_total == 0:
        return []

    resultados = []

    for f_min, f_max in bandas:
        energia = energia_en_banda(
            espectro,
            frecuencias,
            f_min,
            f_max
        )

        porcentaje = (
            100.0
            * energia
            / energia_total
        )

        resultados.append(
            {
                "f_min": f_min,
                "f_max": f_max,
                "energia": energia,
                "porcentaje": porcentaje
            }
        )

    return resultados