from pathlib import Path
import csv

import numpy as np
from scipy.io import wavfile

from src.filtros_temporales import (
    disenar_filtro,
    ecuacion_diferencias
)

from src.filtrado_espectral import (
    filtrar_espectral
)

from src.metricas import (
    mse,
    snr_db,
    alinear_por_retardo,
    energia_util_conservada_pct,
    atenuacion_interferencia_db,
    medir_tiempo_bloque,
    memoria_temporal_bytes,
    memoria_espectral_bytes
)


RAIZ = Path(
    __file__
).resolve().parent

SALIDA = (
    RAIZ
    / "resultados"
)

SALIDA.mkdir(
    parents=True,
    exist_ok=True
)


FS = 16000
N_BLOQUE = 1024

FRECUENCIA_INTERFERENCIA_CASO_1 = 1800.0


def leer_wav(ruta):
    fs, datos = wavfile.read(
        ruta
    )

    if np.issubdtype(
        datos.dtype,
        np.integer
    ):
        info = np.iinfo(
            datos.dtype
        )

        escala = float(
            max(
                abs(int(info.min)),
                abs(int(info.max))
            )
        )

        datos = (
            datos.astype(
                np.float64
            )
            / escala
        )

    else:
        datos = datos.astype(
            np.float64
        )

    if datos.ndim == 2:
        datos = datos.mean(
            axis=1
        )

    return fs, datos


def evaluar_caso(
    numero,
    sos
):
    carpeta = (
        RAIZ
        / "pruebas"
        / f"caso_{numero}"
    )

    fs_limpia, limpia = leer_wav(
        carpeta
        / "01_referencia_limpia.wav"
    )

    fs_entrada, entrada = leer_wav(
        carpeta
        / "03_entrada_contaminada.wav"
    )

    if (
        fs_limpia != FS
        or fs_entrada != FS
    ):
        raise ValueError(
            f"Frecuencia de muestreo inválida en caso {numero}"
        )

    if len(limpia) != len(entrada):
        raise ValueError(
            f"Longitudes diferentes en caso {numero}"
        )

    # ========================================================
    # Entrada
    # ========================================================

    snr_entrada = snr_db(
        limpia,
        entrada
    )

    mse_entrada = mse(
        limpia,
        entrada
    )

    # ========================================================
    # Filtro temporal causal
    # ========================================================

    temporal = ecuacion_diferencias(
        entrada,
        sos
    )

    snr_temporal_causal = snr_db(
        limpia,
        temporal
    )

    mse_temporal_causal = mse(
        limpia,
        temporal
    )

    (
        limpia_temporal,
        temporal_alineada,
        retardo_temporal
    ) = alinear_por_retardo(
        limpia,
        temporal,
        max_retardo=64
    )

    snr_temporal_alineada = snr_db(
        limpia_temporal,
        temporal_alineada
    )

    mse_temporal_alineada = mse(
        limpia_temporal,
        temporal_alineada
    )

    energia_temporal = (
        energia_util_conservada_pct(
            limpia_temporal,
            temporal_alineada
        )
    )

    # ========================================================
    # Filtro espectral
    # ========================================================

    espectral = filtrar_espectral(
        entrada,
        fs=FS,
        N=N_BLOQUE,
        reduccion=1.0
    )

    snr_espectral = snr_db(
        limpia,
        espectral
    )

    mse_espectral = mse(
        limpia,
        espectral
    )

    energia_espectral = (
        energia_util_conservada_pct(
            limpia,
            espectral
        )
    )

    # ========================================================
    # Atenuación de interferencia dominante
    # ========================================================

    if numero == 1:
        atenuacion_temporal = (
            atenuacion_interferencia_db(
                entrada,
                temporal,
                FS,
                FRECUENCIA_INTERFERENCIA_CASO_1
            )
        )

        atenuacion_espectral = (
            atenuacion_interferencia_db(
                entrada,
                espectral,
                FS,
                FRECUENCIA_INTERFERENCIA_CASO_1
            )
        )

    else:
        # En casos 2 y 3 no existe una única interferencia
        # tonal dominante equivalente al Caso 1.
        atenuacion_temporal = np.nan
        atenuacion_espectral = np.nan

    return {
        "caso": numero,

        "snr_entrada_db":
            snr_entrada,

        "mse_entrada":
            mse_entrada,

        "snr_temporal_causal_db":
            snr_temporal_causal,

        "mse_temporal_causal":
            mse_temporal_causal,

        "retardo_temporal_muestras":
            retardo_temporal,

        "snr_temporal_alineada_db":
            snr_temporal_alineada,

        "delta_snr_temporal_db":
            snr_temporal_alineada
            - snr_entrada,

        "mse_temporal_alineada":
            mse_temporal_alineada,

        "energia_util_temporal_pct":
            energia_temporal,

        "atenuacion_interferencia_temporal_db":
            atenuacion_temporal,

        "snr_espectral_db":
            snr_espectral,

        "delta_snr_espectral_db":
            snr_espectral
            - snr_entrada,

        "mse_espectral":
            mse_espectral,

        "energia_util_espectral_pct":
            energia_espectral,

        "atenuacion_interferencia_espectral_db":
            atenuacion_espectral
    }


def medir_rendimiento(
    sos,
    ejemplo
):
    bloque = ejemplo[
        :N_BLOQUE
    ]

    tiempo_temporal = (
        medir_tiempo_bloque(
            lambda x:
                ecuacion_diferencias(
                    x,
                    sos
                ),
            bloque
        )
    )

    tiempo_espectral = (
        medir_tiempo_bloque(
            lambda x:
                filtrar_espectral(
                    x,
                    fs=FS,
                    N=N_BLOQUE,
                    reduccion=1.0
                ),
            bloque
        )
    )

    memoria_temporal = (
        memoria_temporal_bytes(
            numero_secciones=len(sos),
            N=N_BLOQUE
        )
    )

    memoria_espectral = (
        memoria_espectral_bytes(
            N=N_BLOQUE
        )
    )

    return {
        "tiempo_temporal_ms_bloque":
            tiempo_temporal,

        "tiempo_espectral_ms_bloque":
            tiempo_espectral,

        "memoria_temporal_bytes":
            memoria_temporal["total"],

        "memoria_espectral_bytes":
            memoria_espectral["total"]
    }


def guardar_csv(
    ruta,
    filas
):
    with ruta.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as archivo:

        escritor = csv.DictWriter(
            archivo,
            fieldnames=list(
                filas[0].keys()
            )
        )

        escritor.writeheader()
        escritor.writerows(
            filas
        )


def main():
    sos = disenar_filtro()

    resultados = []

    for caso in (
        1,
        2,
        3
    ):
        resultado = evaluar_caso(
            caso,
            sos
        )

        resultados.append(
            resultado
        )

    # Se usa el Caso 1 para la medición de tiempo,
    # porque el tamaño de bloque es igual en todos.
    _, ejemplo = leer_wav(
        RAIZ
        / "pruebas"
        / "caso_1"
        / "03_entrada_contaminada.wav"
    )

    rendimiento = medir_rendimiento(
        sos,
        ejemplo
    )

    for fila in resultados:
        fila.update(
            rendimiento
        )

    ruta_csv = (
        SALIDA
        / "metricas.csv"
    )

    guardar_csv(
        ruta_csv,
        resultados
    )

    print()
    print(
        "========================================"
    )
    print(
        "      COMPARACION DE METODOS"
    )
    print(
        "========================================"
    )

    for fila in resultados:
        print()
        print(
            f"Caso {fila['caso']}"
        )

        print(
            "  SNR entrada:"
            f" {fila['snr_entrada_db']:.2f} dB"
        )

        print(
            "  Temporal alineado:"
            f" {fila['snr_temporal_alineada_db']:.2f} dB"
        )

        print(
            "  Delta temporal:"
            f" {fila['delta_snr_temporal_db']:+.2f} dB"
        )

        print(
            "  Espectral:"
            f" {fila['snr_espectral_db']:.2f} dB"
        )

        print(
            "  Delta espectral:"
            f" {fila['delta_snr_espectral_db']:+.2f} dB"
        )

        print(
            "  Energía útil temporal:"
            f" {fila['energia_util_temporal_pct']:.2f} %"
        )

        print(
            "  Energía útil espectral:"
            f" {fila['energia_util_espectral_pct']:.2f} %"
        )

    print()
    print(
        "Tiempo temporal:"
        f" {rendimiento['tiempo_temporal_ms_bloque']:.4f} ms/bloque"
    )

    print(
        "Tiempo espectral:"
        f" {rendimiento['tiempo_espectral_ms_bloque']:.4f} ms/bloque"
    )

    print(
        "Memoria temporal:"
        f" {rendimiento['memoria_temporal_bytes']} bytes"
    )

    print(
        "Memoria espectral:"
        f" {rendimiento['memoria_espectral_bytes']} bytes"
    )

    print()
    print(
        "CSV guardado en:",
        ruta_csv
    )


if __name__ == "__main__":
    main()