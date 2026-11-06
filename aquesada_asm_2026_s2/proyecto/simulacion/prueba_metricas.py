import numpy as np

from src.senales import (
    generar_p1,
    generar_p2,
    generar_p3,
    generar_p4,
    generar_p5,
)

from src.filtros import (
    disenar_iir,
    filtrar_iir,
)

from src.filtrado_espectral import (
    filtrar_espectral,
)

from src.metricas import (
    calcular_mse,
    calcular_snr_db,
    energia_util_conservada_pct,
    alinear_por_retardo,
)


# =====================================================
# CONFIGURACIÓN
# =====================================================

sos = disenar_iir()

generadores = {
    "P1": generar_p1,
    "P2": generar_p2,
    "P3": generar_p3,
    "P4": generar_p4,
    "P5": generar_p5,
}


# =====================================================
# COMPARACIÓN
# =====================================================

for nombre, generar in generadores.items():

    limpia, ruido, contaminada = generar()

    # -----------------------------------------
    # FILTRADO IIR
    # -----------------------------------------

    salida_iir = filtrar_iir(
        contaminada,
        sos
    )

    (
        limpia_iir,
        salida_iir_alineada,
        retardo_iir
    ) = alinear_por_retardo(
        limpia,
        salida_iir,
        min_retardo=0,
        max_retardo=24
    )

    # Entrada equivalente para la comparación
    # alineada del IIR.
    contaminada_iir = contaminada[
        :len(limpia_iir)
    ]

    # -----------------------------------------
    # FILTRADO ESPECTRAL
    # -----------------------------------------

    salida_espectral = filtrar_espectral(
        contaminada
    )

    # -----------------------------------------
    # VERIFICACIONES BÁSICAS
    # -----------------------------------------

    assert len(salida_iir) == len(limpia)
    assert len(salida_espectral) == len(limpia)

    assert np.all(np.isfinite(salida_iir))
    assert np.all(np.isfinite(salida_espectral))

    assert 0 <= retardo_iir <= 24

    # -----------------------------------------
    # ENTRADA
    # -----------------------------------------

    snr_entrada = calcular_snr_db(
        limpia,
        contaminada
    )

    assert np.isclose(
        snr_entrada,
        5.0,
        atol=1e-3
    )

    # -----------------------------------------
    # IIR
    # -----------------------------------------

    mse_iir = calcular_mse(
        limpia_iir,
        salida_iir_alineada
    )

    snr_iir = calcular_snr_db(
        limpia_iir,
        salida_iir_alineada
    )

    snr_entrada_iir = calcular_snr_db(
        limpia_iir,
        contaminada_iir
    )

    delta_iir = (
        snr_iir
        - snr_entrada_iir
    )

    energia_iir = energia_util_conservada_pct(
        limpia_iir,
        salida_iir_alineada
    )

    # -----------------------------------------
    # ESPECTRAL
    # -----------------------------------------

    mse_esp = calcular_mse(
        limpia,
        salida_espectral
    )

    snr_esp = calcular_snr_db(
        limpia,
        salida_espectral
    )

    delta_esp = (
        snr_esp
        - snr_entrada
    )

    energia_esp = energia_util_conservada_pct(
        limpia,
        salida_espectral
    )

    # -----------------------------------------
    # VERIFICACIONES DE MÉTRICAS
    # -----------------------------------------

    valores = [
        mse_iir,
        snr_iir,
        delta_iir,
        energia_iir,
        mse_esp,
        snr_esp,
        delta_esp,
        energia_esp,
    ]

    assert np.all(
        np.isfinite(valores)
    )

    assert mse_iir >= 0
    assert mse_esp >= 0
    assert energia_iir >= 0
    assert energia_esp >= 0

    # -----------------------------------------
    # MOSTRAR
    # -----------------------------------------

    print("=" * 50)
    print(nombre)
    print("=" * 50)

    print(
        "SNR entrada:",
        f"{snr_entrada:.2f}",
        "dB"
    )

    print()

    print("IIR")
    print(
        "  Retardo estimado:",
        retardo_iir,
        "muestras"
    )
    print(
        "  MSE:",
        f"{mse_iir:.6f}"
    )
    print(
        "  SNR:",
        f"{snr_iir:.2f}",
        "dB"
    )
    print(
        "  Delta SNR:",
        f"{delta_iir:.2f}",
        "dB"
    )
    print(
        "  Energia util:",
        f"{energia_iir:.2f}",
        "%"
    )

    print()

    print("Espectral")
    print(
        "  MSE:",
        f"{mse_esp:.6f}"
    )
    print(
        "  SNR:",
        f"{snr_esp:.2f}",
        "dB"
    )
    print(
        "  Delta SNR:",
        f"{delta_esp:.2f}",
        "dB"
    )
    print(
        "  Energia util:",
        f"{energia_esp:.2f}",
        "%"
    )

    print()


print("Todas las pruebas de métricas PASARON.")
