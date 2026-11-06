import numpy as np

from src.senales import (
    generar_p1,
    generar_p2,
    generar_p3,
    generar_p4,
    generar_p5,
    calcular_snr_db,
)


# =====================================================
# CONFIGURACIÓN
# =====================================================

generadores = {
    "P1": generar_p1,
    "P2": generar_p2,
    "P3": generar_p3,
    "P4": generar_p4,
    "P5": generar_p5,
}


# =====================================================
# PRUEBA 1
# CASOS DE REFERENCIA
# =====================================================

print("=" * 60)
print("PRUEBA DE SEÑALES - CASOS DE REFERENCIA")
print("=" * 60)

for nombre, generar in generadores.items():

    limpia, ruido, contaminada = generar()

    # Verificaciones básicas
    assert len(limpia) == len(ruido) == len(contaminada)
    assert np.all(np.isfinite(limpia))
    assert np.all(np.isfinite(ruido))
    assert np.all(np.isfinite(contaminada))

    snr = calcular_snr_db(
        limpia,
        contaminada
    )

    # Todas las pruebas se generan con SNR objetivo de 5 dB.
    assert np.isclose(
        snr,
        5.0,
        atol=1e-6
    )

    print(
        nombre,
        "-> SNR =",
        f"{snr:.3f}",
        "dB"
    )


# =====================================================
# PRUEBA 2
# SEMILLAS CONFIGURABLES
# =====================================================

print()
print("=" * 60)
print("PRUEBA DE SEÑALES - SEMILLAS")
print("=" * 60)

for nombre, generar in generadores.items():

    limpia_1, ruido_1, contaminada_1 = generar(
        semilla=1001
    )

    limpia_2, ruido_2, contaminada_2 = generar(
        semilla=1002
    )

    # La señal útil debe ser la misma.
    assert np.array_equal(
        limpia_1,
        limpia_2
    )

    # El ruido debe cambiar entre semillas.
    assert not np.array_equal(
        ruido_1,
        ruido_2
    )

    # Las dos realizaciones deben conservar la SNR objetivo.
    snr_1 = calcular_snr_db(
        limpia_1,
        contaminada_1
    )

    snr_2 = calcular_snr_db(
        limpia_2,
        contaminada_2
    )

    assert np.isclose(
        snr_1,
        5.0,
        atol=1e-6
    )

    assert np.isclose(
        snr_2,
        5.0,
        atol=1e-6
    )

    print(
        nombre,
        "| señal útil igual: True",
        "| ruido diferente: True",
        "| SNR:",
        f"{snr_1:.3f}",
        f"{snr_2:.3f}",
        "dB"
    )


print()
print("Todas las pruebas de señales PASARON.")
