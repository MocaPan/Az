import numpy as np

from src.senales import (
    generar_p1,
    generar_p2,
    generar_p3,
    generar_p4,
    generar_p5,
)

from src.filtrado_espectral import (
    filtrar_espectral,
    obtener_parametros,
)


# =====================================================
# PARÁMETROS
# =====================================================

print("Parametros espectrales:")

for clave, valor in obtener_parametros().items():
    print(
        clave,
        "=",
        valor
    )

print()


# =====================================================
# PRUEBAS P1-P5
# =====================================================

generadores = {
    "P1": generar_p1,
    "P2": generar_p2,
    "P3": generar_p3,
    "P4": generar_p4,
    "P5": generar_p5,
}


for nombre, generar in generadores.items():

    limpia, ruido, contaminada = generar()

    salida = filtrar_espectral(
        contaminada
    )

    misma_longitud = (
        len(salida)
        == len(contaminada)
    )

    valores_finitos = np.all(
        np.isfinite(salida)
    )

    rms_entrada = np.sqrt(
        np.mean(
            contaminada ** 2
        )
    )

    rms_salida = np.sqrt(
        np.mean(
            salida ** 2
        )
    )

    print(nombre)
    print(
        "  Longitud correcta:",
        misma_longitud
    )
    print(
        "  Valores finitos:",
        valores_finitos
    )
    print(
        "  RMS entrada:",
        f"{rms_entrada:.3f}"
    )
    print(
        "  RMS salida:",
        f"{rms_salida:.3f}"
    )
    print()