from src.filtros import (
    disenar_fir,
    respuesta_fir,
    retardo_grupo_fir,
    disenar_iir,
    respuesta_iir,
    estimar_costo_fir,
    estimar_costo_iir,
    estimar_memoria_fir,
    estimar_memoria_iir,
)


# =====================================================
# FIR
# =====================================================

fir = disenar_fir()

coef_fir = fir["coeficientes"]

print("FIR")
print("Num taps:", fir["num_taps"])
print("Orden:", fir["orden"])
print("Beta:", fir["beta"])
print("fc:", fir["fc"], "Hz")

print()


# =====================================================
# IIR
# =====================================================

sos = disenar_iir()

print("IIR")
print("Secciones SOS:", len(sos))

print()


# =====================================================
# COSTO
# =====================================================

costo_fir = estimar_costo_fir(
    fir["num_taps"]
)

costo_iir = estimar_costo_iir(
    len(sos)
)

print("Costo FIR:")
print(
    "Multiplicaciones/muestra:",
    costo_fir["multiplicaciones_muestra"]
)
print(
    "Sumas/muestra:",
    costo_fir["sumas_muestra"]
)
print(
    "Multiplicaciones/s:",
    costo_fir["multiplicaciones_segundo"]
)

print()

print("Costo IIR:")
print(
    "Multiplicaciones/muestra:",
    costo_iir["multiplicaciones_muestra"]
)
print(
    "Multiplicaciones/s:",
    costo_iir["multiplicaciones_segundo"]
)

print()


# =====================================================
# MEMORIA
# =====================================================

mem_fir = estimar_memoria_fir(
    fir["num_taps"]
)

mem_iir = estimar_memoria_iir(
    len(sos)
)

print("Memoria FIR:")
print(
    "Coeficientes:",
    mem_fir["coeficientes_bytes"],
    "bytes"
)
print(
    "Estados:",
    mem_fir["estados_bytes"],
    "bytes"
)
print(
    "Total:",
    mem_fir["total_bytes"],
    "bytes"
)

print()

print("Memoria IIR:")
print(
    "Coeficientes:",
    mem_iir["coeficientes_bytes"],
    "bytes"
)
print(
    "Estados:",
    mem_iir["estados_bytes"],
    "bytes"
)
print(
    "Total:",
    mem_iir["total_bytes"],
    "bytes"
)

print()

print(
    "Relacion memoria FIR/IIR:",
    mem_fir["total_bytes"]
    / mem_iir["total_bytes"]
)