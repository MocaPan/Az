from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

BASE = Path("proyecto/pruebas_fisicas")
CAPTURAS = BASE / "capturas"
RESULTADOS = BASE / "resultados" / "P2_original"
RESULTADOS.mkdir(parents=True, exist_ok=True)

FS = 16000
N = 1024
TONOS = (500, 900, 1300)
FILTROS = ("iir", "espectral")
COLUMNAS = ("entrada", "iir", "espectral")


def cargar(r):
    ruta = CAPTURAS / f"P2_original_R{r}.csv"

    with ruta.open(newline="", encoding="utf-8") as archivo:
        lector = csv.DictReader(archivo)
        filas = list(lector)

    if len(filas) != N:
        raise ValueError(f"{ruta.name}: se esperaban {N} muestras")

    datos = {
        c: np.array([float(fila[c]) for fila in filas])
        for c in COLUMNAS
    }

    if not all(np.all(np.isfinite(x)) for x in datos.values()):
        raise ValueError(f"{ruta.name}: valores no finitos")

    return datos


def espectro(x):
    x = np.asarray(x, dtype=float)
    f = np.fft.rfftfreq(len(x), 1 / FS)
    ventana = np.hanning(len(x))

    e = np.abs(
        np.fft.rfft((x - np.mean(x)) * ventana)
    ) ** 2

    return f, e


def energia_bandas(x):
    f, e = espectro(x)

    banda_ruido = (f >= 2000) & (f <= 8000)

    banda_util = np.zeros(len(f), dtype=bool)
    for hz in TONOS:
        banda_util |= np.abs(f - hz) <= 50

    return (
        float(np.sum(e[banda_util])),
        float(np.sum(e[banda_ruido]))
    )


def amplitud_tono(x, hz):
    t = np.arange(len(x)) / FS

    matriz = np.column_stack((
        np.sin(2 * np.pi * hz * t),
        np.cos(2 * np.pi * hz * t),
        np.ones(len(x))
    ))

    coef, *_ = np.linalg.lstsq(
        matriz, x, rcond=None
    )

    return float(np.hypot(coef[0], coef[1]))


def relacion_db(a, b):
    if a <= 0 or b <= 0:
        return float("nan")

    return float(10 * np.log10(a / b))


registros = []

print("\n=== ANALISIS P2 ORIGINAL ===")

for r in (1, 2, 3):
    datos = cargar(r)

    util_entrada, ruido_entrada = energia_bandas(
        datos["entrada"]
    )

    for filtro in FILTROS:
        util_salida, ruido_salida = energia_bandas(
            datos[filtro]
        )

        registro = {
            "repeticion": r,
            "filtro": filtro,
            "atenuacion_ruido_db": relacion_db(
                ruido_entrada, ruido_salida
            ),
            "delta_snr_espectral_db": (
                relacion_db(util_salida, ruido_salida)
                - relacion_db(util_entrada, ruido_entrada)
            )
        }

        for hz in TONOS:
            a_in = amplitud_tono(datos["entrada"], hz)
            a_out = amplitud_tono(datos[filtro], hz)

            registro[f"atenuacion_{hz}hz_db"] = relacion_db(
                a_in**2, a_out**2
            )

        registros.append(registro)

    # Grafica temporal
    t_ms = np.arange(N) * 1000 / FS

    plt.figure(figsize=(12, 5))

    for columna, etiqueta in (
        ("entrada", "Entrada contaminada"),
        ("iir", "Salida IIR"),
        ("espectral", "Salida FFT/IFFT")
    ):
        plt.plot(
            t_ms,
            datos[columna],
            label=etiqueta,
            linewidth=0.9
        )

    plt.xlim(0, 20)
    plt.xlabel("Tiempo (ms)")
    plt.ylabel("Amplitud digital")
    plt.title(f"P2 original - Comparacion temporal R{r}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        RESULTADOS / f"P2_R{r}_temporal.png",
        dpi=180
    )
    plt.close()

    # Grafica espectral
    plt.figure(figsize=(12, 5))

    for columna, etiqueta in (
        ("entrada", "Entrada contaminada"),
        ("iir", "Salida IIR"),
        ("espectral", "Salida FFT/IFFT")
    ):
        f, e = espectro(datos[columna])

        plt.plot(
            f,
            10 * np.log10(np.maximum(e, 1e-20)),
            label=etiqueta,
            linewidth=0.9
        )

    plt.axvspan(
        2000, 8000,
        alpha=0.08,
        label="Banda de ruido"
    )

    plt.xlim(0, 8000)
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Potencia espectral (dB relativos)")
    plt.title(f"P2 original - Espectros R{r}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        RESULTADOS / f"P2_R{r}_espectro.png",
        dpi=180
    )
    plt.close()


# Guardar metricas de cada repeticion
detalle = RESULTADOS / "P2_metricas.csv"

with detalle.open("w", newline="", encoding="utf-8") as f:
    escritor = csv.DictWriter(
        f, fieldnames=list(registros[0].keys())
    )
    escritor.writeheader()
    escritor.writerows(registros)


# Calcular medias y desviaciones estandar
resumen = []
metricas = [
    "atenuacion_ruido_db",
    "delta_snr_espectral_db",
    "atenuacion_500hz_db",
    "atenuacion_900hz_db",
    "atenuacion_1300hz_db"
]

for filtro in FILTROS:
    print(f"\nFILTRO: {filtro.upper()}")

    for metrica in metricas:
        valores = np.array([
            fila[metrica]
            for fila in registros
            if fila["filtro"] == filtro
        ])

        if not np.all(np.isfinite(valores)):
            raise ValueError(
                f"Resultados no validos: {filtro}, {metrica}"
            )

        media = float(np.mean(valores))
        desv = float(np.std(valores, ddof=1))

        print(
            f"{metrica:28} "
            f"media={media:8.2f} dB | "
            f"desv={desv:.2f} dB"
        )

        resumen.append({
            "filtro": filtro,
            "metrica": metrica,
            "media_db": round(media, 4),
            "desviacion_db": round(desv, 4)
        })


archivo_resumen = RESULTADOS / "P2_resumen.csv"

with archivo_resumen.open(
    "w", newline="", encoding="utf-8"
) as f:
    escritor = csv.DictWriter(
        f, fieldnames=list(resumen[0].keys())
    )
    escritor.writeheader()
    escritor.writerows(resumen)


print("\nArchivos generados:")
print(f"  {detalle}")
print(f"  {archivo_resumen}")

for r in (1, 2, 3):
    print(f"  P2_R{r}_temporal.png")
    print(f"  P2_R{r}_espectro.png")

print("\nANALISIS P2 ORIGINAL COMPLETADO")
