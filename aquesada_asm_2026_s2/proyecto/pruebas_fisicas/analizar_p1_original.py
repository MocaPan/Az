from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

BASE = Path("proyecto/pruebas_fisicas")
CAPTURAS = BASE / "capturas"
RESULTADOS = BASE / "resultados" / "P1_original"
RESULTADOS.mkdir(parents=True, exist_ok=True)

FS = 16000
N = 1024
TONOS = (1000, 2500)
FILTROS = ("iir", "espectral")
COLUMNAS = ("entrada", "iir", "espectral")


def cargar(r):
    ruta = CAPTURAS / f"P1_original_R{r}.csv"

    with ruta.open(newline="", encoding="utf-8") as archivo:
        lector = csv.DictReader(archivo)
        columnas = set(lector.fieldnames or [])

        if not {"indice", *COLUMNAS}.issubset(columnas):
            raise ValueError(f"Columnas incorrectas: {ruta.name}")

        filas = list(lector)

    if len(filas) != N:
        raise ValueError(
            f"{ruta.name}: se esperaban {N} muestras"
        )

    indices = [int(f["indice"]) for f in filas]

    if indices != list(range(N)):
        raise ValueError(
            f"{ruta.name}: indices incorrectos"
        )

    datos = {
        nombre: np.array(
            [float(f[nombre]) for f in filas]
        )
        for nombre in COLUMNAS
    }

    if not all(
        np.all(np.isfinite(x)) for x in datos.values()
    ):
        raise ValueError(f"Datos invalidos en {ruta.name}")

    return datos


def amplitudes_tonos(x):
    # Ajuste conjunto de las dos senoides y el offset.
    # Evita atribuir componentes de un tono al otro.
    t = np.arange(len(x)) / FS

    matriz = np.column_stack([
        np.sin(2 * np.pi * 1000 * t),
        np.cos(2 * np.pi * 1000 * t),
        np.sin(2 * np.pi * 2500 * t),
        np.cos(2 * np.pi * 2500 * t),
        np.ones(len(x))
    ])

    coef, *_ = np.linalg.lstsq(
        matriz, x, rcond=None
    )

    return {
        1000: float(np.hypot(coef[0], coef[1])),
        2500: float(np.hypot(coef[2], coef[3]))
    }


def atenuacion_db(amplitud_entrada, amplitud_salida):
    if amplitud_entrada <= 0 or amplitud_salida <= 0:
        raise ValueError("Amplitud no positiva")

    return float(
        20 * np.log10(
            amplitud_entrada / amplitud_salida
        )
    )


def espectro(x):
    ventana = np.hanning(len(x))
    f = np.fft.rfftfreq(len(x), 1 / FS)
    e = np.abs(
        np.fft.rfft((x - x.mean()) * ventana)
    )
    return f, e


registros = []

print("\n=== ANALISIS P1 ORIGINAL ===")

for r in (1, 2, 3):
    datos = cargar(r)

    amplitudes_entrada = amplitudes_tonos(
        datos["entrada"]
    )

    for filtro in FILTROS:
        amplitudes_salida = amplitudes_tonos(
            datos[filtro]
        )

        atenuacion_util = atenuacion_db(
            amplitudes_entrada[1000],
            amplitudes_salida[1000]
        )

        atenuacion_ruido = atenuacion_db(
            amplitudes_entrada[2500],
            amplitudes_salida[2500]
        )

        registros.append({
            "repeticion": r,
            "filtro": filtro,
            "atenuacion_1000hz_db": atenuacion_util,
            "atenuacion_2500hz_db": atenuacion_ruido,
            "amplitud_util_entrada": amplitudes_entrada[1000],
            "amplitud_util_salida": amplitudes_salida[1000],
            "amplitud_ruido_entrada": amplitudes_entrada[2500],
            "amplitud_ruido_salida": amplitudes_salida[2500]
        })

    # Grafica temporal
    t_ms = np.arange(N) * 1000 / FS

    plt.figure(figsize=(12, 5))

    for nombre, etiqueta in (
        ("entrada", "Entrada contaminada"),
        ("iir", "Salida IIR"),
        ("espectral", "Salida FFT/IFFT")
    ):
        plt.plot(
            t_ms,
            datos[nombre],
            linewidth=1,
            label=etiqueta
        )

    plt.xlim(0, 15)
    plt.xlabel("Tiempo (ms)")
    plt.ylabel("Amplitud digital")
    plt.title(f"P1 original - Comparacion temporal R{r}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        RESULTADOS / f"P1_R{r}_temporal.png",
        dpi=180
    )
    plt.close()

    # Grafica espectral
    plt.figure(figsize=(12, 5))

    for nombre, etiqueta in (
        ("entrada", "Entrada contaminada"),
        ("iir", "Salida IIR"),
        ("espectral", "Salida FFT/IFFT")
    ):
        f, e = espectro(datos[nombre])

        plt.plot(
            f,
            20 * np.log10(np.maximum(e, 1e-12)),
            label=etiqueta,
            linewidth=1
        )

    plt.axvline(
        1000, linestyle="--", alpha=0.5,
        label="Tono util 1000 Hz"
    )

    plt.axvline(
        2500, linestyle="--", alpha=0.5,
        label="Interferencia 2500 Hz"
    )

    plt.xlim(0, 3500)
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Magnitud espectral (dB relativos)")
    plt.title(f"P1 original - Comparacion espectral R{r}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        RESULTADOS / f"P1_R{r}_espectro.png",
        dpi=180
    )
    plt.close()


# Guardar metricas individuales
detalle = RESULTADOS / "P1_metricas.csv"

with detalle.open(
    "w", newline="", encoding="utf-8"
) as archivo:
    escritor = csv.DictWriter(
        archivo,
        fieldnames=list(registros[0].keys())
    )
    escritor.writeheader()
    escritor.writerows(registros)


# Calcular estadisticas de tres repeticiones
resumen = []

for filtro in FILTROS:
    print(f"\nFILTRO: {filtro.upper()}")

    for metrica in (
        "atenuacion_1000hz_db",
        "atenuacion_2500hz_db"
    ):
        valores = np.array([
            fila[metrica]
            for fila in registros
            if fila["filtro"] == filtro
        ])

        media = float(np.mean(valores))
        desviacion = float(np.std(valores, ddof=1))

        print(
            f"{metrica:25} "
            f"media={media:8.2f} dB | "
            f"desv={desviacion:.2f} dB"
        )

        resumen.append({
            "filtro": filtro,
            "metrica": metrica,
            "media_db": round(media, 4),
            "desviacion_db": round(desviacion, 4)
        })


archivo_resumen = RESULTADOS / "P1_resumen.csv"

with archivo_resumen.open(
    "w", newline="", encoding="utf-8"
) as archivo:
    escritor = csv.DictWriter(
        archivo,
        fieldnames=list(resumen[0].keys())
    )
    escritor.writeheader()
    escritor.writerows(resumen)


print("\nArchivos generados:")
print(f"  {detalle}")
print(f"  {archivo_resumen}")

for r in (1, 2, 3):
    print(f"  P1_R{r}_temporal.png")
    print(f"  P1_R{r}_espectro.png")

print("\nANALISIS P1 COMPLETADO")
