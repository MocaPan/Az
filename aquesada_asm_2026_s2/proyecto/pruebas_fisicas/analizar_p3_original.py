from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

BASE = Path("proyecto/pruebas_fisicas")
CAPTURAS = BASE / "capturas"
RESULTADOS = BASE / "resultados" / "P3_original"
RESULTADOS.mkdir(parents=True, exist_ok=True)

FS = 16000
N = 1024
FILTROS = ("iir", "espectral")
COLUMNAS = ("entrada", "iir", "espectral")


def cargar(r):
    ruta = CAPTURAS / f"P3_original_R{r}.csv"

    with ruta.open(newline="", encoding="utf-8") as archivo:
        lector = csv.DictReader(archivo)
        if not {"indice", *COLUMNAS}.issubset(lector.fieldnames or []):
            raise ValueError(f"Columnas incorrectas: {ruta.name}")
        filas = list(lector)

    if len(filas) != N:
        raise ValueError(f"{ruta.name}: muestras incorrectas")

    if [int(f["indice"]) for f in filas] != list(range(N)):
        raise ValueError(f"{ruta.name}: indices incorrectos")

    datos = {
        c: np.array([float(f[c]) for f in filas])
        for c in COLUMNAS
    }

    if not all(np.all(np.isfinite(x)) for x in datos.values()):
        raise ValueError(f"{ruta.name}: datos no validos")

    return datos


def espectro(x):
    f = np.fft.rfftfreq(len(x), 1 / FS)
    ventana = np.hanning(len(x))
    e = np.abs(
        np.fft.rfft((x - np.mean(x)) * ventana)
    ) ** 2
    return f, e


def energia_banda(x, inferior, superior):
    f, e = espectro(x)
    mascara = (f >= inferior) & (f <= superior)
    return float(np.sum(e[mascara]))


def potencia_total(x):
    return float(np.mean(np.asarray(x) ** 2))


def reduccion_db(entrada, salida):
    if entrada <= 0 or salida <= 0:
        raise ValueError("Energia no positiva")
    return float(10 * np.log10(entrada / salida))


registros = []

print("\n=== ANALISIS P3 ORIGINAL ===")

for r in (1, 2, 3):
    datos = cargar(r)
    entrada = datos["entrada"]

    p_entrada = potencia_total(entrada)
    baja_entrada = energia_banda(entrada, 100, 2000)
    alta_entrada = energia_banda(entrada, 2000, 8000)

    for filtro in FILTROS:
        salida = datos[filtro]

        p_salida = potencia_total(salida)
        baja_salida = energia_banda(salida, 100, 2000)
        alta_salida = energia_banda(salida, 2000, 8000)

        registros.append({
            "repeticion": r,
            "filtro": filtro,
            "energia_total_relativa_pct":
                100 * p_salida / p_entrada,
            "energia_baja_relativa_pct":
                100 * baja_salida / baja_entrada,
            "reduccion_energia_alta_db":
                reduccion_db(alta_entrada, alta_salida),
            "rms_entrada": float(np.sqrt(p_entrada)),
            "rms_salida": float(np.sqrt(p_salida))
        })

    # Grafica temporal
    t = np.arange(N) * 1000 / FS

    plt.figure(figsize=(11, 5))

    for columna, etiqueta in (
        ("entrada", "Entrada contaminada"),
        ("iir", "Salida IIR"),
        ("espectral", "Salida FFT/IFFT")
    ):
        plt.plot(
            t, datos[columna],
            label=etiqueta,
            linewidth=0.9
        )

    plt.xlim(0, 30)
    plt.xlabel("Tiempo (ms)")
    plt.ylabel("Amplitud digital")
    plt.title(f"P3 original - Comparacion temporal R{r}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        RESULTADOS / f"P3_R{r}_temporal.png",
        dpi=180
    )
    plt.close()

    # Grafica espectral
    plt.figure(figsize=(11, 5))

    for columna, etiqueta in (
        ("entrada", "Entrada contaminada"),
        ("iir", "Salida IIR"),
        ("espectral", "Salida FFT/IFFT")
    ):
        f, energia = espectro(datos[columna])

        plt.plot(
            f,
            10 * np.log10(np.maximum(energia, 1e-20)),
            label=etiqueta,
            linewidth=0.9
        )

    for hz in (400, 750, 1200):
        plt.axvline(hz, linestyle="--", alpha=0.3)

    plt.xlim(0, 8000)
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Potencia espectral (dB relativos)")
    plt.title(f"P3 original - Espectros R{r}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        RESULTADOS / f"P3_R{r}_espectro.png",
        dpi=180
    )
    plt.close()


# Guardar valores por repeticion
detalle = RESULTADOS / "P3_metricas.csv"

with detalle.open("w", newline="", encoding="utf-8") as archivo:
    escritor = csv.DictWriter(
        archivo, fieldnames=list(registros[0].keys())
    )
    escritor.writeheader()
    escritor.writerows(registros)


# Estadisticas
metricas = (
    "energia_total_relativa_pct",
    "energia_baja_relativa_pct",
    "reduccion_energia_alta_db",
    "rms_entrada",
    "rms_salida"
)

resumen = []

for filtro in FILTROS:
    print(f"\nFILTRO: {filtro.upper()}")

    for metrica in metricas:
        valores = np.array([
            fila[metrica]
            for fila in registros
            if fila["filtro"] == filtro
        ])

        media = float(np.mean(valores))
        desviacion = float(np.std(valores, ddof=1))

        unidad = "dB" if metrica.endswith("_db") else (
            "%" if metrica.endswith("_pct") else "ADC"
        )

        print(
            f"{metrica:30} "
            f"media={media:9.2f} {unidad} | "
            f"desv={desviacion:.2f} {unidad}"
        )

        resumen.append({
            "filtro": filtro,
            "metrica": metrica,
            "media": round(media, 4),
            "desviacion": round(desviacion, 4),
            "unidad": unidad
        })


archivo_resumen = RESULTADOS / "P3_resumen.csv"

with archivo_resumen.open(
    "w", newline="", encoding="utf-8"
) as archivo:
    escritor = csv.DictWriter(
        archivo, fieldnames=list(resumen[0].keys())
    )
    escritor.writeheader()
    escritor.writerows(resumen)


print("\n=== ARCHIVOS GENERADOS ===")
print(detalle)
print(archivo_resumen)

for r in (1, 2, 3):
    print(f"P3_R{r}_temporal.png")
    print(f"P3_R{r}_espectro.png")

print("\nANALISIS P3 ORIGINAL COMPLETADO")
