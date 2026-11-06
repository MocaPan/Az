from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

BASE = Path("proyecto/pruebas_fisicas")
CAPTURAS = BASE / "capturas"
RESULTADOS = BASE / "resultados" / "P4_original"
RESULTADOS.mkdir(parents=True, exist_ok=True)

FS = 16000
N = 1024
SEGMENTO = 128
FRECUENCIA = 900

COLUMNAS = ("entrada", "iir", "espectral")
FILTROS = ("iir", "espectral")

ETIQUETAS = {
    "entrada": "Entrada contaminada",
    "iir": "Filtro IIR",
    "espectral": "Filtro FFT/IFFT"
}

FASES = {
    1: "Crecimiento, maximo y decaimiento",
    2: "Crecimiento, maximo y decaimiento",
    3: "Principalmente decaimiento"
}


def cargar(repeticion):
    ruta = CAPTURAS / f"P4_original_R{repeticion}.csv"

    with ruta.open(newline="", encoding="utf-8") as archivo:
        lector = csv.DictReader(archivo)

        if lector.fieldnames != [
            "indice", "entrada", "iir", "espectral"
        ]:
            raise ValueError(f"Columnas incorrectas: {ruta.name}")

        filas = list(lector)

    if len(filas) != N:
        raise ValueError(f"{ruta.name}: muestras incorrectas")

    indices = np.array(
        [float(fila["indice"]) for fila in filas]
    )

    if not np.array_equal(indices, np.arange(N)):
        raise ValueError(f"{ruta.name}: indices incorrectos")

    datos = {}

    for columna in COLUMNAS:
        x = np.array(
            [float(fila[columna]) for fila in filas],
            dtype=float
        )

        if not np.all(np.isfinite(x)):
            raise ValueError(
                f"{ruta.name}: datos no finitos en {columna}"
            )

        datos[columna] = x

    return datos


def potencia(x):
    return float(np.mean(np.asarray(x) ** 2))


def espectro(x):
    f = np.fft.rfftfreq(len(x), 1 / FS)
    ventana = np.hanning(len(x))

    magnitud = np.abs(
        np.fft.rfft(
            (x - np.mean(x)) * ventana
        )
    )

    return f, magnitud


def energia_banda(x, inferior, superior):
    f, magnitud = espectro(x)
    banda = (f >= inferior) & (f <= superior)
    return float(np.sum(magnitud[banda] ** 2))


def amplitudes_900(x):
    resultado = []

    t = np.arange(SEGMENTO) / FS
    ventana = np.hanning(SEGMENTO)

    portadora = np.exp(
        -2j * np.pi * FRECUENCIA * t
    )

    for i in range(N // SEGMENTO):
        segmento = x[
            i * SEGMENTO:(i + 1) * SEGMENTO
        ]

        coeficiente = np.sum(
            (segmento - segmento.mean())
            * ventana
            * portadora
        )

        amplitud = (
            2 * np.abs(coeficiente) / ventana.sum()
        )

        resultado.append(float(amplitud))

    return np.array(resultado)


def reduccion_db(entrada, salida):
    if entrada <= 0 or salida <= 0:
        raise ValueError("Energia no positiva")
    return float(10 * np.log10(entrada / salida))


registros = []
envolventes = []

print("\n=== ANALISIS P4 ORIGINAL ===")

for r in (1, 2, 3):
    datos = cargar(r)
    entrada = datos["entrada"]

    pot_entrada = potencia(entrada)
    alta_entrada = energia_banda(entrada, 2000, 8000)

    print(f"\n--- REPETICION R{r} ---")
    print("Fase:", FASES[r])
    print("RMS entrada:", f"{np.sqrt(pot_entrada):.2f}")

    amplitudes = {}

    for columna in COLUMNAS:
        amplitudes[columna] = amplitudes_900(
            datos[columna]
        )

        for i, valor in enumerate(amplitudes[columna]):
            envolventes.append({
                "repeticion": r,
                "senal": columna,
                "segmento_inicio_ms": i * 8,
                "segmento_fin_ms": (i + 1) * 8,
                "amplitud_900hz": valor
            })

    pico_entrada = float(
        np.max(amplitudes["entrada"])
    )

    indice_entrada = int(
        np.argmax(amplitudes["entrada"])
    )

    print(
        "Maximo 900 Hz entrada:",
        f"{pico_entrada:.2f}",
        f"en {indice_entrada*8}-{(indice_entrada+1)*8} ms"
    )

    for filtro in FILTROS:
        salida = datos[filtro]

        pico_salida = float(
            np.max(amplitudes[filtro])
        )

        indice_salida = int(
            np.argmax(amplitudes[filtro])
        )

        registro = {
            "repeticion": r,
            "filtro": filtro,
            "fase_capturada": FASES[r],
            "energia_total_relativa_pct":
                100 * potencia(salida) / pot_entrada,
            "reduccion_energia_alta_db":
                reduccion_db(
                    alta_entrada,
                    energia_banda(salida, 2000, 8000)
                ),
            "pico_900_entrada": pico_entrada,
            "pico_900_salida": pico_salida,
            "pico_900_relativo_pct":
                100 * pico_salida / pico_entrada,
            "segmento_maximo_entrada_ms":
                indice_entrada * 8,
            "segmento_maximo_salida_ms":
                indice_salida * 8,
            "rms_entrada":
                float(np.sqrt(pot_entrada)),
            "rms_salida":
                float(np.sqrt(potencia(salida)))
        }

        registros.append(registro)

        print(
            f"{filtro.upper()}: "
            f"pico 900 Hz={pico_salida:.2f}, "
            f"relativo={registro['pico_900_relativo_pct']:.2f}%, "
            f"energia total={registro['energia_total_relativa_pct']:.2f}%"
        )

    # Grafica 1: señales temporales
    tiempo = np.arange(N) * 1000 / FS

    plt.figure(figsize=(11, 5))

    for columna in COLUMNAS:
        plt.plot(
            tiempo,
            datos[columna],
            label=ETIQUETAS[columna],
            linewidth=0.9
        )

    plt.xlabel("Tiempo dentro del bloque (ms)")
    plt.ylabel("Amplitud digital")
    plt.title(f"P4 original R{r} - Respuesta temporal")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTADOS / f"P4_R{r}_temporal.png",
        dpi=180
    )
    plt.close()

    # Grafica 2: evolucion de 900 Hz
    centros = np.arange(8) * 8 + 4

    plt.figure(figsize=(11, 5))

    for columna in COLUMNAS:
        plt.plot(
            centros,
            amplitudes[columna],
            marker="o",
            label=ETIQUETAS[columna]
        )

    plt.xlabel("Tiempo dentro del bloque (ms)")
    plt.ylabel("Amplitud estimada de 900 Hz")
    plt.title(f"P4 original R{r} - Evolucion a 900 Hz")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTADOS / f"P4_R{r}_envolvente_900hz.png",
        dpi=180
    )
    plt.close()

    # Grafica 3: espectros
    plt.figure(figsize=(11, 5))

    for columna in COLUMNAS:
        f, mag = espectro(datos[columna])

        plt.plot(
            f,
            20 * np.log10(np.maximum(mag, 1e-12)),
            label=ETIQUETAS[columna],
            linewidth=0.9
        )

    plt.axvline(
        FRECUENCIA,
        linestyle="--",
        color="black",
        alpha=0.5,
        label="900 Hz"
    )

    plt.xlim(0, 4000)
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Magnitud espectral (dB relativos)")
    plt.title(f"P4 original R{r} - Espectros")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTADOS / f"P4_R{r}_espectro.png",
        dpi=180
    )
    plt.close()


# Guardar metricas por repeticion
ruta_metricas = RESULTADOS / "P4_metricas.csv"

with ruta_metricas.open(
    "w", newline="", encoding="utf-8"
) as archivo:
    escritor = csv.DictWriter(
        archivo,
        fieldnames=list(registros[0].keys())
    )
    escritor.writeheader()
    escritor.writerows(registros)


# Guardar la evolucion temporal de cada señal
ruta_envolventes = RESULTADOS / "P4_envolventes.csv"

with ruta_envolventes.open(
    "w", newline="", encoding="utf-8"
) as archivo:
    escritor = csv.DictWriter(
        archivo,
        fieldnames=list(envolventes[0].keys())
    )
    escritor.writeheader()
    escritor.writerows(envolventes)


# Resumen estadistico
metricas_resumen = (
    "energia_total_relativa_pct",
    "reduccion_energia_alta_db",
    "pico_900_relativo_pct",
    "rms_entrada",
    "rms_salida"
)

resumen = []

print("\n=== RESUMEN ESTADISTICO P4 ===")

for filtro in FILTROS:
    print(f"\nFILTRO: {filtro.upper()}")

    for metrica in metricas_resumen:
        valores = np.array([
            fila[metrica]
            for fila in registros
            if fila["filtro"] == filtro
        ])

        media = float(np.mean(valores))
        desviacion = float(np.std(valores, ddof=1))

        unidad = (
            "%" if metrica.endswith("_pct")
            else "dB" if metrica.endswith("_db")
            else "ADC"
        )

        print(
            f"{metrica:32} "
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


ruta_resumen = RESULTADOS / "P4_resumen.csv"

with ruta_resumen.open(
    "w", newline="", encoding="utf-8"
) as archivo:
    escritor = csv.DictWriter(
        archivo,
        fieldnames=list(resumen[0].keys())
    )
    escritor.writeheader()
    escritor.writerows(resumen)


print("\n=== ARCHIVOS GENERADOS ===")
print(ruta_metricas)
print(ruta_resumen)
print(ruta_envolventes)

for r in (1, 2, 3):
    print(f"P4_R{r}_temporal.png")
    print(f"P4_R{r}_envolvente_900hz.png")
    print(f"P4_R{r}_espectro.png")

print("\nANALISIS P4 ORIGINAL COMPLETADO")
