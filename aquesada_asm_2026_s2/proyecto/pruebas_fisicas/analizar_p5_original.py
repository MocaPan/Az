from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

BASE = Path("proyecto/pruebas_fisicas")
CAPTURAS = BASE / "capturas"
RESULTADOS = BASE / "resultados" / "P5_original"
RESULTADOS.mkdir(parents=True, exist_ok=True)

FS = 16000
N = 1024
TONOS = (500, 900, 1300)
COLUMNAS = ("entrada", "iir", "espectral")
FILTROS = ("iir", "espectral")


def cargar(repeticion):
    ruta = CAPTURAS / f"P5_original_R{repeticion}.csv"

    with ruta.open(newline="", encoding="utf-8") as archivo:
        lector = csv.DictReader(archivo)

        esperadas = ["indice", *COLUMNAS]

        if lector.fieldnames != esperadas:
            raise ValueError(f"Columnas incorrectas: {ruta}")

        filas = list(lector)

    if len(filas) != N:
        raise ValueError(
            f"{ruta.name}: se esperaban {N} muestras"
        )

    indices = [int(f["indice"]) for f in filas]

    if indices != list(range(N)):
        raise ValueError(f"Indices incorrectos: {ruta.name}")

    datos = {
        columna: np.array(
            [float(f[columna]) for f in filas],
            dtype=float
        )
        for columna in COLUMNAS
    }

    for columna, x in datos.items():
        if not np.all(np.isfinite(x)):
            raise ValueError(
                f"{ruta.name}: valores invalidos en {columna}"
            )

    return datos


def espectro(x):
    x = np.asarray(x, dtype=float)
    f = np.fft.rfftfreq(len(x), 1 / FS)
    ventana = np.hanning(len(x))

    amplitud = np.abs(
        np.fft.rfft((x - np.mean(x)) * ventana)
    )

    return f, amplitud


def magnitud_tono(x, hz):
    f, amplitud = espectro(x)
    indice = np.argmin(np.abs(f - hz))
    return float(amplitud[indice])


def energia_banda(x, inferior, superior):
    f, amplitud = espectro(x)
    mascara = (f >= inferior) & (f <= superior)

    return float(np.sum(amplitud[mascara] ** 2))


def potencia_total(x):
    return float(np.mean(np.asarray(x) ** 2))


def atenuacion_amplitud(entrada, salida):
    if entrada <= 0 or salida <= 0:
        return float("nan")

    return float(20 * np.log10(entrada / salida))


def reduccion_energia(entrada, salida):
    if entrada <= 0 or salida <= 0:
        return float("nan")

    return float(10 * np.log10(entrada / salida))


registros = []

print("\n=== ANALISIS P5 ORIGINAL ===")

for repeticion in (1, 2, 3):
    datos = cargar(repeticion)
    entrada = datos["entrada"]

    magnitudes_entrada = {
        hz: magnitud_tono(entrada, hz)
        for hz in TONOS
    }

    energia_solapada_entrada = energia_banda(
        entrada, 800, 1500
    )

    energia_alta_entrada = energia_banda(
        entrada, 2000, 8000
    )

    potencia_entrada = potencia_total(entrada)

    print(f"\n--- REPETICION R{repeticion} ---")

    for filtro in FILTROS:
        salida = datos[filtro]

        registro = {
            "repeticion": repeticion,
            "filtro": filtro
        }

        for hz in TONOS:
            registro[f"atenuacion_{hz}hz_db"] = (
                atenuacion_amplitud(
                    magnitudes_entrada[hz],
                    magnitud_tono(salida, hz)
                )
            )

        registro["energia_solapada_relativa_pct"] = (
            100
            * energia_banda(salida, 800, 1500)
            / energia_solapada_entrada
        )

        registro["reduccion_energia_solapada_db"] = (
            reduccion_energia(
                energia_solapada_entrada,
                energia_banda(salida, 800, 1500)
            )
        )

        registro["reduccion_energia_alta_db"] = (
            reduccion_energia(
                energia_alta_entrada,
                energia_banda(salida, 2000, 8000)
            )
        )

        registro["energia_total_relativa_pct"] = (
            100 * potencia_total(salida) / potencia_entrada
        )

        registro["rms_entrada"] = float(
            np.sqrt(potencia_entrada)
        )

        registro["rms_salida"] = float(
            np.sqrt(potencia_total(salida))
        )

        registros.append(registro)

        print(
            f"{filtro.upper()}: "
            f"500 Hz={registro['atenuacion_500hz_db']:.2f} dB, "
            f"900 Hz={registro['atenuacion_900hz_db']:.2f} dB, "
            f"1300 Hz={registro['atenuacion_1300hz_db']:.2f} dB"
        )

    # Grafica temporal
    tiempo_ms = np.arange(N) * 1000 / FS

    plt.figure(figsize=(11, 5))

    for columna, etiqueta in (
        ("entrada", "Entrada contaminada"),
        ("iir", "Salida IIR"),
        ("espectral", "Salida FFT/IFFT")
    ):
        plt.plot(
            tiempo_ms,
            datos[columna],
            label=etiqueta,
            linewidth=0.9
        )

    plt.xlabel("Tiempo (ms)")
    plt.ylabel("Amplitud digital")
    plt.title(
        f"P5 original - Comparacion temporal R{repeticion}"
    )
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        RESULTADOS / f"P5_R{repeticion}_temporal.png",
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
        f, amplitud = espectro(datos[columna])

        plt.plot(
            f,
            20 * np.log10(np.maximum(amplitud, 1e-12)),
            label=etiqueta,
            linewidth=0.9
        )

    plt.axvspan(
        800, 1500,
        alpha=0.12,
        label="Banda solapada"
    )

    for hz in TONOS:
        plt.axvline(
            hz,
            linestyle="--",
            alpha=0.4
        )

    plt.xlim(0, 4000)
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Magnitud espectral (dB relativos)")
    plt.title(
        f"P5 original - Comparacion espectral R{repeticion}"
    )
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        RESULTADOS / f"P5_R{repeticion}_espectro.png",
        dpi=180
    )
    plt.close()


# Guardar metricas individuales
ruta_metricas = RESULTADOS / "P5_metricas.csv"

with ruta_metricas.open(
    "w", newline="", encoding="utf-8"
) as archivo:
    escritor = csv.DictWriter(
        archivo,
        fieldnames=list(registros[0].keys())
    )
    escritor.writeheader()
    escritor.writerows(registros)


# Resumen estadistico
metricas = [
    "atenuacion_500hz_db",
    "atenuacion_900hz_db",
    "atenuacion_1300hz_db",
    "energia_solapada_relativa_pct",
    "reduccion_energia_solapada_db",
    "reduccion_energia_alta_db",
    "energia_total_relativa_pct",
    "rms_entrada",
    "rms_salida"
]

resumen = []

print("\n=== RESUMEN ESTADISTICO P5 ===")

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
                f"Resultados no finitos: {filtro}, {metrica}"
            )

        media = float(np.mean(valores))
        desviacion = float(np.std(valores, ddof=1))

        unidad = (
            "%" if metrica.endswith("_pct")
            else "dB" if metrica.endswith("_db")
            else "ADC"
        )

        print(
            f"{metrica:35} "
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


ruta_resumen = RESULTADOS / "P5_resumen.csv"

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

for repeticion in (1, 2, 3):
    print(f"P5_R{repeticion}_temporal.png")
    print(f"P5_R{repeticion}_espectro.png")

print("\nANALISIS P5 ORIGINAL COMPLETADO")
