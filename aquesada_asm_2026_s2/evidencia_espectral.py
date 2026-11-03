"""Evidencias reproducibles del filtrado espectral para la Tarea 1 (2.4).

Ejecutar desde la raíz del repositorio: python evidencia_espectral.py
No modifica los WAV de prueba ni el módulo src/filtrado_espectral.py.
"""

from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile

from src.filtrado_espectral import filtrar_espectral

RAIZ = Path(__file__).resolve().parent
SALIDA = RAIZ / "resultados" / "espectral"
FS_ESPERADA = 16000
TAM_BLOQUE = 1024
REDUCCION_BASE = 1.0
REDUCCIONES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)


def leer_wav(ruta):
    """Lee WAV mono PCM y lo convierte a flotante, en unidades normalizadas."""
    fs, datos = wavfile.read(ruta)
    if datos.ndim != 1:
        raise ValueError(f"Se esperaba WAV mono: {ruta}")
    if np.issubdtype(datos.dtype, np.integer):
        info = np.iinfo(datos.dtype)
        escala = float(max(abs(int(info.min)), abs(int(info.max))))
        datos = datos.astype(np.float64) / escala
    else:
        datos = datos.astype(np.float64)
    return fs, datos


def snr_db(referencia, estimacion):
    energia = np.sum(referencia ** 2)
    error = np.sum((referencia - estimacion) ** 2)
    return float(10 * np.log10(energia / max(error, 1e-30)))


def metricas(referencia, entrada, salida):
    mse = float(np.mean((referencia - salida) ** 2))
    antes = snr_db(referencia, entrada)
    despues = snr_db(referencia, salida)
    energia_ref = float(np.sum(referencia ** 2))
    energia_sal = float(np.sum(salida ** 2))
    # Es una relación de energías, no una medición de energía útil recuperada.
    return {
        "snr_entrada_db": antes,
        "snr_salida_db": despues,
        "mejora_snr_db": despues - antes,
        "mse_salida": mse,
        "energia_salida_sobre_referencia_pct": 100 * energia_sal / energia_ref,
    }


def dibujar_caso(numero, fs, limpia, entrada, filtrada, detalles):
    tiempo = np.arange(len(limpia)) / fs
    tramo = min(len(limpia), int(fs * 0.035))
    frecuencias, psd_limpia = signal.welch(limpia, fs, nperseg=4096)
    _, psd_entrada = signal.welch(entrada, fs, nperseg=4096)
    _, psd_filtrada = signal.welch(filtrada, fs, nperseg=4096)

    fig, ejes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
    ejes[0].plot(tiempo[:tramo] * 1000, limpia[:tramo], label="Referencia limpia", lw=1.4)
    ejes[0].plot(tiempo[:tramo] * 1000, entrada[:tramo], label="Contaminada", alpha=0.65)
    ejes[0].plot(tiempo[:tramo] * 1000, filtrada[:tramo], label="Filtrada FFT/IFFT", alpha=0.8)
    ejes[0].set(xlabel="Tiempo (ms)", ylabel="Amplitud normalizada", title=f"Caso {numero}: comparación temporal")
    ejes[0].grid(alpha=0.3)
    ejes[0].legend(loc="upper right")

    for curva, etiqueta in ((psd_limpia, "Referencia limpia"), (psd_entrada, "Contaminada"), (psd_filtrada, "Filtrada FFT/IFFT")):
        ejes[1].plot(frecuencias, 10 * np.log10(np.maximum(curva, 1e-16)), label=etiqueta)
    ejes[1].set(xlim=(0, 3000), xlabel="Frecuencia (Hz)", ylabel="Densidad espectral (dB/Hz)", title="Densidad espectral de potencia (Welch)")
    ejes[1].grid(alpha=0.3)
    ejes[1].legend()
    fig.savefig(SALIDA / f"caso_{numero}_comparacion_espectral.png", dpi=170)
    plt.close(fig)

    if numero == 1:
        fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
        ax.plot(frecuencias, 10 * np.log10(np.maximum(psd_entrada, 1e-16)), label="Contaminada")
        ax.plot(frecuencias, 10 * np.log10(np.maximum(psd_filtrada, 1e-16)), label="Filtrada")
        for pico in detalles["frecuencias_picos_hz"]:
            if pico <= 3000:
                ax.axvline(pico, linestyle="--", alpha=0.6, label=f"Pico estimado: {pico:.1f} Hz")
        ax.set(xlim=(1450, 2200), xlabel="Frecuencia (Hz)", ylabel="PSD (dB/Hz)", title="Detección adaptativa de interferencia: caso 1")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.savefig(SALIDA / "caso_1_interferencia_detectada.png", dpi=170)
        plt.close(fig)


def dibujar_compromiso(referencia, entrada, fs, caso):
    filas = []
    for reduccion in REDUCCIONES:
        salida = filtrar_espectral(entrada, fs=fs, N=TAM_BLOQUE, reduccion=reduccion)
        info = metricas(referencia, entrada, salida)
        filas.append({"caso": caso, "reduccion": reduccion, **info})

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    x = [fila["reduccion"] for fila in filas]
    axes[0].plot(x, [fila["snr_salida_db"] for fila in filas], "o-")
    axes[0].axhline(filas[0]["snr_entrada_db"], linestyle="--", label="SNR de entrada")
    axes[0].set(xlabel="Parámetro de reducción", ylabel="SNR de salida (dB)", title=f"Caso {caso}: intensidad y SNR")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].plot(x, [fila["mse_salida"] for fila in filas], "o-")
    axes[1].set(xlabel="Parámetro de reducción", ylabel="MSE frente a referencia", title="Compromiso de distorsión (MSE)")
    axes[1].grid(alpha=0.3)
    fig.savefig(SALIDA / f"caso_{caso}_compromiso_reduccion.png", dpi=170)
    plt.close(fig)
    return filas


def guardar_csv(ruta, filas):
    if not filas:
        return
    with ruta.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(filas[0].keys()))
        escritor.writeheader()
        escritor.writerows(filas)


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    resultados = []
    configuraciones = []
    compromiso = []

    for caso in (1, 2, 3):
        carpeta = RAIZ / "pruebas" / f"caso_{caso}"
        fs_limpia, limpia = leer_wav(carpeta / "01_referencia_limpia.wav")
        fs_entrada, entrada = leer_wav(carpeta / "03_entrada_contaminada.wav")
        if fs_limpia != fs_entrada or fs_limpia != FS_ESPERADA:
            raise ValueError(f"Frecuencias de muestreo incompatibles en caso {caso}")
        if len(limpia) != len(entrada):
            raise ValueError(f"Longitudes incompatibles en caso {caso}")

        salida, detalles = filtrar_espectral(entrada, fs=fs_limpia, N=TAM_BLOQUE, reduccion=REDUCCION_BASE, detalle=True)
        if len(salida) != len(entrada) or not np.all(np.isfinite(salida)):
            raise AssertionError(f"Salida inválida en caso {caso}")
        resultado = {"caso": caso, "fs_hz": fs_limpia, "muestras": len(salida), **metricas(limpia, entrada, salida)}
        resultados.append(resultado)
        configuraciones.append({"caso": caso, "reduccion": REDUCCION_BASE, **detalles})
        dibujar_caso(caso, fs_limpia, limpia, entrada, salida, detalles)
        compromiso.extend(dibujar_compromiso(limpia, entrada, fs_limpia, caso))
        print(f"Caso {caso}: SNR entrada {resultado['snr_entrada_db']:.2f} dB; salida {resultado['snr_salida_db']:.2f} dB; mejora {resultado['mejora_snr_db']:+.2f} dB; picos {detalles['frecuencias_picos_hz']}")

    guardar_csv(SALIDA / "metricas_espectrales.csv", resultados)
    guardar_csv(SALIDA / "compromiso_reduccion.csv", compromiso)
    (SALIDA / "parametros_espectrales.json").write_text(
        json.dumps({"fs_hz": FS_ESPERADA, "N": TAM_BLOQUE, "solapamiento": 0.5, "reduccion_base": REDUCCION_BASE, "casos": configuraciones}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Figuras y tablas guardadas en {SALIDA.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
