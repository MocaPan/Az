from pathlib import Path
import argparse
import csv
import re
import time
import winsound
import serial
import numpy as np

BASE = Path(__file__).resolve().parent
WAV = BASE / "senales_originales/P4/03_entrada_contaminada.wav"
CAPTURAS = BASE / "capturas"

PATRON_INICIO = re.compile(
    r"^DIAG_BEGIN,bloque=(\d+),muestras=(\d+)$"
)

PATRON_MUESTRA = re.compile(
    r"^(\d+),([-+]?\d+(?:\.\d+)?),"
    r"([-+]?\d+(?:\.\d+)?),"
    r"([-+]?\d+(?:\.\d+)?)$"
)


def capturar(puerto, demora, repeticion):
    if not WAV.is_file():
        raise RuntimeError(f"No existe el WAV: {WAV}")

    CAPTURAS.mkdir(parents=True, exist_ok=True)

    destino = CAPTURAS / f"P4_candidato_R{repeticion}.csv"

    if destino.exists():
        raise RuntimeError(
            f"{destino.name} ya existe. No se sobrescribira."
        )

    print("=== CAPTURA CONTROLADA P4 ===")
    print("Puerto:", puerto)
    print("Audio:", WAV)
    print(f"Solicitud D a los {demora:.3f} s aprox.")
    print("Destino:", destino)

    with serial.Serial(
        puerto,
        115200,
        timeout=0.5,
        write_timeout=2
    ) as esp:

        print("\nEsperando inicio del ESP32-S3...")
        time.sleep(3)
        esp.reset_input_buffer()

        print("Iniciando reproduccion del WAV...")

        inicio = time.perf_counter()

        winsound.PlaySound(
            str(WAV),
            winsound.SND_FILENAME
            | winsound.SND_ASYNC
            | winsound.SND_NODEFAULT
        )

        try:
            restante = demora - (time.perf_counter() - inicio)

            if restante > 0:
                time.sleep(restante)

            instante_d = time.perf_counter() - inicio

            esp.write(b"D\n")
            esp.flush()

            print(
                f"Comando D enviado a los "
                f"{instante_d:.3f} s del reloj de Python"
            )

            muestras = {}
            bloque = None
            esperado = None
            fin = False
            ultima_linea = ''

            limite = time.monotonic() + 120

            while time.monotonic() < limite:
                linea = esp.readline().decode(
                    "utf-8", errors="replace"
                ).strip()

                if not linea:
                    continue

                ultima_linea = linea

                inicio_diag = PATRON_INICIO.fullmatch(linea)

                if inicio_diag:
                    bloque = int(inicio_diag.group(1))
                    esperado = int(inicio_diag.group(2))

                    if esperado != 1024:
                        raise RuntimeError(
                            f"Diagnostico de {esperado} muestras"
                        )

                    muestras.clear()
                    print("Bloque recibido:", bloque)
                    continue

                if bloque is None:
                    continue

                if linea == f"DIAG_END,bloque={bloque}":
                    fin = True
                    break

                coincidencia = PATRON_MUESTRA.fullmatch(linea)

                if coincidencia:
                    indice = int(coincidencia.group(1))

                    if indice in muestras:
                        raise RuntimeError("Indice duplicado")

                    if indice % 256 == 0:
                        print(f'Recibiendo muestra {indice}/1024', flush=True)

                    muestras[indice] = [
                        indice,
                        float(coincidencia.group(2)),
                        float(coincidencia.group(3)),
                        float(coincidencia.group(4))
                    ]

            if not fin:
                print(f'Muestras recibidas: {len(muestras)}/1024')
                print(f'Ultima linea recibida: {ultima_linea!r}')
                raise RuntimeError(
                    "No se recibio DIAG_END correctamente"
                )

            if sorted(muestras) != list(range(1024)):
                raise RuntimeError(
                    "La captura no tiene 1024 indices completos"
                )

            datos = np.array(
                [muestras[i] for i in range(1024)]
            )

            if not np.all(np.isfinite(datos)):
                raise RuntimeError("Hay datos no finitos")

            with destino.open(
                "x", newline="", encoding="utf-8"
            ) as archivo:
                escritor = csv.writer(archivo)
                escritor.writerow([
                    "indice", "entrada", "iir", "espectral"
                ])
                escritor.writerows(datos)

            x = datos[:, 1]
            fs = 16000

            ventana = np.hanning(len(x))
            frecuencias = np.fft.rfftfreq(
                len(x), 1 / fs
            )
            magnitudes = np.abs(
                np.fft.rfft(
                    (x - x.mean()) * ventana
                )
            )

            k900 = np.argmin(
                np.abs(frecuencias - 900)
            )

            vecindad = (
                (frecuencias >= 600)
                & (frecuencias <= 1200)
                & (np.abs(frecuencias - 900) >= 80)
            )

            fondo = np.median(magnitudes[vecindad])
            relacion = magnitudes[k900] / max(fondo, 1e-12)

            print("\n=== VALIDACION INICIAL ===")
            print("Muestras:", len(x))
            print(
                "RMS entrada:",
                f"{np.sqrt(np.mean(x**2)):.2f}"
            )
            print(
                "Pico absoluto:",
                f"{np.max(np.abs(x)):.2f}"
            )
            print(
                "Magnitud cerca de 900 Hz:",
                f"{magnitudes[k900]:.2f}"
            )
            print(
                "Relacion pico 900 Hz / fondo local:",
                f"{relacion:.2f}"
            )
            print("Archivo guardado:", destino)
            print("\nCAPTURA CANDIDATA COMPLETADA")

        finally:
            winsound.PlaySound(None, 0)
            print("Reproduccion detenida.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM4")
    parser.add_argument("--demora", type=float, default=1.93)
    parser.add_argument("--repeticion", type=int, default=1)

    args = parser.parse_args()

    if not 0 <= args.demora <= 6:
        parser.error("La demora debe estar entre 0 y 6 segundos")

    try:
        capturar(
            args.port,
            args.demora,
            args.repeticion
        )
    except (
        serial.SerialException,
        OSError,
        RuntimeError
    ) as error:
        parser.exit(1, f"ERROR: {error}\n")
