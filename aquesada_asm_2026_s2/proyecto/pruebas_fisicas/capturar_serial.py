
import argparse
import csv
import re
from datetime import datetime
from pathlib import Path

import serial


BASE = Path(__file__).resolve().parent
CARPETA_CAPTURAS = BASE / "capturas"

PATRON_MUESTRA = re.compile(
    r"^(\d+),([-+]?\d+(?:\.\d+)?),"
    r"([-+]?\d+(?:\.\d+)?),"
    r"([-+]?\d+(?:\.\d+)?)$"
)

PATRON_INICIO = re.compile(
    r"^DIAG_BEGIN,bloque=(\d+),muestras=(\d+)$"
)

def capturar(puerto, velocidad, timeout_segundos=120):
    CARPETA_CAPTURAS.mkdir(parents=True, exist_ok=True)

    print(f"Conectando con {puerto} a {velocidad} baudios...")

    with serial.Serial(
        port=puerto,
        baudrate=velocidad,
        timeout=0.5,
        write_timeout=2,
    ) as esp32:

        # El puerto puede reiniciar el ESP32 al abrirse.
        # No enviamos D inmediatamente: primero esperamos
        # a que termine de arrancar y reciba bloques.
        import time
        time.sleep(3)

        esp32.reset_input_buffer()

        print("Solicitando captura D...")
        esp32.write(b"D\n")
        esp32.flush()

        inicio = time.monotonic()
        bloque = None
        cantidad_esperada = None
        muestras = {}
        final_recibido = False

        while time.monotonic() - inicio < timeout_segundos:
            texto = esp32.readline().decode(
                "utf-8", errors="replace"
            ).strip()

            if not texto:
                continue

            coincidencia = PATRON_INICIO.fullmatch(texto)

            if coincidencia:
                bloque = int(coincidencia.group(1))
                cantidad_esperada = int(coincidencia.group(2))
                muestras.clear()
                print(
                    f"Capturando bloque {bloque}, "
                    f"{cantidad_esperada} muestras..."
                )
                continue

            if bloque is None:
                continue

            if texto == f"DIAG_END,bloque={bloque}":
                final_recibido = True
                break

            coincidencia = PATRON_MUESTRA.fullmatch(texto)

            if coincidencia:
                indice = int(coincidencia.group(1))

                if 0 <= indice < cantidad_esperada:
                    muestras[indice] = [
                        indice,
                        float(coincidencia.group(2)),
                        float(coincidencia.group(3)),
                        float(coincidencia.group(4)),
                    ]

        if bloque is None:
            raise RuntimeError(
                "No se recibió DIAG_BEGIN. "
                "Verificá ambos nodos y el puerto."
            )

        if not final_recibido:
            raise RuntimeError(
                "La captura no terminó correctamente."
            )

        indices_esperados = set(range(cantidad_esperada))
        indices_recibidos = set(muestras.keys())

        if indices_recibidos != indices_esperados:
            faltantes = sorted(
                indices_esperados - indices_recibidos
            )
            raise RuntimeError(
                f"Captura incompleta. "
                f"Muestras faltantes: {faltantes}"
            )

        fecha = datetime.now().strftime("%Y%m%d_%H%M%S")
        ruta = CARPETA_CAPTURAS / (
            f"diagnostico_bloque_{bloque}_{fecha}.csv"
        )

        with ruta.open(
            "w", newline="", encoding="utf-8"
        ) as archivo:
            escritor = csv.writer(archivo)
            escritor.writerow([
                "indice", "entrada", "iir", "espectral"
            ])

            for indice in sorted(muestras):
                escritor.writerow(muestras[indice])

        print("Captura completada correctamente.")
        print(f"Muestras guardadas: {len(muestras)}")
        print(f"Archivo: {ruta}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Captura de diagnóstico del ESP32-S3"
    )
    parser.add_argument(
        "--port", default="COM4",
        help="Puerto serial del ESP32-S3"
    )
    parser.add_argument(
        "--baud", type=int, default=115200,
        help="Velocidad del monitor serial"
    )

    args = parser.parse_args()

    try:
        capturar(args.port, args.baud)
    except (serial.SerialException, RuntimeError) as error:
        print(f"ERROR: {error}")
        raise SystemExit(1)
