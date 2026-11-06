from pathlib import Path
import importlib.util
import wave
import json
import hashlib

import numpy as np

BASE = Path("proyecto/pruebas_fisicas")
ORIGEN = Path("proyecto/simulacion/src/senales.py")
DESTINO = BASE / "senales_originales"

# Cargar el codigo original sin modificarlo.
spec = importlib.util.spec_from_file_location(
    "senales_simulacion_original", ORIGEN
)

if spec is None or spec.loader is None:
    raise RuntimeError("No se pudo cargar senales.py")

modulo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(modulo)

FS = modulo.FS
DURACION = modulo.DURACION

GENERADORES = {
    "P1": modulo.generar_p1,
    "P2": modulo.generar_p2,
    "P3": modulo.generar_p3,
    "P4": modulo.generar_p4,
    "P5": modulo.generar_p5,
}

# Evitar sobrescribir señales previamente generadas.
for nombre in GENERADORES:
    carpeta = DESTINO / nombre

    if carpeta.exists() and any(carpeta.iterdir()):
        raise SystemExit(
            f"ERROR: {carpeta} ya contiene archivos. "
            "No se sobrescribira."
        )


def convertir_pcm(datos):
    datos = np.asarray(datos, dtype=np.float64)

    if not np.all(np.isfinite(datos)):
        raise ValueError("Hay valores no finitos")

    if np.max(np.abs(datos)) > 1.000001:
        raise ValueError("La señal supera el rango PCM")

    pcm = np.round(
        np.clip(datos, -1.0, 1.0) * 32767
    ).astype("<i2")

    return pcm


def guardar_wav(ruta, datos):
    pcm = convertir_pcm(datos)

    with wave.open(str(ruta), "wb") as archivo:
        archivo.setnchannels(1)
        archivo.setsampwidth(2)
        archivo.setframerate(FS)
        archivo.writeframes(pcm.tobytes())

    return hashlib.sha256(pcm.tobytes()).hexdigest()


print("\n=== EXPORTACION DE SEÑALES ORIGINALES ===")

for nombre, generador in GENERADORES.items():

    # Utilizar exactamente la señal base del generador.
    limpia, ruido, contaminada = generador()

    limpia = np.asarray(limpia, dtype=np.float64)
    ruido = np.asarray(ruido, dtype=np.float64)
    contaminada = np.asarray(contaminada, dtype=np.float64)

    if not np.allclose(
        limpia + ruido,
        contaminada,
        rtol=1e-10,
        atol=1e-10
    ):
        raise ValueError(
            f"{nombre}: la mezcla no coincide con limpia + ruido"
        )

    if not (
        len(limpia) == len(ruido) == len(contaminada)
        == int(FS * DURACION)
    ):
        raise ValueError(
            f"{nombre}: tamaño de señal incorrecto"
        )

    # Ganancia comun para evitar recorte al exportar WAV.
    pico = max(
        np.max(np.abs(limpia)),
        np.max(np.abs(ruido)),
        np.max(np.abs(contaminada))
    )

    ganancia = 0.90 / pico

    limpia_wav = limpia * ganancia
    ruido_wav = ruido * ganancia
    contaminada_wav = contaminada * ganancia

    carpeta = DESTINO / nombre
    carpeta.mkdir(parents=True, exist_ok=True)

    archivos = {
        "01_referencia_limpia.wav": limpia_wav,
        "02_ruido.wav": ruido_wav,
        "03_entrada_contaminada.wav": contaminada_wav,
    }

    hashes = {}

    for archivo, datos in archivos.items():
        hashes[archivo] = guardar_wav(
            carpeta / archivo, datos
        )

    # Referencias sin cuantizacion, con la misma ganancia.
    np.savez(
        carpeta / "referencias_digitales.npz",
        limpia=limpia_wav,
        ruido=ruido_wav,
        contaminada=contaminada_wav,
        fs=FS
    )

    potencia_limpia = np.mean(limpia**2)
    potencia_ruido = np.mean(ruido**2)

    snr = 10 * np.log10(
        potencia_limpia / potencia_ruido
    )

    parametros = {
        "prueba": nombre,
        "origen": "proyecto/simulacion/src/senales.py",
        "generador": generador.__name__,
        "semilla": (
            None if nombre == "P1"
            else modulo.SEMILLAS[nombre]
        ),
        "frecuencia_muestreo_hz": FS,
        "duracion_segundos": DURACION,
        "muestras": len(limpia),
        "snr_digital_db": float(snr),
        "ganancia_exportacion": float(ganancia),
        "formato_wav": "PCM mono 16 bits",
        "hashes_sha256_pcm": hashes
    }

    (carpeta / "parametros.json").write_text(
        json.dumps(
            parametros,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print(f"\n{nombre}")
    print(f"  Muestras: {len(limpia)}")
    print(f"  Duracion: {DURACION} segundos")
    print(f"  SNR digital: {snr:.2f} dB")
    print(f"  Ganancia exportacion: {ganancia:.6f}")
    print(f"  Carpeta: {carpeta}")

print("\n=== EXPORTACION COMPLETADA ===")
