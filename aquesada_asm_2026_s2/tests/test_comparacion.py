"""Pruebas automáticas de comparación entre filtrado temporal y espectral.

Ejecutar desde la raíz del repositorio:

python -m unittest discover -s tests -p "test_comparacion.py" -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.io import wavfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from src.filtros_temporales import (
    disenar_filtro,
    ecuacion_diferencias,
)

from src.filtrado_espectral import (
    filtrar_espectral,
)

from src.metricas import (
    mse,
    snr_db,
    alinear_por_retardo,
    energia_util_conservada_pct,
    memoria_temporal_bytes,
    memoria_espectral_bytes,
)


FS = 16000
N = 1024


def leer_wav(ruta):
    """Lee un WAV y lo convierte a float64 normalizado."""
    fs, datos = wavfile.read(ruta)

    if np.issubdtype(datos.dtype, np.integer):
        info = np.iinfo(datos.dtype)

        escala = float(
            max(
                abs(int(info.min)),
                abs(int(info.max)),
            )
        )

        datos = datos.astype(np.float64) / escala

    else:
        datos = datos.astype(np.float64)

    if datos.ndim == 2:
        datos = datos.mean(axis=1)

    return fs, datos


class TestComparacionMetodos(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sos = disenar_filtro()

        cls.casos = {}

        for numero in range(1, 4):
            carpeta = (
                ROOT
                / "pruebas"
                / f"caso_{numero}"
            )

            fs_ref, referencia = leer_wav(
                carpeta
                / "01_referencia_limpia.wav"
            )

            fs_in, entrada = leer_wav(
                carpeta
                / "03_entrada_contaminada.wav"
            )

            if fs_ref != FS or fs_in != FS:
                raise ValueError(
                    f"Frecuencia de muestreo incorrecta "
                    f"en caso {numero}"
                )

            cls.casos[numero] = {
                "referencia": referencia,
                "entrada": entrada,
            }

    def test_archivos_y_longitudes(self):
        """Las señales de cada caso deben ser compatibles."""
        for numero, datos in self.casos.items():
            referencia = datos["referencia"]
            entrada = datos["entrada"]

            self.assertEqual(
                referencia.shape,
                entrada.shape,
                f"Longitud incompatible en caso {numero}",
            )

            self.assertGreater(
                len(referencia),
                0,
            )

            self.assertTrue(
                np.isfinite(referencia).all()
            )

            self.assertTrue(
                np.isfinite(entrada).all()
            )

    def test_ambos_metodos_producen_salida_valida(self):
        """Temporal y espectral deben mantener longitud y valores finitos."""
        for numero, datos in self.casos.items():
            entrada = datos["entrada"]

            temporal = ecuacion_diferencias(
                entrada,
                self.sos,
            )

            espectral = filtrar_espectral(
                entrada,
                fs=FS,
                N=N,
                reduccion=1.0,
            )

            self.assertEqual(
                entrada.shape,
                temporal.shape,
                f"Longitud temporal inválida en caso {numero}",
            )

            self.assertEqual(
                entrada.shape,
                espectral.shape,
                f"Longitud espectral inválida en caso {numero}",
            )

            self.assertTrue(
                np.isfinite(temporal).all()
            )

            self.assertTrue(
                np.isfinite(espectral).all()
            )

    def test_snr_espectral_mejora_los_tres_casos(self):
        """
        Con la configuración actual, el método espectral debe mejorar
        la SNR respecto a la entrada en los tres casos.
        """
        for numero, datos in self.casos.items():
            referencia = datos["referencia"]
            entrada = datos["entrada"]

            espectral = filtrar_espectral(
                entrada,
                fs=FS,
                N=N,
                reduccion=1.0,
            )

            snr_entrada = snr_db(
                referencia,
                entrada,
            )

            snr_salida = snr_db(
                referencia,
                espectral,
            )

            self.assertGreater(
                snr_salida,
                snr_entrada,
                (
                    f"El filtrado espectral no mejoró la SNR "
                    f"del caso {numero}: "
                    f"{snr_entrada:.2f} -> {snr_salida:.2f} dB"
                ),
            )

            print(
                f"   Caso {numero}: espectral "
                f"{snr_entrada:.2f} -> {snr_salida:.2f} dB"
            )

    def test_comparacion_temporal_vs_espectral(self):
        """
        Compara las dos estrategias utilizando la salida temporal
        alineada únicamente para evaluación offline.
        """
        for numero, datos in self.casos.items():
            referencia = datos["referencia"]
            entrada = datos["entrada"]

            temporal = ecuacion_diferencias(
                entrada,
                self.sos,
            )

            (
                ref_temporal,
                temporal_alineada,
                retardo,
            ) = alinear_por_retardo(
                referencia,
                temporal,
                max_retardo=64,
            )

            espectral = filtrar_espectral(
                entrada,
                fs=FS,
                N=N,
                reduccion=1.0,
            )

            snr_temporal = snr_db(
                ref_temporal,
                temporal_alineada,
            )

            snr_espectral = snr_db(
                referencia,
                espectral,
            )

            mse_temporal = mse(
                ref_temporal,
                temporal_alineada,
            )

            mse_espectral = mse(
                referencia,
                espectral,
            )

            # Con la configuración actual del prototipo,
            # el método espectral obtuvo mejor desempeño
            # en los tres casos medidos.
            self.assertGreater(
                snr_espectral,
                snr_temporal,
                (
                    f"Resultado inesperado en caso {numero}: "
                    f"SNR espectral={snr_espectral:.2f}, "
                    f"SNR temporal={snr_temporal:.2f}"
                ),
            )

            self.assertLess(
                mse_espectral,
                mse_temporal,
                (
                    f"Resultado inesperado en caso {numero}: "
                    f"MSE espectral={mse_espectral:.6e}, "
                    f"MSE temporal={mse_temporal:.6e}"
                ),
            )

            print(
                f"   Caso {numero}: "
                f"temporal={snr_temporal:.2f} dB, "
                f"espectral={snr_espectral:.2f} dB, "
                f"retardo={retardo} muestras"
            )

    def test_energia_util_finita_y_positiva(self):
        """La estimación de energía útil debe producir valores válidos."""
        for numero, datos in self.casos.items():
            referencia = datos["referencia"]
            entrada = datos["entrada"]

            temporal = ecuacion_diferencias(
                entrada,
                self.sos,
            )

            (
                ref_temporal,
                temporal_alineada,
                _,
            ) = alinear_por_retardo(
                referencia,
                temporal,
                max_retardo=64,
            )

            espectral = filtrar_espectral(
                entrada,
                fs=FS,
                N=N,
                reduccion=1.0,
            )

            energia_temporal = (
                energia_util_conservada_pct(
                    ref_temporal,
                    temporal_alineada,
                )
            )

            energia_espectral = (
                energia_util_conservada_pct(
                    referencia,
                    espectral,
                )
            )

            self.assertTrue(
                np.isfinite(energia_temporal)
            )

            self.assertTrue(
                np.isfinite(energia_espectral)
            )

            self.assertGreater(
                energia_temporal,
                0.0,
            )

            self.assertGreater(
                energia_espectral,
                0.0,
            )

    def test_estimacion_memoria(self):
        """El cálculo de memoria debe coincidir con la configuración usada."""
        memoria_temporal = memoria_temporal_bytes(
            numero_secciones=len(self.sos),
            N=N,
        )

        memoria_espectral = memoria_espectral_bytes(
            N=N,
        )

        self.assertEqual(
            memoria_temporal["total"],
            4288,
        )

        self.assertEqual(
            memoria_espectral["total"],
            16396,
        )

        self.assertGreater(
            memoria_espectral["total"],
            memoria_temporal["total"],
        )

        print(
            "   Memoria temporal:",
            memoria_temporal["total"],
            "bytes",
        )

        print(
            "   Memoria espectral:",
            memoria_espectral["total"],
            "bytes",
        )


if __name__ == "__main__":
    unittest.main()