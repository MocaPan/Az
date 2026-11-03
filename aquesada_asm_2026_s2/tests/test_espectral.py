"""Pruebas del apartado 2.4: python -m unittest discover -s tests -p test_espectral.py -v."""

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.filtrado_espectral import filtrar_espectral


class TestFiltradoEspectral(unittest.TestCase):
    def test_parametros_invalidos(self):
        x = np.zeros(2048)
        for n in (0, 31, 1000):
            with self.subTest(N=n), self.assertRaises(ValueError):
                filtrar_espectral(x, N=n)
        with self.assertRaises(ValueError):
            filtrar_espectral(x, reduccion=-0.1)

    def test_reconstruccion_sin_atenuacion(self):
        """Con reduccion=0 la ISTFT debe recuperar la señal de entrada."""
        rng = np.random.default_rng(2026)
        x = rng.normal(size=8193)
        y = filtrar_espectral(x, reduccion=0.0)
        error = float(np.max(np.abs(x - y)))
        self.assertEqual(y.shape, x.shape)
        self.assertLess(error, 1e-10)
        print(f"    Error máximo de reconstrucción: {error:.3e}")

    def test_salida_real_y_finita_en_longitudes_distintas(self):
        rng = np.random.default_rng(42)
        for longitud in (1024, 4096, 5003):
            with self.subTest(longitud=longitud):
                x = rng.standard_normal(longitud)
                y, info = filtrar_espectral(x, detalle=True)
                self.assertEqual(y.shape, x.shape)
                self.assertTrue(np.isrealobj(y))
                self.assertTrue(np.isfinite(y).all())
                self.assertEqual(info["N"], 1024)
                self.assertEqual(info["salto"], 512)
                self.assertEqual(info["solapamiento"], 0.5)

    def test_parametro_de_reduccion_modifica_salida(self):
        rng = np.random.default_rng(15)
        x = rng.standard_normal(4096)
        y_sin = filtrar_espectral(x, reduccion=0.0)
        y_con = filtrar_espectral(x, reduccion=1.0)
        self.assertGreater(np.linalg.norm(y_con - y_sin), 1e-3)

    def test_deteccion_interferencia_sin_frecuencia_fija(self):
        fs = 16000
        t = np.arange(fs * 2) / fs
        x = 0.8 * np.sin(2 * np.pi * 500 * t) + 0.65 * np.sin(2 * np.pi * 1800 * t)
        _, info = filtrar_espectral(x, fs=fs, detalle=True)
        picos = info["frecuencias_picos_hz"]
        self.assertTrue(any(abs(f - 1800) <= 2 * fs / 1024 for f in picos), picos)
        print(f"    Interferencias detectadas: {picos[:8]}")

    def test_tres_casos_del_repositorio(self):
        for caso in (1, 2, 3):
            ruta = ROOT / "pruebas" / f"caso_{caso}" / "03_entrada_contaminada.wav"
            with self.subTest(caso=caso):
                self.assertTrue(ruta.is_file(), f"No existe {ruta}")
                fs, x = wavfile.read(ruta)
                self.assertEqual(fs, 16000)
                if np.issubdtype(x.dtype, np.integer):
                    x = x.astype(np.float64) / max(abs(np.iinfo(x.dtype).min), np.iinfo(x.dtype).max)
                else:
                    x = x.astype(np.float64)
                if x.ndim == 2:
                    x = x.mean(axis=1)
                y = filtrar_espectral(x, fs=fs)
                self.assertEqual(len(y), len(x))
                self.assertTrue(np.isfinite(y).all())
                print(f"    Caso {caso}: {len(y)} muestras procesadas")


if __name__ == "__main__":
    unittest.main()
