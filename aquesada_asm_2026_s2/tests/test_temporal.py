"""Pruebas del apartado 2.3. Ejecutar desde la raíz: python -m unittest discover -s tests -p test_temporal.py -v"""
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.filtros_temporales import (
    disenar_filtro, ecuacion_diferencias, validar, propiedades, cuantizacion
)


class TestFiltroTemporal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sos = disenar_filtro()

    def test_estabilidad_y_coeficientes(self):
        p = propiedades(self.sos)
        self.assertTrue(p['estable'])
        self.assertLess(p['radio_polo_max'], 1.0)
        self.assertEqual(self.sos.shape, (6, 6))

    def test_ecuacion_de_diferencias_contra_scipy(self):
        x = np.random.default_rng(2026).standard_normal(2048)
        error, coincide = validar(x, self.sos, tolerancia=1e-10)
        self.assertTrue(coincide, f'Diferencia máxima: {error:.3e}')
        print(f'   Error máximo contra SciPy: {error:.3e}')

    def test_aplicacion_senal_del_repositorio(self):
        ruta = ROOT / 'pruebas' / 'caso_1' / '03_entrada_contaminada.wav'
        self.assertTrue(ruta.exists(), f'Falta el archivo: {ruta}')
        fs, x = wavfile.read(ruta)
        self.assertEqual(fs, 16000)
        if np.issubdtype(x.dtype, np.integer):
            x = x.astype(np.float64) / max(abs(np.iinfo(x.dtype).min), np.iinfo(x.dtype).max)
        else:
            x = x.astype(np.float64)
        if x.ndim == 2:
            x = x.mean(axis=1)
        y = ecuacion_diferencias(x[:2048], self.sos)
        self.assertEqual(x[:2048].shape, y.shape)
        self.assertTrue(np.isfinite(y).all())
        self.assertGreater(np.max(np.abs(y)), 0)
        print(f'   WAV leído: {fs} Hz; muestras procesadas: {len(y)}')

    def test_estabilidad_tras_cuantizacion_float32(self):
        p = propiedades(cuantizacion(self.sos, np.float32))
        self.assertTrue(p['estable'])
        print(f"   Radio máximo con float32: {p['radio_polo_max']:.9f}")


if __name__ == '__main__':
    unittest.main()
