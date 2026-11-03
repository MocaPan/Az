# aquesada_asm_2026_s2

Proyecto desarrollado para el curso **CE1110 Análisis de Señales Mixtas** del Instituto Tecnológico de Costa Rica.

El objetivo es diseñar y evaluar un sistema distribuido de reducción adaptativa de ruido en tiempo real. El prototipo incluye generación reproducible de señales, análisis mediante FFT, filtrado temporal IIR, reducción espectral mediante FFT/IFFT y comparación de ambas estrategias.

## Estructura del repositorio

    .
    ├── metadatos/
    │   └── configuraciones.json
    ├── pruebas/
    │   ├── caso_1/
    │   ├── caso_2/
    │   └── caso_3/
    ├── resultados/
    │   ├── espectral/
    │   ├── figuras/
    │   ├── temporal/
    │   └── metricas.csv
    ├── src/
    │   ├── analisis_fft.py
    │   ├── filtrado_espectral.py
    │   ├── filtros_temporales.py
    │   └── metricas.py
    ├── tests/
    │   ├── test_comparacion.py
    │   ├── test_espectral.py
    │   └── test_temporal.py
    ├── comparacion_metodos.py
    ├── evidencia_espectral.py
    ├── evidencia_temporal.py
    └── README.md

## Casos de prueba

Se utilizan tres señales reproducibles con los siguientes parámetros comunes:

- Frecuencia de muestreo: `16 kHz`
- Tamaño de bloque: `1024` muestras
- Duración: `6 s`
- Frecuencias útiles: `500 Hz`, `900 Hz` y `1300 Hz`
- SNR objetivo: `5 dB`

Los tres casos evaluados son:

1. Interferencia sinusoidal en `1800 Hz`.
2. Ruido blanco.
3. Ruido de banda limitada entre `800 Hz` y `1500 Hz`.

Cada caso incluye la señal limpia, el ruido, la señal contaminada y sus metadatos correspondientes.

## Métodos implementados

### Filtrado temporal

Se utiliza un filtro IIR Butterworth pasa bajas de orden 12 con frecuencia de corte de `1500 Hz`, implementado mediante secciones de segundo orden.

La ecuación de diferencias se aplica manualmente muestra a muestra y se compara contra la implementación de referencia `scipy.signal.sosfilt`.

### Filtrado espectral

Se utiliza procesamiento por bloques mediante la siguiente cadena:

    Ventana Hann
        ↓
    FFT
        ↓
    Estimación de ruido
        ↓
    Modificación espectral
        ↓
    IFFT
        ↓
    Reconstrucción

El procesamiento utiliza bloques de `1024` muestras y `50 %` de solapamiento.

## Comparación de métodos

El script principal de comparación es:

    python comparacion_metodos.py

Este genera el archivo:

    resultados/metricas.csv

Las métricas calculadas incluyen:

- MSE
- SNR
- ΔSNR
- Energía útil conservada
- Atenuación de interferencia
- Tiempo de ejecución por bloque
- Memoria aproximada

## Pruebas automáticas

Para ejecutar todas las pruebas:

    python -m unittest discover -s tests -v

También pueden ejecutarse individualmente:

    python -m unittest discover -s tests -p "test_temporal.py" -v

    python -m unittest discover -s tests -p "test_espectral.py" -v

    python -m unittest discover -s tests -p "test_comparacion.py" -v

Las pruebas verifican estabilidad, equivalencia con funciones de referencia, validez de las salidas y comparación automática entre los métodos temporal y espectral.

## Evidencias reproducibles

Las figuras y resultados pueden regenerarse mediante:

    python evidencia_temporal.py

    python evidencia_espectral.py

    python comparacion_metodos.py

Los resultados generados se almacenan en la carpeta `resultados/`.

## Implementación embebida propuesta

La implementación final se plantea sobre tres nodos basados en **ESP32-S3**:

    Nodo 1: adquisición
        ↓
    Nodo 2: procesamiento DSP
        ↓
    Nodo 3: salida y monitoreo

De acuerdo con los resultados obtenidos, la estrategia principal seleccionada para el prototipo embebido es la reducción espectral mediante FFT/IFFT.

Los parámetros propuestos son:

- Frecuencia de muestreo: `16 kHz`
- Tamaño de bloque: `1024`
- Formato numérico: `float32`

## Repositorio

Repositorio del proyecto:

https://github.com/aman-qp/aquesada_asm_2026_s2

## Tag de entrega

La versión correspondiente a la entrega de la Tarea 1 se identifica mediante el tag:

    tarea1_prototipo

## Integrantes

- Amanda Jimena Quesada Porras
- John Mauricio Rojas Machado