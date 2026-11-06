# Evidencia de pruebas físicas P1-P5

## Descripción

Mediciones realizadas mediante nodos ESP32 y ESP32-S3,
con procesamiento temporal IIR y espectral FFT/IFFT.

Se utiliza una frecuencia de muestreo nominal de 16000 Hz
y bloques de 1024 muestras (64 ms).

Los estímulos provienen de las funciones originales de
`proyecto/simulacion/src/senales.py`.

Los archivos WAV se generan con
`proyecto/pruebas_fisicas/exportar_senales_originales.py`,
aplicando una ganancia común a señal limpia, ruido y mezcla
para evitar recorte en la exportación PCM.

## Contenido

Cada carpeta P1-P5 incluye:

- Tres capturas físicas seleccionadas: entrada, IIR y FFT/IFFT.
- CSV de métricas individuales y resumen estadístico.
- Gráficas temporales y espectrales.
- En P4, gráficas de evolución de la componente de 900 Hz.

## Procedimiento

P1, P2, P3 y P5: tres capturas de bloques durante
la reproducción del estímulo contaminado.

P4: captura coordinada desde Python con reproducción WAV
y solicitud serial D aproximadamente a los 2.12 segundos.
Se conservaron las ventanas que permitieron observar
distintas fases del transitorio de 900 Hz.

Origen de las capturas definitivas de P4:

- R1: P4_candidato_R4.csv (crecimiento y decaimiento).
- R2: P4_candidato_R5.csv (crecimiento y decaimiento).
- R3: P4_candidato_R7.csv (principalmente decaimiento).

Los intentos de calibración descartados permanecen en los
datos locales y no integran el conjunto de evidencia publicado.

## Interpretación y limitaciones

Las métricas publicadas comparan señales digitales medidas
en el nodo de procesamiento.

La reducción de energía por banda no es una medición
calibrada de atenuación acústica.

La energía conservada en una banda puede contener tanto
señal útil como interferencia.

El delta SNR espectral de P2 es una estimación basada en
bandas de frecuencia, no un delta SNR verdadero calculado
contra una referencia limpia alineada.

No se dispone todavía de referencias físicas limpias
temporalmente alineadas para calcular MSE o delta SNR
verdaderos.

Las tres capturas de P4 corresponden a ventanas seleccionadas
tras la calibración de tiempo y no son fases equivalentes
del transitorio. No permiten estimar por sí solas una
latencia absoluta del sistema.

Las repeticiones físicas corresponden a nuevas adquisiciones
de los estímulos base; no deben confundirse con las semillas
de repetición del notebook de simulación.
