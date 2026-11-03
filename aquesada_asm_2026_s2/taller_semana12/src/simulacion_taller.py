import os
import wave
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#parámetros del sistema
FS = 8000                  #frecuencia de muestreo [Hz]
N = 512                    #muestras por símbolo (potencia de 2)
DF = FS / N                #resolución de la FFT [Hz]
FRECS = np.array([1000.0, 1500.0, 2000.0, 2500.0])   #tono de cada símbolo (00, 01, 10, 11)
BINS = np.round(FRECS / DF).astype(int)              #bin de la FFT de cada tono
SEMILLA = 2026

#carpetas de salida son relativas a la carpeta del taller
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTADOS = os.path.join(RAIZ, "resultados")
FIGURAS = os.path.join(RESULTADOS, "figuras")
os.makedirs(FIGURAS, exist_ok=True)

rng = np.random.default_rng(SEMILLA)


#vector de tiempo de un bloque de n muestras
def tiempo(n=N):
    return np.arange(n) / FS


#espectro de magnitud en dB un seno de amplitud 1 dentro de un bin da 0 dB
def espectro_db(x, ventana=None):
    w = np.ones(len(x)) if ventana is None else ventana
    X = np.fft.rfft(x * w)
    mag = 2 * np.abs(X) / np.sum(w)
    f = np.fft.rfftfreq(len(x), 1 / FS)
    return f, 20 * np.log10(mag + 1e-12)


#emisor convierte una lista de símbolos 0 a 3 en un tono por bloque
def modular(simbolos):
    return np.concatenate([np.sin(2 * np.pi * FRECS[s] * tiempo()) for s in simbolos])


#receptor FFT por bloque y decisión por el bin de mayor magnitud
def demodular(x):
    bloques = x[: len(x) // N * N].reshape(-1, N)
    mags = np.abs(np.fft.rfft(bloques, axis=1))[:, BINS]
    return np.argmax(mags, axis=1)


#canal ruido blanco gaussiano para una SNR dada en dB
def ruido_para_snr(senal, snr_db):
    p_ruido = np.mean(senal ** 2) / 10 ** (snr_db / 10)
    return rng.normal(0, np.sqrt(p_ruido), senal.shape)


#tasa de error de símbolo por Monte Carlo con error de frecuencia opcional
def simular_ser(snr_db, n_sim=3000, offset_hz=0.0):
    sim = rng.integers(0, 4, n_sim)
    f = FRECS[sim].reshape(-1, 1) + offset_hz
    x = np.sin(2 * np.pi * f * tiempo().reshape(1, -1))
    x = x + ruido_para_snr(x, snr_db)
    mags = np.abs(np.fft.rfft(x, axis=1))[:, BINS]
    return np.mean(np.argmax(mags, axis=1) != sim)


#guarda la figura actual en resultados/figuras
def guardar(nombre):
    ruta = os.path.join(FIGURAS, nombre)
    plt.tight_layout()
    plt.savefig(ruta, dpi=200)
    plt.close()
    print("  guardado:", ruta)


#guarda la señal como WAV de 16 bits para poder escucharla
def guardar_wav(x, nombre):
    datos = (x / np.max(np.abs(x)) * 0.8 * 32767).astype(np.int16)
    ruta = os.path.join(RESULTADOS, nombre)
    with wave.open(ruta, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(FS)
        w.writeframes(datos.tobytes())
    print("  guardado:", ruta)


#imprime y guarda la tabla del plan de frecuencias
def tabla_frecuencias():
    print(f"fs = {FS} Hz | N = {N} | resolución = {DF:.3f} Hz | símbolo = {N / FS * 1e3:.1f} ms")
    print("símbolo  bits  frecuencia[Hz]  bin")
    lineas = ["simbolo,bits,frecuencia_hz,bin"]
    for s in range(4):
        print(f"   {s}      {s:02b}     {FRECS[s]:8.1f}      {BINS[s]}")
        lineas.append(f"{s},{s:02b},{FRECS[s]},{BINS[s]}")
    with open(os.path.join(RESULTADOS, "plan_frecuencias.csv"), "w") as f:
        f.write("\n".join(lineas) + "\n")


#figura 1 fuga espectral tono en bin vs fuera de bin, rectangular vs Hann
def fig_fuga_espectral():
    f_on, f_off = FRECS[0], FRECS[0] + DF / 2
    hann = np.hanning(N)
    plt.figure(figsize=(7, 3.8))
    for f0, w, et in [(f_on, None, "En bin, rectangular"),
                      (f_off, None, "Fuera de bin, rectangular"),
                      (f_off, hann, "Fuera de bin, Hann")]:
        f, m = espectro_db(np.sin(2 * np.pi * f0 * tiempo()), w)
        plt.plot(f, m, label=et)
    plt.xlim(700, 1300)
    plt.ylim(-120, 5)
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Magnitud [dB]")
    plt.title("Fuga espectral según la alineación con el bin")
    plt.grid(alpha=.3)
    plt.legend()
    guardar("fig1_fuga_espectral.png")


#figura 2 espectro de las 4 frecuencias del plan junto con Nyquist
def fig_plan_frecuencias():
    plt.figure(figsize=(7.5, 3.8))
    for s in range(4):
        f, m = espectro_db(np.sin(2 * np.pi * FRECS[s] * tiempo()))
        plt.plot(f, m, label=f"{s:02b}: {FRECS[s]:.0f} Hz")
    plt.axvline(FS / 2, color="red", ls=":", lw=1)
    plt.text(FS / 2 - 80, -100, "Nyquist", rotation=90, color="red", fontsize=8)
    plt.ylim(-110, 5)
    plt.xlim(0, FS / 2)
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Magnitud [dB]")
    plt.title(f"Plan de frecuencias 4-FSK (fs={FS} Hz, N={N}, Δf={DF:.3f} Hz)")
    plt.grid(alpha=.3)
    plt.legend(ncol=2, loc="upper right")
    guardar("fig2_plan_frecuencias.png")


#figura 3 mensaje transmitido, recibido con ruido y espectrograma
def fig_modulacion_demo():
    simbolos = [0, 1, 2, 3, 3, 2, 1, 0]
    tx = modular(simbolos)
    rx = tx + ruido_para_snr(tx, -3)
    dec = demodular(rx)
    print("enviado    :", simbolos)
    print("decodificado:", dec.tolist(), "(SNR = -3 dB)")
    guardar_wav(tx, "mensaje_tx.wav")
    guardar_wav(rx, "mensaje_rx_snr-3dB.wav")
    t = np.arange(len(tx)) / FS * 1e3
    fig, ax = plt.subplots(3, 1, figsize=(7.5, 7))
    ax[0].plot(t, tx, lw=.6)
    ax[0].set_title("Señal transmitida (4-FSK)")
    ax[1].plot(t, rx, lw=.6, color="C1")
    ax[1].set_title("Señal recibida, SNR = -3 dB")
    for a in ax[:2]:
        a.set_xlabel("Tiempo [ms]")
        a.set_ylabel("Amplitud")
        a.grid(alpha=.3)
        for i in range(len(simbolos) + 1):
            a.axvline(i * N / FS * 1e3, color="k", lw=.4, ls=":")
    ax[2].specgram(rx, NFFT=N, Fs=FS, noverlap=0, cmap="viridis")
    ax[2].set_title("Espectrograma de la señal recibida")
    ax[2].set_xlabel("Tiempo [s]")
    ax[2].set_ylabel("Frecuencia [Hz]")
    guardar("fig3_modulacion_demo.png")


#figura 4 error de símbolo vs SNR y vs error de frecuencia del emisor
def fig_desempeno():
    snrs = np.arange(-24, 1, 2)
    ser = [simular_ser(s) for s in snrs]
    offs = np.linspace(0, DF, 17)
    ser_off = [simular_ser(-14, offset_hz=o) for o in offs]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].semilogy(snrs, np.maximum(ser, 1e-4), "o-")
    ax[0].set_xlabel("SNR [dB]")
    ax[0].set_ylabel("Tasa de error de símbolo")
    ax[0].set_title("Desempeño del receptor con FFT")
    ax[0].grid(alpha=.3, which="both")
    ax[1].plot(offs, ser_off, "o-", color="C1")
    ax[1].axvline(DF / 2, color="red", ls=":", label="Δf/2")
    ax[1].set_xlabel("Error de frecuencia del emisor [Hz]")
    ax[1].set_ylabel("Tasa de error de símbolo")
    ax[1].set_title("Sensibilidad al error de frecuencia (SNR = -14 dB)")
    ax[1].grid(alpha=.3)
    ax[1].legend()
    guardar("fig4_desempeno.png")


if __name__ == "__main__":
    print("== Plan de frecuencias ==")
    tabla_frecuencias()
    print("\n== Resultados ==")
    fig_fuga_espectral()
    fig_plan_frecuencias()
    fig_modulacion_demo()
    fig_desempeno()