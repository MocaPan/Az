"""Evidencias reproducibles del apartado 2.3. Ejecutar desde la raíz del repositorio."""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
from src.filtros_temporales import disenar_filtro, ecuacion_diferencias, validar, propiedades, cuantizacion

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'resultados' / 'temporal'
DEST.mkdir(parents=True,exist_ok=True)
sos = disenar_filtro()
p = propiedades(sos)


def leer(path):
    fs, samples=wavfile.read(path)
    if np.issubdtype(samples.dtype,np.integer):
        info=np.iinfo(samples.dtype)
        samples=samples.astype(np.float64)/max(abs(info.min),info.max)
    else:
        samples=samples.astype(np.float64)
    if samples.ndim>1: samples=samples.mean(axis=1)
    return fs,samples


def snr(s,y):
    return float(10*np.log10(np.sum(s*s)/max(np.sum((s-y)**2),1e-25)))


def mag(f):
    w,h=signal.sosfreqz(sos,worN=np.asarray(f),fs=16000)
    return 20*np.log10(np.maximum(abs(h),1e-14))

# Respuestas y especificaciones efectivas
f=np.array([500.,900.,1300.,1500.,1800.,2500.])
mag_db=mag(f)
fig,axs=plt.subplots(3,1,figsize=(9,10),layout='constrained')
axs[0].plot(p['frecuencias'],p['magnitud_db']); axs[0].set_ylabel('Magnitud (dB)');axs[0].set_ylim(-110,5)
axs[1].plot(p['frecuencias'],p['fase_rad']);axs[1].set_ylabel('Fase desenrollada (rad)')
axs[2].plot(p['frecuencias'],p['retardo_grupo_muestras']);axs[2].set_ylabel('Retardo de grupo (muestras)'); axs[2].set_ylim(0,65)
for ax in axs: ax.set_xlim(0,3500);ax.grid(True,alpha=.3);ax.set_xlabel('Frecuencia (Hz)')
fig.savefig(DEST/'respuestas_iir.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(6,6),layout='constrained')
ax.scatter(p['ceros'].real,p['ceros'].imag,facecolors='none',edgecolors='tab:blue',s=90,label='Ceros')
ax.scatter(p['polos'].real,p['polos'].imag,marker='x',color='tab:red',s=70,label='Polos')
phi=np.linspace(0,2*np.pi,500);ax.plot(np.cos(phi),np.sin(phi),'k--',alpha=.6,label='Círculo unidad')
ax.set(xlabel='Parte real',ylabel='Parte imaginaria',title='Polos y ceros del IIR Butterworth');ax.axis('equal');ax.grid(True,alpha=.3);ax.legend()
fig.savefig(DEST/'polos_ceros_iir.png',dpi=160);plt.close(fig)

rows=[]
for caso in range(1,4):
    carpeta=ROOT/'pruebas'/f'caso_{caso}'
    fs,s=leer(carpeta/'01_referencia_limpia.wav')
    _,x=leer(carpeta/'03_entrada_contaminada.wav')
    if fs!=16000 or len(s)!=len(x): raise ValueError('FS o longitud incompatibles')
    y=ecuacion_diferencias(x,sos)
    e,ok=validar(x[:2048],sos,tolerancia=1e-10)
    if not ok: raise AssertionError('Ecuación propia no coincide con SciPy')
    # Medición causal (la verdadera salida). Una alineación exploratoria es solo análisis offline.
    snr_in=snr(s,x);snr_out=snr(s,y)
    # Encontrar un desplazamiento global que reduzca error, sin usarse en filtrado real.
    candidates=range(0,33)
    best=min(candidates,key=lambda d: np.mean((s[:len(s)-d if d else None]-y[d:])**2))
    saligned=s[:len(s)-best] if best else s
    yaligned=y[best:]
    mse=float(np.mean((s-y)**2))
    aligned=float(np.mean((saligned-yaligned)**2))
    rows.append(dict(caso=caso,snr_entrada_db=snr_in,snr_temporal_causal_db=snr_out,delta_snr_causal_db=snr_out-snr_in,mse_causal=mse,retardo_alineacion_muestras=best,snr_alineada_offline_db=snr(saligned,yaligned),mse_alineada_offline=aligned,error_max_vs_scipy=e))
    t=np.arange(min(320,len(s)))/fs*1000
    fig,axs=plt.subplots(2,1,figsize=(10,7),layout='constrained')
    axs[0].plot(t,s[:len(t)],label='Limpia');axs[0].plot(t,x[:len(t)],label='Contaminada',alpha=.55);axs[0].plot(t,y[:len(t)],label='IIR causal',alpha=.9)
    axs[0].set(xlabel='Tiempo (ms)',ylabel='Amplitud normalizada',title=f'Caso {caso}: señales temporales');axs[0].legend();axs[0].grid(alpha=.3)
    n=len(x);fr=np.fft.rfftfreq(n,1/fs)
    for arr,label in [(s,'Limpia'),(x,'Contaminada'),(y,'IIR causal')]:
        sp=np.abs(np.fft.rfft(arr*np.hanning(n)))/n
        axs[1].plot(fr,20*np.log10(np.maximum(sp,1e-8)),label=label,alpha=.8)
    axs[1].set(xlim=(0,2800),ylim=(-110,0),xlabel='Frecuencia (Hz)',ylabel='Magnitud (dB, ref. amplitud normalizada)',title='Espectros comparados');axs[1].grid(alpha=.3);axs[1].legend()
    fig.savefig(DEST/f'caso_{caso}_comparacion_temporal.png',dpi=160);plt.close(fig)

with (DEST/'metricas_temporales.csv').open('w',newline='',encoding='utf-8') as out:
    writer=csv.DictWriter(out,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
with (DEST/'parametros_iir.json').open('w',encoding='utf-8') as out:
    json.dump(dict(fs_hz=16000,orden=12,corte_hz=1500,sos=sos.tolist(),frecuencias_evaluadas_hz=f.tolist(),magnitud_db=mag_db.tolist(),max_radio_polo=p['radio_polo_max'],sos_n_secciones=len(sos),bytes_coeficientes_float32=int(sos.size*4),bytes_estados_float32=int(len(sos)*2*4),bytes_bloque_float32=1024*4,radio_polo_float32=propiedades(cuantizacion(sos,np.float32))['radio_polo_max']),out,indent=2,ensure_ascii=False)
print('Filtro estable:',p['estable'],'radio:',p['radio_polo_max'])
print('Magnitudes:',dict(zip(f,mag_db.round(3))))
for r in rows: print(f"Caso {r['caso']}: SNR entrada {r['snr_entrada_db']:.2f}; causal {r['snr_temporal_causal_db']:.2f}; alineada {r['snr_alineada_offline_db']:.2f}; retardo {r['retardo_alineacion_muestras']} muestras")
print('Archivos generados en resultados/temporal')
