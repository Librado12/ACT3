"""
Diseño y análisis de filtros digitales: pasa bajos, pasa altos y pasa bandas.

Qué hace este programa:
  1. Genera una señal de prueba (3 senoidales + ruido blanco).
  2. Diseña filtros IIR (Butterworth, Chebyshev tipo I) y FIR (ventana Hamming).
  3. Aplica los filtros a la señal ruidosa.
  4. Grafica la señal antes y después, y su espectro de frecuencia.
  5. Calcula métricas (RMSE y SNR) para evaluar qué tan bien funcionó cada filtro.

Requisitos: pip install numpy scipy matplotlib
Ejecución:  python filtros_digitales.py
"""

# --- INICIO: importaciones ---
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
# --- FIN: importaciones ---

os.makedirs("figuras", exist_ok=True)

# ----------------------------------------------------------------------
# 1. DEFINICIÓN DE LA SEÑAL DE ENTRADA
# ----------------------------------------------------------------------
# --- INICIO: senal ---
fs = 1000                              # Frecuencia de muestreo: 1000 muestras por segundo (Hz)
duracion = 1.0                         # La señal dura 1 segundo
t = np.arange(0, duracion, 1 / fs)     # Vector de tiempo: 0, 0.001, 0.002, ... 0.999

# Tres componentes senoidales: una grave, una media y una aguda
f_baja, f_media, f_alta = 5, 50, 200   # Frecuencias en Hz
comp_baja = 1.0 * np.sin(2 * np.pi * f_baja * t)     # Amplitud 1.0
comp_media = 0.8 * np.sin(2 * np.pi * f_media * t)   # Amplitud 0.8
comp_alta = 0.6 * np.sin(2 * np.pi * f_alta * t)     # Amplitud 0.6

# Ruido blanco gaussiano (semilla fija para que el resultado sea repetible)
rng = np.random.default_rng(seed=42)
ruido = 0.5 * rng.standard_normal(t.size)

x_limpia = comp_baja + comp_media + comp_alta   # Señal "ideal", sin ruido
x = x_limpia + ruido                            # Señal que recibimos: con ruido
# --- FIN: senal ---


# --- INICIO: espectro ---
def espectro(senal, fs):
    """Devuelve (frecuencias, magnitud) de la señal usando la FFT."""
    N = len(senal)
    X = np.fft.rfft(senal)                  # FFT para señales reales
    f = np.fft.rfftfreq(N, d=1 / fs)        # Eje de frecuencias en Hz
    magnitud = 2 * np.abs(X) / N            # Normalizada: una senoidal de amplitud A da A
    return f, magnitud
# --- FIN: espectro ---


# ----------------------------------------------------------------------
# 2. DISEÑO DE LOS FILTROS
# ----------------------------------------------------------------------
# --- INICIO: diseno ---
orden = 4                  # Orden del filtro: más alto = transición más abrupta
fc_pb = 20                 # Pasa bajos: deja pasar por debajo de 20 Hz
fc_pa = 100                # Pasa altos: deja pasar por encima de 100 Hz
banda = (30, 80)           # Pasa bandas: deja pasar entre 30 y 80 Hz

# Filtros IIR Butterworth (devuelven coeficientes b y a)
b_pb, a_pb = signal.butter(orden, fc_pb, btype="lowpass", fs=fs)
b_pa, a_pa = signal.butter(orden, fc_pa, btype="highpass", fs=fs)
b_bp, a_bp = signal.butter(orden, banda, btype="bandpass", fs=fs)

# Filtro IIR Chebyshev tipo I (rizo máximo de 1 dB en la banda de paso)
b_cheb, a_cheb = signal.cheby1(orden, 1, fc_pb, btype="lowpass", fs=fs)

# Filtro FIR con ventana Hamming (101 coeficientes). Un FIR no tiene "a": se usa a = 1
b_fir = signal.firwin(101, fc_pb, window="hamming", fs=fs)
a_fir = 1.0
# --- FIN: diseno ---


# ----------------------------------------------------------------------
# 3. APLICACIÓN DE LOS FILTROS A LA SEÑAL
# ----------------------------------------------------------------------
# --- INICIO: aplicacion ---
# filtfilt filtra hacia adelante y hacia atrás: no desplaza la señal en el tiempo (fase cero)
y_pb = signal.filtfilt(b_pb, a_pb, x)           # Pasa bajos Butterworth
y_pa = signal.filtfilt(b_pa, a_pa, x)           # Pasa altos Butterworth
y_bp = signal.filtfilt(b_bp, a_bp, x)           # Pasa bandas Butterworth
y_cheb = signal.filtfilt(b_cheb, a_cheb, x)     # Pasa bajos Chebyshev
y_fir = signal.filtfilt(b_fir, a_fir, x)        # Pasa bajos FIR (ventana Hamming)
# --- FIN: aplicacion ---


# ----------------------------------------------------------------------
# 4. VISUALIZACIÓN DE RESULTADOS
# ----------------------------------------------------------------------
# --- INICIO: visualizacion ---
def graficar_antes_despues(titulo, y, objetivo, f_ref, archivo, color):
    """Dibuja la señal antes/después del filtro (arriba) y sus espectros (abajo)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.5))

    # Arriba: dominio del tiempo (solo los primeros 0.2 s para verlo con detalle)
    n = int(0.2 * fs)
    ax1.plot(t[:n], x[:n], color="0.65", lw=1, label="Entrada con ruido")
    ax1.plot(t[:n], objetivo[:n], "k--", lw=1.2, label="Componente que queremos conservar")
    ax1.plot(t[:n], y[:n], color=color, lw=2, label="Salida del filtro")
    ax1.set_title(titulo + " - dominio del tiempo")
    ax1.set_xlabel("Tiempo (s)")
    ax1.set_ylabel("Amplitud")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(alpha=0.3)

    # Abajo: dominio de la frecuencia (espectro)
    f, mx = espectro(x, fs)
    _, my = espectro(y, fs)
    ax2.plot(f, mx, color="0.65", lw=1, label="Entrada")
    ax2.plot(f, my, color=color, lw=1.5, label="Salida")
    for fr in f_ref:
        ax2.axvline(fr, color="red", ls=":", lw=1.2)
    ax2.set_title(titulo + " - espectro de frecuencia (líneas rojas = frecuencias de corte)")
    ax2.set_xlabel("Frecuencia (Hz)")
    ax2.set_ylabel("Magnitud")
    ax2.set_xlim(0, 300)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(archivo, dpi=150)
    plt.close(fig)
# --- FIN: visualizacion ---


# ----------------------------------------------------------------------
# 5. EVALUACIÓN DE LA EFECTIVIDAD
# ----------------------------------------------------------------------
# --- INICIO: metricas ---
def rmse(a, b):
    """Error cuadrático medio: qué tan lejos está 'a' de 'b' (menor = mejor)."""
    return np.sqrt(np.mean((a - b) ** 2))


def snr_db(objetivo, senal):
    """SNR en dB: potencia del objetivo / potencia del error (mayor = mejor)."""
    error = senal - objetivo
    return 10 * np.log10(np.sum(objetivo ** 2) / np.sum(error ** 2))


# Para cada filtro comparamos contra la componente que SÍ queríamos conservar
casos = {
    "Pasa bajos Butterworth": (y_pb, comp_baja),
    "Pasa altos Butterworth": (y_pa, comp_alta),
    "Pasa bandas Butterworth": (y_bp, comp_media),
    "Pasa bajos Chebyshev I": (y_cheb, comp_baja),
    "Pasa bajos FIR Hamming": (y_fir, comp_baja),
}
resultados = {}
for nombre, (salida, objetivo) in casos.items():
    resultados[nombre] = {
        "rmse_antes": rmse(x, objetivo),
        "rmse_despues": rmse(salida, objetivo),
        "snr_antes": snr_db(objetivo, x),
        "snr_despues": snr_db(objetivo, salida),
    }
    print(f"{nombre:26s} SNR: {resultados[nombre]['snr_antes']:6.2f} dB -> "
          f"{resultados[nombre]['snr_despues']:6.2f} dB | RMSE: "
          f"{resultados[nombre]['rmse_antes']:.3f} -> {resultados[nombre]['rmse_despues']:.3f}")
# --- FIN: metricas ---


# ----------------------------------------------------------------------
# GRÁFICAS (se guardan en la carpeta figuras/)
# ----------------------------------------------------------------------
azul, naranja, verde, morado = "#1F5FA8", "#E07B00", "#2E8B57", "#7B3FA0"

# Fig. 0: formas ideales de los tres filtros (solo para explicar el concepto)
fig, axs = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
fr = np.linspace(0, 100, 1000)
ideales = [
    ("Pasa bajos", (fr < 30).astype(float), azul, [30]),
    ("Pasa altos", (fr > 60).astype(float), naranja, [60]),
    ("Pasa bandas", ((fr > 35) & (fr < 70)).astype(float), verde, [35, 70]),
]
for ax, (nom, h, col, cortes) in zip(axs, ideales):
    ax.fill_between(fr, h, color=col, alpha=0.35)
    ax.plot(fr, h, color=col, lw=2)
    for c in cortes:
        ax.axvline(c, color="red", ls=":", lw=1.2)
    ax.set_title(nom)
    ax.set_xlabel("Frecuencia (u. arbitrarias)")
    ax.set_ylim(0, 1.25)
    ax.grid(alpha=0.3)
axs[0].set_ylabel("Ganancia (0 = bloquea, 1 = pasa)")
fig.tight_layout()
fig.savefig("figuras/fig0_filtros_ideales.png", dpi=150)
plt.close(fig)

# Fig. 1: señal de entrada (tiempo y espectro)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6))
n = int(0.2 * fs)
ax1.plot(t[:n], x[:n], color="0.6", lw=1, label="Señal con ruido (entrada)")
ax1.plot(t[:n], x_limpia[:n], color=azul, lw=2, label="Señal sin ruido (ideal)")
ax1.set_title("Señal de entrada - dominio del tiempo (primeros 0.2 s)")
ax1.set_xlabel("Tiempo (s)")
ax1.set_ylabel("Amplitud")
ax1.legend(loc="upper right", fontsize=8)
ax1.grid(alpha=0.3)
f, mx = espectro(x, fs)
ax2.plot(f, mx, color=azul, lw=1)
for fr_, txt in [(5, "5 Hz"), (50, "50 Hz"), (200, "200 Hz")]:
    ax2.annotate(txt, (fr_, np.interp(fr_, f, mx)), textcoords="offset points",
                 xytext=(6, 4), fontsize=9, color="red")
ax2.set_title("Espectro de la señal de entrada (3 picos + ruido en todas las frecuencias)")
ax2.set_xlabel("Frecuencia (Hz)")
ax2.set_ylabel("Magnitud")
ax2.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("figuras/fig1_senal_entrada.png", dpi=150)
plt.close(fig)

# Fig. 2: respuesta en frecuencia de los tres filtros Butterworth
fig, axs = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
disenos = [
    ("Pasa bajos (fc = 20 Hz)", b_pb, a_pb, azul, [20]),
    ("Pasa altos (fc = 100 Hz)", b_pa, a_pa, naranja, [100]),
    ("Pasa bandas (30 - 80 Hz)", b_bp, a_bp, verde, [30, 80]),
]
for ax, (nom, b, a, col, cortes) in zip(axs, disenos):
    w, h = signal.freqz(b, a, worN=4096, fs=fs)
    ax.plot(w, 20 * np.log10(np.maximum(np.abs(h), 1e-6)), color=col, lw=2)
    for c in cortes:
        ax.axvline(c, color="red", ls=":", lw=1.2)
    ax.axhline(-3, color="0.4", ls="--", lw=0.8)
    ax.set_title(nom, fontsize=10)
    ax.set_xlabel("Frecuencia (Hz)")
    ax.set_xlim(0, 300)
    ax.set_ylim(-80, 5)
    ax.grid(alpha=0.3)
axs[0].set_ylabel("Magnitud (dB)")
fig.tight_layout()
fig.savefig("figuras/fig2_respuesta_frecuencia.png", dpi=150)
plt.close(fig)

# Figs. 3, 4, 5: antes y después de cada filtro Butterworth
graficar_antes_despues("Filtro pasa bajos", y_pb, comp_baja, [fc_pb],
                       "figuras/fig3_pasa_bajos.png", azul)
graficar_antes_despues("Filtro pasa altos", y_pa, comp_alta, [fc_pa],
                       "figuras/fig4_pasa_altos.png", naranja)
graficar_antes_despues("Filtro pasa bandas", y_bp, comp_media, list(banda),
                       "figuras/fig5_pasa_bandas.png", verde)

# Fig. 6: comparación de métodos de diseño (pasa bajos, fc = 20 Hz)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
metodos = [
    ("Butterworth (orden 4)", b_pb, a_pb, azul),
    ("Chebyshev I (orden 4, 1 dB)", b_cheb, a_cheb, naranja),
    ("FIR Hamming (101 coef.)", b_fir, a_fir, morado),
]
for nom, b, a, col in metodos:
    w, h = signal.freqz(b, a, worN=4096, fs=fs)
    mag = np.abs(h)
    ax1.plot(w, 20 * np.log10(np.maximum(mag, 1e-6)), color=col, lw=2, label=nom)
    ax2.plot(w, mag, color=col, lw=2, label=nom)
ax1.axvline(fc_pb, color="red", ls=":", lw=1.2)
ax1.set_title("Magnitud en dB (vista completa)")
ax1.set_xlabel("Frecuencia (Hz)")
ax1.set_ylabel("Magnitud (dB)")
ax1.set_xlim(0, 150)
ax1.set_ylim(-90, 5)
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)
ax2.axvline(fc_pb, color="red", ls=":", lw=1.2)
ax2.set_title("Magnitud lineal (zoom a la banda de paso)")
ax2.set_xlabel("Frecuencia (Hz)")
ax2.set_ylabel("Magnitud")
ax2.set_xlim(0, 40)
ax2.set_ylim(0, 1.15)
ax2.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("figuras/fig6_comparacion_metodos.png", dpi=150)
plt.close(fig)

# Fig. 7: efecto del orden del filtro (pasa bajos Butterworth)
fig, ax = plt.subplots(figsize=(7, 4))
for o, col in zip([2, 4, 8], [azul, naranja, verde]):
    b, a = signal.butter(o, fc_pb, btype="lowpass", fs=fs)
    w, h = signal.freqz(b, a, worN=4096, fs=fs)
    ax.plot(w, 20 * np.log10(np.maximum(np.abs(h), 1e-6)), color=col, lw=2, label=f"Orden {o}")
ax.axvline(fc_pb, color="red", ls=":", lw=1.2)
ax.set_xlim(0, 150)
ax.set_ylim(-90, 5)
ax.set_title("Efecto del orden en un pasa bajos Butterworth (fc = 20 Hz)")
ax.set_xlabel("Frecuencia (Hz)")
ax.set_ylabel("Magnitud (dB)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("figuras/fig7_efecto_orden.png", dpi=150)
plt.close(fig)

# ----------------------------------------------------------------------
# 6. EXPERIMENTOS CON DIFERENTES CONFIGURACIONES
# ----------------------------------------------------------------------
# --- INICIO: experimentos ---
def snr_pasa_bajos(orden_, fc_):
    """SNR de salida de un pasa bajos Butterworth con el orden y la fc indicados."""
    b, a = signal.butter(orden_, fc_, btype="lowpass", fs=fs)
    return snr_db(comp_baja, signal.filtfilt(b, a, x))


experimentos = []
for o in (2, 4, 8):          # Distintos órdenes, con fc = 20 Hz
    experimentos.append((f"Orden {o}, fc = 20 Hz", snr_pasa_bajos(o, 20)))
for fc in (10, 20, 40):      # Distintas frecuencias de corte, con orden = 4
    experimentos.append((f"Orden 4, fc = {fc} Hz", snr_pasa_bajos(4, fc)))
for etiqueta, snr in experimentos:
    print(f"{etiqueta:20s} -> SNR de salida: {snr:6.2f} dB")
# --- FIN: experimentos ---

# Guardamos los números para usarlos en el informe
with open("resultados.json", "w", encoding="utf-8") as fh:
    json.dump({"metricas": resultados, "experimentos": experimentos}, fh, indent=2)
