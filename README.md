# Filtros digitales: pasa bajos, pasa altos y pasa bandas

Diseño y análisis de filtros digitales en Python (SciPy y Matplotlib). El programa genera una señal con ruido, aplica filtros y compara las señales antes y después del filtrado.

## Qué hace

- Genera una señal de prueba: senoidales de 5, 50 y 200 Hz más ruido blanco (fs = 1000 Hz).
- Diseña filtros IIR (Butterworth y Chebyshev tipo I) y FIR (ventana Hamming).
- Aplica los filtros con `filtfilt` (fase cero) y grafica la señal y su espectro antes y después.
- Evalúa la efectividad con RMSE y SNR, y prueba distintas configuraciones (orden y frecuencia de corte).

## Requisitos

```
pip install -r requirements.txt
```

## Cómo ejecutarlo

```
python filtros_digitales.py
```

Las gráficas se guardan en la carpeta `figuras/` y las métricas se imprimen en la terminal.

## Resultados

![Pasa bajos](figuras/fig3_pasa_bajos.png)
![Pasa altos](figuras/fig4_pasa_altos.png)
![Pasa bandas](figuras/fig5_pasa_bandas.png)
![Comparación de métodos](figuras/fig6_comparacion_metodos.png)

| Filtro | SNR antes (dB) | SNR después (dB) |
|---|---|---|
| Pasa bajos Butterworth (20 Hz) | -1.85 | 18.13 |
| Pasa altos Butterworth (100 Hz) | -7.61 | -0.06 |
| Pasa bandas Butterworth (30-80 Hz) | -4.67 | 11.02 |
| Pasa bajos Chebyshev I (20 Hz) | -1.85 | 16.49 |
| Pasa bajos FIR Hamming (20 Hz) | -1.85 | 18.73 |

## Autor

Tu nombre - Materia - Universidad
