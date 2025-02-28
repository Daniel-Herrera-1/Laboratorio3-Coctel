# Laboratorio3-Coctel


# Análisis y Separación de Señales de Voz en un Entorno Ruidoso

## Introducción

El problema de la "fiesta de cóctel" se refiere a la capacidad de un sistema para concentrarse en una sola fuente sonora mientras filtra las demás en un entorno con múltiples emisores de sonido. Este problema es común en sistemas de audición tanto humanos como artificiales, y su resolución es esencial en aplicaciones como la mejora de la voz, el reconocimiento de habla y la cancelación de ruido.

En esta práctica se busca aplicar técnicas avanzadas de procesamiento de señales para analizar y mejorar la calidad de señales de voz capturadas en un ambiente ruidoso. Se emplearán técnicas como la Transformada de Fourier, Análisis de Componentes Independientes (ICA) y el Filtro de Wiener para extraer la señal deseada.

## Objetivos

- Aplicar el análisis en frecuencia de señales de voz en un problema de captura de señales mezcladas.

- Implementar métodos de separación de fuentes sonoras utilizando técnicas de procesamiento de señales.

- Evaluar la efectividad de los métodos aplicados mediante métricas como la Relación Señal-Ruido (SNR).

- Desarrollar una solución computacional que permita extraer señales de interés en un entorno de múltiples fuentes.

## Requerimientos

### Materiales y Software:

- Computador con acceso a internet

## Python y librerías:

- numpy

- matplotlib

- scipy

- sklearn (scikit-learn)

- Sistema de adquisición de datos (tarjeta de sonido o interfaz de audio)

- Micrófonos (mínimo 2)


## Explicación del Código de Procesamiento de Señales de Audio

Este código tiene como objetivo mejorar la calidad del audio capturado por dos micrófonos en un entorno ruidoso. Utiliza técnicas de filtrado, análisis espectral, descomposición en valores singulares (SVD), filtrado de Wiener, análisis de componentes principales (PCA) y análisis de componentes independientes (ICA) para separar la voz del ruido. A continuación, se explica paso a paso cada sección del código.


Para el procesamiento de las señales de audio, se emplean diversas librerías de Python:

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import scipy.signal as signal
from sklearn.decomposition import FastICA, PCA
from scipy.linalg import svd
from scipy.signal import wiener
```
- numpy: Facilita operaciones con matrices y cálculos matemáticos avanzados.

- matplotlib: Permite generar gráficos para visualizar las señales.

- scipy.io.wavfile: Proporciona funciones para leer y escribir archivos de audio en formato WAV.

- scipy.signal: Contiene herramientas para el análisis y filtrado de señales.

- sklearn.decomposition: Incluye métodos como PCA e ICA para analizar y separar señales.

- scipy.linalg: Implementa descomposición en valores singulares (SVD) utilizada en la separación de fuentes.

- scipy.signal.wiener: Aplica el filtro de Wiener para reducción de ruido.

 ## 2. Carga de Archivos de Audio

 El código carga tres archivos de audio que representan las señales capturadas por dos micrófonos y el ruido de fondo:
 
```python
 
mic1 = "microfono1pintura.wav"
mic2 = "microfono-2-pintura.wav"
ruido_ambiente = "ruido sala de pintura.wav"
```
Se utilizan estas rutas de archivo para leer los datos de audio con wav.read():

```python
frecuenciamuestreo_1, data_1 = wav.read(mic1)
frecuenciamuestreo_2, data_2 = wav.read(mic2)
frecuenciamuestreo_ruido, data_ruido = wav.read(ruido_ambiente)
```

Cada archivo se lee obteniendo dos elementos: la frecuencia de muestreo y los datos de la señal en forma de matriz.

## 3. Cálculo de la Relación Señal-Ruido (SNR)

La Relación Señal-Ruido (SNR) mide la calidad de la señal respecto al ruido presente:


```python
def calcular_snr(signal, noise):
    signal_power = np.mean(signal ** 2)  # Calcula la potencia de la señal
    noise_power = np.mean(noise ** 2)  # Calcula la potencia del ruido
    if noise_power == 0:
        return float('inf')  # Evita división por cero
    return 10 * np.log10(signal_power / noise_power)  # Convierte la relación en dB

 # mic1: -8.73 dB
 # mic2: -10.28 dB
```

## 4. Visualización de las Señales

Se generan gráficos para analizar las señales en el dominio del tiempo y la frecuencia.

##Forma de onda

Cada señal capturada por los micrófonos se representa como una función de amplitud en el tiempo:

```python
plt.plot(time, data)
plt.title(f"Forma de onda - {key}")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
```

![image](https://github.com/user-attachments/assets/9f27e548-f990-4f9a-a4f5-a0e0e880977a)
