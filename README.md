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

### Forma de onda

Cada señal capturada por los micrófonos se representa como una función de amplitud en el tiempo:

```python
plt.plot(time, data)
plt.title(f"Forma de onda - {key}")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
```

![image](https://github.com/user-attachments/assets/9f27e548-f990-4f9a-a4f5-a0e0e880977a)

### Interpretación: Este gráfico muestra cómo varía la amplitud de la señal en el tiempo. Permite visualizar ruidos, patrones y características de la voz capturada.

## Espectrograma

El espectrograma muestra la distribución de la energía de la señal en función del tiempo y la frecuencia:

este fragmento de código primero genera una representación visual (espectrograma) de la señal de audio, lo que ayuda a analizar la distribución temporal y frecuencial de la energía, y luego prepara las señales de entrada asegurándose de que tengan la misma longitud, facilitando así el procesamiento conjunto en etapas posteriores.

```python
# Calcular y graficar espectrograma
    f, t, dep = signal.spectrogram(data, sr)
    plt.subplot(len(audio_data), 2, i * 2 + 2)
    eps = 1e-10
    plt.imshow(10 * np.log10(dep + eps), aspect='auto', origin='lower',
               extent=[t.min(), t.max(), f.min(), f.max()])
    plt.title(f"Espectrograma - {key}")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Frecuencia (Hz)")

plt.tight_layout()
plt.show()

# Alinear señales para asegurar longitud mínima en común
min_length = min(len(data_1), len(data_2))
signals_matrix = np.vstack([data_1[:min_length], data_2[:min_length]]).T
```
![image](https://github.com/user-attachments/assets/7cc203b6-88d9-4964-86ab-5e8d1ecd4209)


Interpretación: Un espectrograma permite ver qué frecuencias están presentes en la señal y cómo cambian con el tiempo. Es útil para detectar ruido y componentes no deseados.

## 5. Procesamiento de Señales

- Extraer las componentes principales de la señal mediante SVD
- Reducir el ruido de estas señales utilizando un filtro de Wiener y luego visualizar el resultado

```python
U, S, Vt = svd(signals_matrix, full_matrices=False)
senales_beamform = U[:, :2] @ np.diag(S[:2])
```
El código utiliza la Descomposición en Valores Singulares (SVD) para analizar la matriz de señales (ya alineadas en longitud) obtenida previamente. Con SVD, la matriz signals_matrix se factoriza en tres componentes:

- U: Matriz de vectores ortonormales en el espacio de las muestras, que captura la forma en que se distribuye la energía en el tiempo.
- S: Vector de valores singulares, donde cada valor indica la cantidad de energía (o importancia) de cada componente.
- Vt: Matriz que contiene los coeficientes de combinación para reconstruir las señales originales a partir de los componentes.

## Filtrado con el Filtro de Wiener para Reducción de Ruido

Una vez obtenida la señal beamformed, se procede a reducir aún más el ruido utilizando el filtro de Wiener
```python
senales_sinruido = np.apply_along_axis(lambda x: wiener(x + 1e-10, mysize=29), 0, senales_beamform)
```

Aquí se realiza lo siguiente:

- Se aplica, para cada columna (cada señal) de la matriz senales_beamform, una función lambda que suma un pequeño valor (1e-10) a la señal para evitar problemas numéricos (como dividir por cero) y luego aplica el filtro de Wiener.
- El parámetro mysize=29 determina el tamaño de la ventana que utiliza el filtro para estimar la media y la varianza local, adaptándose a las características de la señal en cada segmento.

El filtro de Wiener es un método adaptativo que reduce el ruido estimando la señal “limpia” en base a las variaciones locales de la misma. El resultado es senales_sinruido, una matriz en la que cada columna es una señal con una reducción significativa del ruido.

## Visualización de las Señales Filtradas

Finalmente, el código genera gráficos para visualizar las señales después del filtrado:

```python
plt.figure(figsize=(12, 6))
for i in range(2):
    plt.subplot(2, 1, i + 1)
    plt.plot(senales_sinruido[:, i])
    plt.title(f"Señal filtrada con Wiener - Microfono {i + 1}")
    plt.xlabel("Muestras")
    plt.ylabel("Amplitud")

plt.tight_layout()
plt.show()
```
![image](https://github.com/user-attachments/assets/4eb66bf3-acc3-4667-b67a-72addc38b8bf)

