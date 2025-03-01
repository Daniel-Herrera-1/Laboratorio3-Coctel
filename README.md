# Laboratorio3-Coctel

## NOTA: El SNR final obtenido tras el procesamiento indica la relación entre la señal útil y el ruido residual. Un SNR más bajo de lo esperado podría deberse a factores como:

- Ubicación de los micrófonos: Si los micrófonos estaban demasiado cerca entre sí, pudieron captar señales muy similares, lo que dificulta la separación efectiva de fuentes.
- Condiciones del entorno: Si la sala no estaba lo suficientemente en silencio, el ruido ambiental pudo haberse mezclado con la señal principal, afectando la calidad del resultado.

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

## Filtrado con aplicación del PCA y el ICA
Para estas tecnicas toca tener en cuenta lo siguiente:  si alguna columna de la matriz senales_sinruido tiene varianza cero indica que todos los valores en esa columna son iguales, lo que puede ser problemático para técnicas como PCA o ICA, ya que no aportan información útil.
Si se detecta varianza cero, se añade un pequeño ruido gaussiano  a las señales.

Manejo de valores NaN o Inf:
La función np.nan_to_num convierte cualquier valor NaN (Not a Number) o Inf (Infinito) en un número finito (generalmente 0). Esto es importante porque valores NaN o Inf pueden causar errores en cálculos posteriores.

* Primero se aplica PCA (Análisis de Componentes Principales) Reducir la dimensionalidad de las señales y eliminar correlaciones entre ellas.

Parámetros:
n_components=2: Se conservan solo 2 componentes principales
whiten=False: No se normaliza la varianza de los componentes.
random_state=42: Fija la semilla aleatoria para reproducibilidad.

```python
pca = PCA(n_components=2, whiten=False, random_state=42)
pca_signals = pca.fit_transform(senales_sinruido)
```

pca.fit_transform(senales_sinruido): Ajusta el modelo PCA a las señales y las transforma en un espacio de menor dimensionalidad.
Las señales resultantes (pca_signals) son una versión estabilizada y reducida de las señales originales.

* Segundo por ICA (Análisis de Componentes Independientes):Separar las fuentes de audio independientes que están mezcladas en las señales.

Parámetros:
n_components=2: 2 componentes independientes (una por cada fuente de audio).
max_iter=4000: se necesitan varias iteraciones para converger.
tol=0.0001: Tolerancia para detener el algoritmo si la convergencia es suficiente.
random_state=42: semilla aleatoria.

```python
ica = FastICA(n_components=2, max_iter=4000, tol=0.0001, random_state=42)
señales_mejoradas = ica.fit_transform(pca_signals)
```

ica.fit_transform(pca_signals): Ajusta el modelo ICA a las señales PCA y las transforma en componentes independientes.
Las señales resultantes (señales_mejoradas) son las fuentes de audio separadas.


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

Interpretacion: el proceso de mejorar las señales de audio capturadas mediante dos pasos principales: extraer las características más relevantes de las señales (reduciendo redundancias y ruido) y aplicar un filtrado que suavice aún más la señal. En resumen, lo que hace este bloque es:


## Manejo de Valores Atipicos

```python
# Verificar si alguna columna de 'senales_sinruido' tiene varianza cero.
if np.any(np.var(senales_sinruido, axis=0) == 0):
    print("¡Atención! Algunas columnas tienen varianza cero. Añadiendo ruido...")
    # Si se detecta que alguna señal es constante (varianza cero), se añade un pequeño ruido gaussiano
    # (media = 0, desviación = 1e-6) a toda la matriz para evitar problemas en análisis posteriores.
    senales_sinruido += np.random.normal(0, 1e-6, senales_sinruido.shape)

# Reemplazar cualquier valor NaN (no numérico) o Inf (infinito) por 0, garantizando datos válidos.
senales_sinruido = np.nan_to_num(senales_sinruido)

# Aplicar PCA para reducir la dimensionalidad y estabilizar las señales.
# Se configuran 2 componentes principales (una para cada micrófono, en este caso) sin aplicar 'whitening'.
pca = PCA(n_components=2, whiten=False, random_state=42)
pca_signals = pca.fit_transform(senales_sinruido)

# Verificar que la transformación con PCA no haya generado valores anómalos:
print("¿NaN en pca_signals?", np.isnan(pca_signals).any())
print("¿Inf en pca_signals?", np.isinf(pca_signals).any())
print("Varianzas de pca_signals:", np.var(pca_signals, axis=0))

#¿NaN en pca_signals? False
#¿Inf en pca_signals? False
#Varianzas de pca_signals: [5928396.2545232  4285693.79366472]
```

###  *1.Corrección de Valores Problemáticos:*

- Se calcula la varianza de cada columna en la matriz **senales_sinruido.**

- Si alguna columna tiene varianza cero, significa que esa señal es constante (no varía), lo que puede generar errores en pasos posteriores.

- En caso de detectar una varianza cero, se imprime una advertencia y se añade un ruido muy pequeño (con media 0 y desviación 1e-6) a toda la matriz, ayudando a "despertar" la variación en esa señal.

- Luego, se usa **np.nan_to_num** para reemplazar valores no numéricos (NaN) o infinitos por cero, asegurando que los datos sean adecuados para el procesamiento.

  
## 2. Aplicación de PCA (Análisis de Componentes Principales):

- Se instancia un objeto PCA configurado para extraer 2 componentes principales. Esto ayuda a reducir la dimensionalidad de los datos y a estabilizar las señales, enfocándose en las variaciones más importantes.

- La transformación se aplica a senales_sinruido con pca.fit_transform(), generando pca_signals que contendrá las dos componentes principales de la señal.
  
## 3. Verificación de la Transformación:

- Se realizan comprobaciones para confirmar que pca_signals no contenga valores NaN o infinitos.
- Se imprimen las varianzas de las dos componentes principales para asegurar que la transformación captura la variabilidad esencial de las señales.

## Aplicar ICA

separar las fuentes mezcladas y luego visualiza las señales resultantes:

```pyhon
# Aplicar ICA para separación de fuentes
ica = FastICA(n_components=2, max_iter=3000, tol=0.0001, random_state=42)
enhanced_signals = ica.fit_transform(pca_signals)
```
- FastICA: Se configura para extraer 2 componentes (correspondientes a 2 fuentes).
- fit_transform: Se aplica a las señales estabilizadas por PCA (pca_signals) para obtener las señales separadas.

```python
# Visualizar señales separadas
plt.figure(figsize=(12, 6))
for i in range(2):
    plt.subplot(2, 1, i + 1)
    plt.plot(enhanced_signals[:, i])
    plt.title(f"Señal Separada por ICA - Microfono {i + 1}")
    plt.xlabel("Tiempo")
    plt.ylabel("Amplitud")

plt.tight_layout()
plt.show()
```
- Se crea una figura y se generan dos subgráficas (una por cada señal separada).
- Cada gráfica muestra la evolución en el tiempo (eje X) de la amplitud (eje Y) de la señal separada.
- Esto permite visualizar cómo ICA ha extraído las fuentes individuales de la mezcla original.

![image](https://github.com/user-attachments/assets/e554de44-3ab9-4e85-a9c6-c3a7051044c1)


## Normalizacion

```python
# Normalizar y guardar las señales separadas
output_files = {}
for i in range(2):
    # Normalizar la señal para que sus valores queden en el rango [-1, 1]
    señal_normalizada = enhanced_signals[:, i] / np.max(np.abs(enhanced_signals[:, i]))
    
    # Escalar la señal normalizada a valores enteros de 16 bits (rango: -32768 a 32767)
    señal_escalada = (señal_normalizada * 32767).astype(np.int16)
    
    # Definir el nombre del archivo WAV de salida para la señal separada
    output_file = f"voz_mejorada_{i + 1}.wav"
    
    # Guardar la señal escalada en un archivo WAV usando la frecuencia de muestreo original
    wav.write(output_file, frecuenciamuestreo_1, señal_escalada)
    
    # Almacenar el nombre del archivo en un diccionario para referencia
    output_files[f"Voz Separada Mejorada {i + 1}"] = output_file
```
- **Normalización:**
Se divide cada señal (enhanced_signals[:, i]) por su valor máximo absoluto para que todos sus valores estén entre -1 y 1.

- **Escalado a int16**:
La señal normalizada se multiplica por 32767 (valor máximo en 16 bits) y se convierte a tipo entero (int16), adaptándola al formato de audio estándar.

- **Guardado**:
La señal procesada se guarda en un archivo WAV usando la frecuencia de muestreo original, y se registra su nombre en el diccionario output_files para referencia futura.


## Calculo SNR Y GUARDADO


```python
# Calcular SNR después del procesamiento
snr_despues = {}
for i in range(2):
    # Selecciona la señal separada para el micrófono i
    señal_procesada = enhanced_signals[:, i]
    # Usa el ruido original (recortado al tamaño de la señal) para el cálculo del SNR
    ruido_procesado = data_ruido[:len(señal_procesada)]
    # Calcula el SNR y lo guarda en el diccionario, etiquetado por el micrófono
    snr_despues[f"Mic {i + 1}"] = calcular_snr(señal_procesada, ruido_procesado)

# Mostrar SNR después del procesamiento
print("\nSNR después del procesamiento:")
for key, snr in snr_despues.items():
    print(f" {key}: {snr:.2f} dB")

# Mostrar los nombres de los archivos generados con las voces separadas
print("Voces Separadas Generadas:")
for key, path in output_files.items():
    print(f" {key}: {path}")

#SNR después del procesamiento:
#Mic 1: -36.46 dB
 #Mic 2: -36.46 dB
#Voces Separadas Generadas:
 #Voz Separada Mejorada 1: voz_mejorada_1.wav
 #Voz Separada Mejorada 2: voz_mejorada_2.wav
```

- **Cálculo del SNR:**
Para cada señal separada (de cada micrófono), se calcula la Relación Señal-Ruido (SNR) usando la función calcular_snr. Se recorta el ruido (data_ruido) para que tenga la misma longitud que la señal procesada. Los valores se almacenan en el diccionario snr_despues.

- **Mostrar SNR:**
Se imprime el SNR de cada micrófono en decibelios, permitiendo evaluar la mejora en la calidad de la señal tras el procesamiento.

- **Mostrar Archivos Generados:**
Se listan los nombres de los archivos WAV que contienen las voces separadas, almacenados en output_files.
