import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import scipy.signal as signal
from sklearn.decomposition import FastICA, PCA
from scipy.linalg import svd
from scipy.signal import wiener

# Archivos de entrada de audio
mic1 = "microfono1pintura.wav"
mic2 = "microfono-2-pintura.wav"  # Corregido el nombre del archivo
ruido_ambiente = "ruido sala de pintura.wav"

# Leer archivos de audio
frecuenciamuestreo_1, data_1 = wav.read(mic1)
frecuenciamuestreo_2, data_2 = wav.read(mic2)
frecuenciamuestreo_ruido, data_ruido = wav.read(ruido_ambiente)

# Crear diccionarios para datos y tasas de muestreo
audio_data = {"mic1": data_1, "mic2": data_2}
frecuencias_muestreo = {"mic1": frecuenciamuestreo_1, "mic2": frecuenciamuestreo_2}


# Función para calcular SNR
def calcular_snr(signal, noise):
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power == 0:  # Evitar división por cero
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)


# Calcular SNR antes del procesamiento
snr_antes = {key: calcular_snr(data, data_ruido[:len(data)]) for key, data in audio_data.items()}

# Mostrar SNR antes del procesamiento
print("SNR antes del procesamiento:")
for key, snr in snr_antes.items():
    print(f" {key}: {snr:.2f} dB")

# Visualizar forma de onda y espectrograma
fig = plt.figure(figsize=(12, 10))

for i, (key, data) in enumerate(audio_data.items()):
    sr = frecuencias_muestreo[key]
    time = np.linspace(0, len(data) / sr, num=len(data))

    # Graficar forma de onda
    plt.subplot(len(audio_data), 2, i * 2 + 1)
    plt.plot(time, data)
    plt.title(f"Forma de onda - {key}")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")

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

# Descomposición en valores singulares (SVD)
U, S, Vt = svd(signals_matrix, full_matrices=False)
senales_beamform = U[:, :2] @ np.diag(S[:2])

# Filtrar señales con el filtro de Wiener para reducir ruido
senales_sinruido = np.apply_along_axis(lambda x: wiener(x + 1e-10, mysize=29), 0, senales_beamform)

# Visualizar señales filtradas
plt.figure(figsize=(12, 6))
for i in range(2):
    plt.subplot(2, 1, i + 1)
    plt.plot(senales_sinruido[:, i])
    plt.title(f"Señal filtrada con Wiener - Microfono {i + 1}")
    plt.xlabel("Muestras")
    plt.ylabel("Amplitud")

plt.tight_layout()
plt.show()

# Manejo de valores atípicos (NaN o Inf)
if np.any(np.var(senales_sinruido, axis=0) == 0):
    senales_sinruido += np.random.normal(0, 1e-6, senales_sinruido.shape)

senales_sinruido = np.nan_to_num(senales_sinruido)

# Aplicar PCA para estabilizar las señales
pca = PCA(n_components=2, whiten=False, random_state=42)
pca_signals = pca.fit_transform(senales_sinruido)

# Verificar valores atípicos en las señales PCA
print("¿NaN en pca_signals?", np.isnan(pca_signals).any())
print("¿Inf en pca_signals?", np.isinf(pca_signals).any())
print("Varianzas de pca_signals:", np.var(pca_signals, axis=0))

# Aplicar ICA para separación de fuentes
ica = FastICA(n_components=2, max_iter=4000, tol=0.0001, random_state=42)
señales_mejoradas = ica.fit_transform(pca_signals)

# Visualizar señales separadas
plt.figure(figsize=(12, 6))
for i in range(2):
    plt.subplot(2, 1, i + 1)
    plt.plot(señales_mejoradas[:, i])
    plt.title(f"Señal Separada por ICA - Microfono {i + 1}")
    plt.xlabel("Muestras")
    plt.ylabel("Amplitud")

plt.tight_layout()
plt.show()

# Normalizar y guardar las señales separadas
output_files = {}
for i in range(2):
    # Normalizar la señal a un rango de [-1, 1]
    señal_normalizada = señales_mejoradas[:, i] / np.max(np.abs(señales_mejoradas[:, i]))

    # Escalar a int16 (rango de -32768 a 32767)
    señal_escalada = (señal_normalizada * 32767).astype(np.int16)

    # Guardar el archivo WAV
    output_file = f"voz_mejorada_{i + 1}.wav"
    wav.write(output_file, frecuenciamuestreo_1, señal_escalada)
    output_files[f"Voz Separada Mejorada {i + 1}"] = output_file

# Calcular SNR después del procesamiento
snr_despues = {}
for i in range(2):
    señal_procesada = señales_mejoradas[:, i]
    ruido_procesado = data_ruido[:len(señal_procesada)]
    snr_despues[f"Mic {i + 1}"] = calcular_snr(señal_procesada, ruido_procesado)

# Mostrar SNR después del procesamiento
print("\nSNR después del procesamiento:")
for key, snr in snr_despues.items():
    print(f" {key}: {snr:.2f} dB")

# Mostrar archivos generados
print("Voces Separadas Generadas:")
for key, path in output_files.items():
    print(f" {key}: {path}")