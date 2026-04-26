import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

fs = 1000  # Частота дискретизации, Гц
N = 5000  # Количество отсчетов
T = 2.0  # Период
tau = 0.5 * T  # τ = 0.5T = 1.0 с
alpha = 2.5  # α = 2.5

# Временная ось
t = np.arange(N) / fs


def signal_variant_9(t, T, tau, alpha):
    s = np.zeros_like(t)
    mask1 = (t >= 0) & (t <= tau)
    t_valid = t[mask1]
    s[mask1] = np.exp(-2 * (alpha * (t_valid / tau - 0.5)) ** 2)
    return s


# 1. Генерация сигнала
s = signal_variant_9(t, T, tau, alpha)
s_real = s  # сигнал вещественный

# 2. Сигнал во временной области
plt.figure(1, figsize=(12, 6))
plt.plot(t[:2000], s_real[:2000])
plt.title(f'Сигнал во временной области s(t)')
plt.xlabel('Время, с')
plt.ylabel('Амплитуда')
plt.grid(True)
plt.xlim(0, 2)
plt.show()

# 3. ДПФ и спектры
S = np.fft.fft(s_real)
freq = np.fft.fftfreq(N, 1 / fs)

# Амплитудный спектр
plt.figure(2, figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(freq[:N // 2], np.abs(S)[:N // 2])
plt.title('Амплитудный спектр')
plt.xlabel('Частота, Гц')
plt.ylabel('Амплитуда')
plt.grid(True)
plt.xlim(0, 50)

# Фазовый спектр
plt.subplot(2, 1, 2)
plt.plot(freq[:N // 2], np.angle(S)[:N // 2])
plt.title('Фазовый спектр')
plt.xlabel('Частота, Гц')
plt.ylabel('Фаза, рад')
plt.grid(True)
plt.xlim(0, 50)
plt.tight_layout()
plt.show()

# 4. Отображение в bin и Гц
bin_axis = np.arange(N)
plt.figure(3, figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(bin_axis[:N // 2], np.abs(S)[:N // 2])
plt.title('Амплитудный спектр (в bin)')
plt.xlabel('bin')
plt.ylabel('Амплитуда')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(freq[:N // 2], np.abs(S)[:N // 2])
plt.title('Амплитудный спектр (в Гц)')
plt.xlabel('Частота, Гц')
plt.ylabel('Амплитуда')
plt.grid(True)
plt.tight_layout()
plt.show()

# 5. Zero-padding: спектры только в бинах!
print("5. Zero-padding анализ:")
for pad_len in [24, 1024]:
    s_pad = np.pad(s_real, (0, pad_len))
    S_pad = np.fft.fft(s_pad)
    bin_axis_pad = np.arange(len(S_pad))

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(bin_axis_pad[:len(S_pad) // 2], np.abs(S_pad)[:len(S_pad) // 2])
    plt.title(f'Амплитудный спектр с добавлением {pad_len} нулей (в bin)')
    plt.xlabel('bin')
    plt.ylabel('Амплитуда')
    plt.grid(True)
    plt.xlim(0, len(S_pad) // 2)

    plt.subplot(2, 1, 2)
    plt.plot(bin_axis_pad[:len(S_pad) // 2], np.angle(S_pad)[:len(S_pad) // 2])
    plt.title(f'Фазовый спектр с добавлением {pad_len} нулей (в bin)')
    plt.xlabel('bin')
    plt.ylabel('Фаза, рад')
    plt.grid(True)
    plt.xlim(0, len(S_pad) // 2)
    plt.tight_layout()
    plt.show()

    print(f"С {pad_len} нулями: частотное разрешение = {fs / (N + pad_len):.3f} Гц")

# 6. Обратное БПФ
s_recovered = np.fft.ifft(S)
plt.figure(6, figsize=(12, 6))
plt.plot(t[:2000], s_real[:2000], 'b-', label='Исходный', linewidth=2)
plt.plot(t[:2000], s_recovered.real[:2000], 'r--', label='Восстановленный', alpha=0.7)
plt.title('Сравнение исходного и восстановленного сигналов')
plt.xlabel('Время, с')
plt.ylabel('Амплитуда')
plt.legend()
plt.grid(True)
plt.xlim(0, 2)
plt.show()

# Проверка точности восстановления
error = np.max(np.abs(s_real - s_recovered.real))
print(f"6. Максимальная ошибка восстановления: {error:.2e}")

# 7. Спектрограммы
print("7. Построение спектрограмм:")
f_sp1, t_sp1, Sxx1 = spectrogram(s_real, fs, window='boxcar', nperseg=20, noverlap=10)
f_sp2, t_sp2, Sxx2 = spectrogram(s_real, fs, window='hann', nperseg=20, noverlap=10)

plt.figure(7, figsize=(12, 8))
# Прямоугольное окно
plt.subplot(2, 1, 1)
plt.pcolormesh(t_sp1, f_sp1, 10 * np.log10(Sxx1 + 1e-12), shading='gouraud')
plt.title('Спектрограмма (прямоугольное окно)')
plt.ylabel('Частота, Гц')
plt.colorbar(label='Мощность, дБ')
plt.ylim(0, 100)
plt.xlim(0, 2)

# Окно Ханна
plt.subplot(2, 1, 2)
plt.pcolormesh(t_sp2, f_sp2, 10 * np.log10(Sxx2 + 1e-12), shading='gouraud')
plt.title('Спектрограмма (окно Ханна)')
plt.xlabel('Время, с')
plt.ylabel('Частота, Гц')
plt.colorbar(label='Мощность, дБ')
plt.ylim(0, 100)
plt.xlim(0, 2)
plt.tight_layout()
plt.show()

# 8. Выводы
print("\n8. ВЫВОДЫ:")
print(f"- Вариант 9: Экспоненциальный импульс с формулой s(t) = exp[-2α(t/τ - 0.5)²]")
print(f"- Параметры: τ = {tau} с (0.5T), T = {T} с, α = {alpha}")
print(f"- На интервале [0, {tau}] с сигнал имеет максимум в точке t = τ/2 = {tau / 2} с")
print(f"- Максимальная амплитуда: s(τ/2) = exp(0) = 1.0")
print(f"- На краях интервала [0, τ]: s(0) = s(τ) = exp(-2α·0.5²) = {np.exp(-2 * alpha * (0.5) ** 2):.4f}")
print(f"- На интервале [{tau}, {T}] с сигнал тождественно равен 0")
print(f"- Спектр имеет колоколообразную форму, характерную для гауссоподобных импульсов")
print(f"- Zero-padding улучшает визуализацию спектра без изменения информации")
print(f"- ОБПФ точно восстанавливает сигнал (ошибка ~ {error:.2e})")
print(f"- Спектрограмма показывает энергию только на интервале [0, {tau}], где присутствует сигнал")
print(f"- Окно Ханна дает лучшее частотное разрешение на спектрограмме")
