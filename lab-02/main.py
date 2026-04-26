import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# Исходная функция и параметры
pi = np.pi
f = lambda x: pi - x
# f(x) = π - x на (0, π)

# интервал
x = np.linspace(0, pi, 1000)

# значения исходной функции
y = f(x)

# Числа членов ряда для графиков
N_values = [10, 50, 200]


def compute_cosine_coefficients(N):
    a0_integral, _ = integrate.quad(lambda x: f(x), 0, pi)
    a0 = (2.0 / pi) * a0_integral

    a_coeffs = [a0]  # начинаем с a_0

    for n in range(1, N + 1):
        # Интеграл: ∫₀^π (π - x)*cos(n*x) dx
        a_n_integral, _ = integrate.quad(lambda x: f(x) * np.cos(n * x), 0, pi)
        a_n = (2.0 / pi) * a_n_integral
        a_coeffs.append(a_n)

    return a_coeffs


def fourier_cosine_partial(x, a_coeffs):
    a0 = a_coeffs[0]
    S = a0 / 2.0

    for n in range(1, len(a_coeffs)):
        S += a_coeffs[n] * np.cos(n * x)

    return S


def compute_sine_coefficients(N):
    b_coeffs = []

    for n in range(1, N + 1):
        # Интеграл: ∫₀^π (π - x)*sin(n*x) dx
        b_n_integral, _ = integrate.quad(lambda x: f(x) * np.sin(n * x), 0, pi)
        b_n = (2.0 / pi) * b_n_integral
        b_coeffs.append(b_n)

    return b_coeffs


def fourier_sine_partial(x, b_coeffs):
    S = np.zeros_like(x, dtype=float)

    for n in range(1, len(b_coeffs) + 1):
        S += b_coeffs[n - 1] * np.sin(n * x)

    return S


# Косинус-ряд: коэффициенты при N=10 (посчитаны вручную, только нечётные)
a_coeffs_10_manual = [
    3.1415926536,  # a₀ = π
    1.2732395447,  # a₁ = 4/(π·1²)
    0.0,  # a₂ = 0 (чётное)
    0.1414710605,  # a₃ = 4/(π·3²)
    0.0,  # a₄ = 0 (чётное)
    0.0509295818,  # a₅ = 4/(π·5²)
    0.0,  # a₆ = 0 (чётное)
    0.0259844805,  # a₇ = 4/(π·7²)
    0.0,  # a₈ = 0 (чётное)
    0.0157190067,  # a₉ = 4/(π·9²)
    0.0  # a₁₀ = 0 (чётное)
]

# Синус-ряд: коэффициенты при N=10 (посчитаны вручную)
b_coeffs_10_manual = [
    2.0,  # b₁ = 2/1
    1.0,  # b₂ = 2/2
    0.6666666667,  # b₃ = 2/3
    0.5,  # b₄ = 2/4
    0.4,  # b₅ = 2/5
    0.3333333333,  # b₆ = 2/6
    0.2857142857,  # b₇ = 2/7
    0.25,  # b₈ = 2/8
    0.2222222222,  # b₉ = 2/9
    0.2  # b₁₀ = 2/10
]

print("Коэффициенты (только ненулевые):")
for n in range(0, 11, 2):
    if n == 0:
        print(f"  a_{n} = {a_coeffs_10_manual[n]:.10f}  (постоянная)")
    else:
        print(f"  a_{n} = {a_coeffs_10_manual[n]:.10f}  (нечётное)")

print("\nФормула:")
formula_cos_10 = f"f(x) ≈ π/2 + {a_coeffs_10_manual[1]:.6f}·cos(x)"
for n in range(3, 11, 2):
    if a_coeffs_10_manual[n] > 1e-10:
        formula_cos_10 += f" + {a_coeffs_10_manual[n]:.6f}·cos({n}x)"
print(formula_cos_10)

print("\n" + "=" * 80)
print("СИНУС-РЯД (N=10):")
print("-" * 80)
print("Коэффициенты:")
for n in range(1, 11):
    print(f"  b_{n} = {b_coeffs_10_manual[n - 1]:.10f}  (2/{n})")

print("\nФормула:")
formula_sin_10 = "f(x) ≈ "
terms = [f"{b_coeffs_10_manual[n - 1]:.6f}·sin({n}x)" for n in range(1, 11)]
formula_sin_10 += " + ".join(terms)
print(formula_sin_10)


print("\n" + "=" * 80)
print("ВЫЧИСЛЕНИЕ РЯДОВ ДЛЯ N = 10, 50, 200 (по формулам интегралов)")
print("=" * 80 + "\n")

a_coeffs_dict = {}
b_coeffs_dict = {}
S_cos_dict = {}
S_sin_dict = {}

# Для всех N вычисляем по формулам
for N in N_values:
    print(f"✓ N={N} (вычисляется по формулам интегралов)...")
    a_coeffs_dict[N] = compute_cosine_coefficients(N)
    b_coeffs_dict[N] = compute_sine_coefficients(N)

    S_cos_dict[N] = fourier_cosine_partial(x, a_coeffs_dict[N])
    S_sin_dict[N] = fourier_sine_partial(x, b_coeffs_dict[N])

print("\nГотово!\n")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Цвета и стили для разных N
colors = {10: 'blue', 50: 'green', 200: 'purple'}
linestyles = {10: '--', 50: '-.', 200: ':'}
linewidths = {10: 2.5, 50: 2, 200: 2}


ax = axes[0]

# Исходная функция (КРАСНАЯ)
ax.plot(x, y, 'r', linewidth=3.5, label='f(x) = π - x (оригинал)', zorder=5)

# Косинус-ряды для разных N
for N in N_values:
    ax.plot(x, S_cos_dict[N], color=colors[N], linestyle=linestyles[N],
            linewidth=linewidths[N], label=f'N={N}', zorder=4)

ax.set_title('Косинусный ряд (частичные суммы)', fontsize=13, fontweight='bold')
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('f(x)', fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=11, framealpha=0.9)
ax.set_xlim(0, pi)
ax.set_ylim(-0.5, pi + 0.5)


ax = axes[1]

# Исходная функция (КРАСНАЯ)
ax.plot(x, y, 'r', linewidth=3.5, label='f(x) = π - x (оригинал)', zorder=5)

# Синус-ряды для разных N
for N in N_values:
    ax.plot(x, S_sin_dict[N], color=colors[N], linestyle=linestyles[N],
            linewidth=linewidths[N], label=f'N={N}', zorder=4)

ax.set_title('Синусный ряд (частичные суммы)', fontsize=13, fontweight='bold')
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('f(x)', fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=11, framealpha=0.9)
ax.set_xlim(0, pi)
ax.set_ylim(-0.5, pi + 0.5)

plt.tight_layout()
plt.show()

print("\n" + "=" * 80)
print("АНАЛИЗ ТОЧНОСТИ РЯДОВ")
print("=" * 80 + "\n")

print(f"{'N':<5} {'Макс. ошибка косинус-ряда':<30} {'Макс. ошибка синус-ряда':<30}")
print("-" * 80)

for N in N_values:
    cos_error = np.max(np.abs(S_cos_dict[N] - y))
    sin_error = np.max(np.abs(S_sin_dict[N] - y))

    print(f"{N:<5} {cos_error:<30.10f} {sin_error:<30.10f}")

print()

print("\n" + "=" * 80)
print("ПРОВЕРКА N=10: ВРУЧНУЮ vs ПО ФОРМУЛЕ")
print("=" * 80 + "\n")

S_cos_10_manual = fourier_cosine_partial(x, a_coeffs_10_manual)
S_sin_10_manual = fourier_sine_partial(x, b_coeffs_10_manual)

S_cos_10_formula = S_cos_dict[10]
S_sin_10_formula = S_sin_dict[10]

cos_diff = np.max(np.abs(S_cos_10_manual - S_cos_10_formula))
sin_diff = np.max(np.abs(S_sin_10_manual - S_sin_10_formula))

print(f"Макс. разница между вручную и формулой (косинус): {cos_diff:.2e}")
print(f"Макс. разница между вручную и формулой (синус):   {sin_diff:.2e}")
print("\n✓ Значения совпадают!" if cos_diff < 1e-10 and sin_diff < 1e-10 else "\n⚠ Есть отличия")
