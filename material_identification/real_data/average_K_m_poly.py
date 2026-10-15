import numpy as np


COEFFS_K = [
    264.455613254,
    -177.37322503,
    27.0304578037,
]

COEFFS_M = [
    0.41365193633,
    -0.263045279559,
    0.128785905521,
]


def poly_value(coeffs, x):
    """
    Значение полинома.

    coeffs = [a0, a1, a2, ...]
    P(x) = a0 + a1*x + a2*x^2 + ...
    """
    return sum(c * x**i for i, c in enumerate(coeffs))


def average_poly_value(coeffs, x_min, x_max):
    """
    Среднее значение полинома на отрезке [x_min, x_max].
    """
    if x_min == x_max:
        return poly_value(coeffs, x_min)

    integral = 0.0

    for i, c in enumerate(coeffs):
        integral += c / (i + 1) * (x_max**(i + 1) - x_min**(i + 1))

    return integral / (x_max - x_min)


# Диапазон, например по деформации epsilon
eps_min = 0.1
eps_max = 0.86

K_avg = average_poly_value(COEFFS_K, eps_min, eps_max)
m_avg = average_poly_value(COEFFS_M, eps_min, eps_max)

print(f"Среднее K на [{eps_min}, {eps_max}] = {K_avg}")
print(f"Среднее m на [{eps_min}, {eps_max}] = {m_avg}")