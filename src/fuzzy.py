# src/fuzzy.py

import numpy as np


def mu1(x: float, delta: float) -> float:
    """Левая трапециевидная функция принадлежности."""
    if x <= -delta:
        return 1.0
    if x >= delta:
        return 0.0
    return (delta - x) / (2 * delta)


def mu2(x: float, delta: float) -> float:
    """Правая трапециевидная функция принадлежности."""
    return 1.0 - mu1(x, delta)


def fuzzy_weights(x1: float, x2: float, delta: float) -> tuple[float, float, float, float]:
    """Весовые коэффициенты w_ij для пары (x1, x2)."""
    m1 = mu1(x1, delta)
    m2 = mu2(x1, delta)
    n1 = mu1(x2, delta)
    n2 = mu2(x2, delta)

    w11 = m1 * n1
    w12 = m1 * n2
    w21 = m2 * n1
    w22 = m2 * n2
    return w11, w12, w21, w22


def regressor_vector(x1: float, x2: float, delta: float) -> np.ndarray:
    """Вектор регрессоров размерности 12 для двухфакторной нечеткой линейной модели."""
    w11, w12, w21, w22 = fuzzy_weights(x1, x2, delta)
    return np.array(
        [
            w11,
            w11 * x1,
            w11 * x2,
            w12,
            w12 * x1,
            w12 * x2,
            w21,
            w21 * x1,
            w21 * x2,
            w22,
            w22 * x1,
            w22 * x2,
        ],
        dtype=float,
    )


def basis_vector(x1: float, x2: float, delta: float) -> np.ndarray:
    """Совместимость с прежним кодом: теперь это 12‑мерный вектор регрессоров."""
    return regressor_vector(x1, x2, delta)