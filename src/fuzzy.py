# src/fuzzy.py

import numpy as np


def mu1(x: float, delta: float) -> float:
    # Левая трапециевидная функция принадлежности
    if x <= -delta:
        return 1.0
    elif x >= delta:
        return 0.0
    else:
        return (delta - x) / (2 * delta)


def mu2(x: float, delta: float) -> float:
    # Правая трапециевидная функция принадлежности
    return 1.0 - mu1(x, delta)


def basis_vector(x1: float, x2: float, delta: float) -> np.ndarray:
    # Вектор базисных функций для двухфакторной модели
    # Размерность: 4 = 2 партиции x1 * 2 партиции x2
    m1 = mu1(x1, delta)
    m2 = mu2(x1, delta)
    n1 = mu1(x2, delta)
    n2 = mu2(x2, delta)
    return np.array([m1 * n1, m1 * n2, m2 * n1, m2 * n2])