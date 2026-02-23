# src/model.py

import numpy as np
from src.fuzzy import basis_vector


def generate_grid(n: int = 21) -> list[tuple[float, float]]:
    # Регулярная сетка n x n на [-1, 1] x [-1, 1]
    pts = np.linspace(-1, 1, n)
    return [(x1, x2) for x1 in pts for x2 in pts]


def compute_info_matrix(points: list[tuple[float, float]], delta: float) -> np.ndarray:
    # Информационная матрица M = (1/N) * sum( f(xi) * f(xi)^T )
    # points — список точек плана (не сетка, а уже отобранные точки)
    n = len(points)
    dim = 4
    M = np.zeros((dim, dim))
    for x1, x2 in points:
        fv = basis_vector(x1, x2, delta)
        M += np.outer(fv, fv)
    return M / n