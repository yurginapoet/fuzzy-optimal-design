# src/model.py

import numpy as np
import matplotlib.pyplot as plt

from src.fuzzy import regressor_vector


def generate_grid(n: int = 21) -> list[tuple[float, float]]:
    """Регулярная сетка n×n на [-1, 1] × [-1, 1]."""
    pts = np.linspace(-1, 1, n)
    return [(float(x1), float(x2)) for x1 in pts for x2 in pts]


def compute_info_matrix(points: list[tuple[float, float]], delta: float) -> np.ndarray:
    """Информационная матрица M = Σ f(x) f(x)^T для выбранного плана."""
    dim = 12
    M = np.zeros((dim, dim), dtype=float)
    for x1, x2 in points:
        f = regressor_vector(x1, x2, delta)
        M += np.outer(f, f)
    return M


def a_criterion(M: np.ndarray, reg: float = 1e-8) -> float:
    """Критерий A‑оптимальности: trace( (M + reg I)^{-1} )."""
    dim = M.shape[0]
    A = M + reg * np.eye(dim)
    try:
        invA = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        invA = np.linalg.pinv(A)
    return float(np.trace(invA))


def synthesize_a_optimal_plan(
    delta: float,
    grid: list[tuple[float, float]],
    max_points: int = 60,
    reg: float = 1e-8,
) -> tuple[list[tuple[float, float]], np.ndarray]:
    """Последовательный алгоритм достраивания A‑оптимального плана."""
    num_candidates = len(grid)
    dim = 12

    # Предварительно считаем регрессоры для всех узлов сетки
    F = np.zeros((num_candidates, dim), dtype=float)
    for i, (x1, x2) in enumerate(grid):
        F[i, :] = regressor_vector(x1, x2, delta)

    selected_mask = np.zeros(num_candidates, dtype=bool)
    selected_indices: list[int] = []
    M = np.zeros((dim, dim), dtype=float)

    for _ in range(max_points):
        best_idx = -1
        best_value = np.inf

        for idx in range(num_candidates):
            if selected_mask[idx]:
                continue
            f = F[idx]
            Mcand = M + np.outer(f, f)
            value = a_criterion(Mcand, reg=reg)
            if value < best_value:
                best_value = value
                best_idx = idx

        if best_idx < 0:
            break

        selected_mask[best_idx] = True
        selected_indices.append(best_idx)
        f_best = F[best_idx]
        M += np.outer(f_best, f_best)

    plan_points = [grid[i] for i in selected_indices]
    return plan_points, M


def analyze_matrix(M: np.ndarray, reg: float = 1e-8) -> dict:
    """Вычисление характеристик информационной матрицы."""
    dim = M.shape[0]
    A = M + reg * np.eye(dim)
    try:
        invA = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        invA = np.linalg.pinv(A)

    trace_inv = float(np.trace(invA))
    det = float(np.linalg.det(M))
    rank = int(np.linalg.matrix_rank(M))
    try:
        cond = float(np.linalg.cond(M))
    except np.linalg.LinAlgError:
        cond = float("inf")

    return {
        "trace_inv": trace_inv,
        "det": det,
        "rank": rank,
        "cond": cond,
    }


def plot_plan(points: list[tuple[float, float]], delta: float, save: bool = True) -> None:
    """График расположения точек плана на квадрате [-1, 1] × [-1, 1].

    По умолчанию сохраняет рисунок в файл и не блокирует выполнение программы.
    """
    if not points:
        return
    xs, ys = zip(*points)
    plt.figure(figsize=(5, 5))
    plt.scatter(xs, ys, c="tab:blue", s=30, edgecolors="k")
    plt.xlim(-1.05, 1.05)
    plt.ylim(-1.05, 1.05)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(f"A-оптимальный план, δ = {delta}")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        filename = f"plan_delta_{delta:.2f}.png"
        plt.savefig(filename, dpi=200)
        plt.close()


def print_plan_table(points: list[tuple[float, float]]) -> None:
    """Простой табличный вывод координат точек плана."""
    print("№\tx1\t\tx2")
    for i, (x1, x2) in enumerate(points, start=1):
        print(f"{i:2d}\t{x1: .3f}\t{x2: .3f}")


def run_synthesis_for_deltas(
    deltas: list[float] | tuple[float, ...] = (0.5, 0.4, 0.3, 0.2),
    max_points: int = 60,
) -> None:
    """Запуск синтеза планов для набора значений δ и вывод результатов.

    Для каждого δ формируется таблица координат, численные характеристики
    информационной матрицы и сохраняется рисунок с расположением точек плана.
    """
    grid = generate_grid(n=21)

    for delta in deltas:
        print(f"\n===== δ = {delta} =====")
        plan_points, M = synthesize_a_optimal_plan(delta, grid, max_points=max_points)
        stats = analyze_matrix(M)

        print("\nКоординаты точек плана:")
        print_plan_table(plan_points)

        print("\nХарактеристики информационной матрицы M:")
        print(f"trace(M^(-1)) ≈ {stats['trace_inv']:.6e}")
        print(f"det(M)        ≈ {stats['det']:.6e}")
        print(f"rank(M)       = {stats['rank']}")
        print(f"cond(M)       ≈ {stats['cond']:.6e}")

        plot_plan(plan_points, delta, save=True)


if __name__ == "__main__":
    run_synthesis_for_deltas()