# src/model.py

import numpy as np
import matplotlib.pyplot as plt

from src.fuzzy import regressor_vector


def generate_grid(n: int = 21) -> list[tuple[float, float]]:
    """Regular n x n grid on [-1, 1] x [-1, 1]."""
    pts = np.linspace(-1, 1, n)
    return [(float(x1), float(x2)) for x1 in pts for x2 in pts]


def compute_info_matrix(points: list[tuple[float, float]], delta: float) -> np.ndarray:
    """Information matrix M = sum f(x) f(x)^T for selected points."""
    dim = 12
    M = np.zeros((dim, dim), dtype=float)
    for x1, x2 in points:
        f = regressor_vector(x1, x2, delta)
        M += np.outer(f, f)
    return M


def a_criterion(M: np.ndarray, num_points: int, gamma: float = 1e-8) -> float:
    """A-optimality criterion in the style of lab #3.

    If s < m, use M + gamma*I.
    If s >= m, use plain M.
    """
    dim = M.shape[0]
    A = M + gamma * np.eye(dim) if num_points < dim else M
    try:
        invA = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        invA = np.linalg.pinv(A)
    return float(np.trace(invA))


def synthesize_a_optimal_plan(
    delta: float,
    grid: list[tuple[float, float]],
    max_points: int = 60,
    gamma: float = 1e-8,
    log_every: int = 0,
) -> tuple[list[tuple[float, float]], np.ndarray]:
    """Sequential A-optimal plan construction.

    Multiplicities are allowed in a discrete plan.
    """
    num_candidates = len(grid)
    dim = 12

    FFt = np.zeros((num_candidates, dim, dim), dtype=float)
    for i, (x1, x2) in enumerate(grid):
        f = regressor_vector(x1, x2, delta)
        FFt[i, :, :] = np.outer(f, f)

    selected_indices: list[int] = []
    selected_once = np.zeros(num_candidates, dtype=bool)
    M = np.zeros((dim, dim), dtype=float)

    for step in range(max_points):
        best_idx = -1
        best_value = np.inf
        rank_M = np.linalg.matrix_rank(M)

        for idx in range(num_candidates):
            # As in lab #3: no repeats before first m points.
            if step < dim and selected_once[idx]:
                continue
            Mcand = M + FFt[idx]
            # As in lab #3: enforce rank growth on early steps.
            if step < dim and np.linalg.matrix_rank(Mcand) <= rank_M:
                continue
            value = a_criterion(Mcand, num_points=step + 1, gamma=gamma)
            if value < best_value:
                best_value = value
                best_idx = idx

        if best_idx < 0:
            break

        selected_indices.append(best_idx)
        selected_once[best_idx] = True
        M += FFt[best_idx]

        if log_every > 0 and (step == 0 or (step + 1) % log_every == 0 or (step + 1) == max_points):
            cur_val = a_criterion(M, num_points=step + 1, gamma=gamma)
            uniq = len(set(selected_indices))
            print(f"{step + 1:3d}\t{cur_val: .6e}\t{uniq:3d}")

    plan_points = [grid[i] for i in selected_indices]
    return plan_points, M


def build_spectrum(
    points: list[tuple[float, float]],
) -> list[tuple[float, float, int, float]]:
    """Build plan spectrum: unique points, counts r_i and weights p_i."""
    if not points:
        return []

    counts: dict[tuple[float, float], int] = {}
    for p in points:
        counts[p] = counts.get(p, 0) + 1

    total = len(points)
    spectrum = [(x1, x2, r, r / total) for (x1, x2), r in counts.items()]
    spectrum.sort(key=lambda row: (-row[2], row[0], row[1]))
    return spectrum


def analyze_matrix(M: np.ndarray, num_points: int, gamma: float = 1e-8) -> dict:
    """Compute matrix metrics."""
    dim = M.shape[0]
    A = M + gamma * np.eye(dim) if num_points < dim else M
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
    """Plot plan points on [-1, 1] x [-1, 1]."""
    if not points:
        return
    spectrum = build_spectrum(points)
    xs = [row[0] for row in spectrum]
    ys = [row[1] for row in spectrum]
    rs = np.array([row[2] for row in spectrum], dtype=float)
    sizes = 35 + 15 * (rs - 1)

    plt.figure(figsize=(5, 5))
    plt.scatter(xs, ys, c="tab:blue", s=sizes, edgecolors="k", alpha=0.85)
    plt.xlim(-1.05, 1.05)
    plt.ylim(-1.05, 1.05)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(f"A-optimal plan, delta = {delta} (size = multiplicity)")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        filename = f"plan_delta_{delta:.2f}.png"
        plt.savefig(filename, dpi=200)
        plt.close()


def print_plan_table(points: list[tuple[float, float]]) -> None:
    """Print full point list (with repeats)."""
    print("No\tx1\t\tx2")
    for i, (x1, x2) in enumerate(points, start=1):
        print(f"{i:2d}\t{x1: .3f}\t{x2: .3f}")


def print_spectrum_table(points: list[tuple[float, float]]) -> None:
    """Print spectrum: unique points, counts and weights."""
    spectrum = build_spectrum(points)
    print("No\tx1\t\tx2\tr_i\tp_i")
    for i, (x1, x2, r, p) in enumerate(spectrum, start=1):
        print(f"{i:2d}\t{x1: .3f}\t{x2: .3f}\t{r:2d}\t{p:.4f}")


def run_synthesis_for_deltas(
    deltas: list[float] | tuple[float, ...] = (0.5, 0.4, 0.3, 0.2),
    max_points: int = 60,
    gamma: float = 1e-8,
    log_every: int = 0,
) -> None:
    """Run plan synthesis for selected deltas."""
    grid = generate_grid(n=21)
    summary_rows: list[tuple[float, int, float, float, int, float]] = []

    for delta in deltas:
        print(f"\n===== delta = {delta} =====")
        if log_every > 0:
            print("Iter\ttrace(A^-1)\tuniq")
        plan_points, M = synthesize_a_optimal_plan(
            delta,
            grid,
            max_points=max_points,
            gamma=gamma,
            log_every=log_every,
        )
        stats = analyze_matrix(M, num_points=len(plan_points), gamma=gamma)
        uniq = len(build_spectrum(plan_points))
        summary_rows.append((delta, uniq, stats["trace_inv"], stats["det"], stats["rank"], stats["cond"]))

        print("\nPlan points (with repeats):")
        print_plan_table(plan_points)
        print("\nSpectrum (unique points, multiplicities, weights):")
        print_spectrum_table(plan_points)

        print("\nInfo matrix M metrics:")
        print(f"trace(M^(-1)) ~= {stats['trace_inv']:.6e}")
        print(f"det(M)        ~= {stats['det']:.6e}")
        print(f"rank(M)       = {stats['rank']}")
        print(f"cond(M)       ~= {stats['cond']:.6e}")

        plot_plan(plan_points, delta, save=True)

    print("\n===== Summary by delta =====")
    print("delta\tuniq\ttrace(M^-1)\t\tdet(M)\t\trank\tcond(M)")
    for delta, uniq, trace_inv, det, rank, cond in summary_rows:
        print(f"{delta:.1f}\t{uniq:2d}\t{trace_inv: .6e}\t{det: .6e}\t{rank:2d}\t{cond: .6e}")


if __name__ == "__main__":
    run_synthesis_for_deltas(log_every=10)
