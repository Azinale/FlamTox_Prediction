# mfea_optimizer.py
# -----------------------------------------------------------------------------
# Thay thế MFEA-II bằng GA thuần numpy cho chọn đặc trưng đa nhiệm
# Giữ nguyên class tên: MFEAIIOptimizer để main.py không cần sửa
# -----------------------------------------------------------------------------

from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple
import copy

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from config import DataConfig, AlgorithmConfig
from data_processor import DataProcessor


# ----------------------------- Nhóm tiện ích "DEAP-like" nhỏ gọn -----------------------------

class Fitness:
    """Đối tượng fitness tối giản, tương thích cách dùng trong main.py"""
    def __init__(self):
        self.values: Tuple[float, ...] | None = None  # vd: (mse,)
    @property
    def valid(self) -> bool:
        return self.values is not None


class Individual(list):
    """
    Cá thể là list[int] các bit 0/1.
    Có thêm các thuộc tính để tương thích main.py: fitness, scalar_fitness, factorial_rank, skill_factor
    """
    def __init__(self, genes: List[int], task_id: int):
        super().__init__(genes)
        self.fitness = Fitness()
        self.scalar_fitness: float = float("-inf")
        self.factorial_rank: float = float("inf")
        self.skill_factor: int = task_id  # gắn theo task khi khởi tạo


# ----------------------------- GA helpers -----------------------------

def repair_bits(ind: Individual, min_f: int, max_f: int) -> None:
    """Đảm bảo số lượng bit 1 nằm trong [min_f, max_f]. Sửa tại chỗ."""
    ones_idx = [i for i, b in enumerate(ind) if b == 1]
    zeros_idx = [i for i, b in enumerate(ind) if b == 0]
    cnt = len(ones_idx)

    if cnt < min_f:
        need = min_f - cnt
        if zeros_idx:
            flip = random.sample(zeros_idx, min(need, len(zeros_idx)))
            for i in flip:
                ind[i] = 1
    elif cnt > max_f:
        need = cnt - max_f
        if ones_idx:
            flip = random.sample(ones_idx, min(need, len(ones_idx)))
            for i in flip:
                ind[i] = 0


def init_individual(n_features: int, min_f: int, max_f: int, task_id: int) -> Individual:
    target = random.randint(min_f, max_f)
    indices = list(range(n_features))
    random.shuffle(indices)
    genes = [0] * n_features
    for i in indices[:target]:
        genes[i] = 1
    return Individual(genes, task_id)


def evaluate_individual(
    ind: Individual,
    task_id: int,
    data: Dict[str, Any],
    min_f: int,
    max_f: int,
) -> Tuple[float]:
    """Fitness = MSE(LinearRegression) + penalty số feature; nhỏ hơn là tốt."""
    selected = [i for i, b in enumerate(ind) if b == 1]
    if not (min_f <= len(selected) <= max_f) or len(selected) == 0:
        return (float("inf"),)

    if task_id == 0:
        X_train, X_test = data["X_tox_train"], data["X_tox_test"]
        y_train, y_test = data["y_tox_train"], data["y_tox_test"]
    else:
        X_train, X_test = data["X_flam_train"], data["X_flam_test"]
        y_train, y_test = data["y_flam_train"], data["y_flam_test"]

    Xtr, Xte = X_train[:, selected], X_test[:, selected]
    try:
        model = LinearRegression()
        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)
        if np.isnan(y_pred).any() or np.isinf(y_pred).any():
            return (float("inf"),)
        mse = mean_squared_error(y_test, y_pred)

        # Penalty khuyến khích ít feature (scale theo [min_f, max_f])
        rng = max(1, max_f - min_f)
        penalty = 0.05 * (len(selected) - min_f) / rng

        return (mse + penalty,)
    except Exception:
        return (float("inf"),)


def tournament_select(pop: List[Individual], k: int, tourn_size: int) -> List[Individual]:
    """Chọn lọc tournament dựa trên fitness.values[0] (nhỏ hơn tốt hơn)."""
    chosen: List[Individual] = []
    n = len(pop)
    if n == 0:
        return chosen
    for _ in range(k):
        size = min(tourn_size, n)
        aspirants = random.sample(pop, size)
        best = min(aspirants, key=lambda ind: ind.fitness.values[0] if ind.fitness.valid else float("inf"))
        chosen.append(best)
    return chosen


def two_point_crossover(p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
    """Lai ghép hai điểm; trả về bản sao con."""
    n = len(p1)
    if n < 2:
        return copy.deepcopy(p1), copy.deepcopy(p2)
    i, j = sorted(random.sample(range(n), 2))
    c1 = Individual(list(p1), p1.skill_factor)
    c2 = Individual(list(p2), p2.skill_factor)
    c1[i:j], c2[i:j] = c2[i:j], c1[i:j]
    return c1, c2


def mutate(ind: Individual, indpb: float, min_f: int, max_f: int) -> None:
    """Đột biến lật bit với xác suất indpb mỗi gen + repair."""
    for i in range(len(ind)):
        if random.random() < indpb:
            ind[i] = 1 - ind[i]
    repair_bits(ind, min_f, max_f)


def assign_ranks_and_scalar_fitness(pop: List[Individual]) -> None:
    """
    Gán factorial_rank và scalar_fitness để tương thích main.py:
    - factorial_rank: xếp hạng theo fitness tăng dần (1 tốt nhất).
    - scalar_fitness: nghịch đảo rank chuẩn hóa (lớn hơn tốt hơn).
    """
    valid_inds = [ind for ind in pop if ind.fitness.valid]
    if not valid_inds:
        return
    # Sắp xếp theo fitness (nhỏ hơn tốt hơn)
    valid_inds_sorted = sorted(valid_inds, key=lambda x: x.fitness.values[0])
    # Gán rank 1..m
    for rank, ind in enumerate(valid_inds_sorted, start=1):
        ind.factorial_rank = float(rank)
    max_rank = float(len(valid_inds_sorted))
    for ind in valid_inds_sorted:
        ind.scalar_fitness = 1.0 / (ind.factorial_rank / max_rank)  # 1 / rank_normalized


# ----------------------------- GA lõi cho một task -----------------------------

def run_ga_for_task(
    task_id: int,
    data: Dict[str, Any],
    n_features: int,
    algo: AlgorithmConfig,
) -> Individual:
    """
    Chạy GA thuần cho một task, trả về best individual (đã có fitness/thuộc tính tương thích).
    """
    pop_size = algo.pop_size
    ngen = algo.num_generations
    cxpb = algo.crossover_rate
    mutpb = algo.mutation_rate
    tourn = algo.tournament_size
    min_f = algo.min_features
    max_f = min(algo.max_features, n_features)  # đảm bảo không vượt số features thực tế

    # --- Khởi tạo quần thể ---
    population: List[Individual] = [init_individual(n_features, min_f, max_f, task_id) for _ in range(pop_size)]

    # Đánh giá ban đầu
    for ind in population:
        ind.fitness.values = evaluate_individual(ind, task_id, data, min_f, max_f)

    # Hall-of-fame (best-so-far)
    best = min(population, key=lambda x: x.fitness.values[0] if x.fitness.valid else float("inf"))
    best = copy.deepcopy(best)

    # Vòng lặp tiến hóa
    for _gen in range(ngen):
        # --- Selection ---
        parents = tournament_select(population, pop_size, tourn)

        # --- Crossover + Mutation để tạo offspring ---
        offspring: List[Individual] = []
        # ghép cặp liên tiếp
        for i in range(0, pop_size, 2):
            p1 = parents[i]
            p2 = parents[i + 1] if i + 1 < pop_size else parents[0]
            if random.random() < cxpb:
                c1, c2 = two_point_crossover(p1, p2)
            else:
                c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)
            # Đột biến
            mutate(c1, mutpb, min_f, max_f)
            mutate(c2, mutpb, min_f, max_f)
            offspring.append(c1)
            offspring.append(c2)

        # Nếu lẻ, cắt bớt
        offspring = offspring[:pop_size]

        # Đánh giá offspring
        for ind in offspring:
            ind.fitness.values = evaluate_individual(ind, task_id, data, min_f, max_f)

        # Elitism: giữ lại best-so-far vào thế hệ mới
        # thay thế một cá thể tệ nhất bằng bản sao best
        worst_idx = np.argmax([ind.fitness.values[0] if ind.fitness.valid else float("inf") for ind in offspring])
        offspring[worst_idx] = copy.deepcopy(best)

        # Cập nhật best
        cur_best = min(offspring, key=lambda x: x.fitness.values[0] if x.fitness.valid else float("inf"))
        if (cur_best.fitness.valid and
            (not best.fitness.valid or cur_best.fitness.values[0] < best.fitness.values[0])):
            best = copy.deepcopy(cur_best)

        # Thế hệ mới
        population = offspring
        if _gen % 1 == 0 or _gen == ngen:
            num_feat = sum(best)
            print(f"--- Thế hệ {_gen}/{ngen} ---")
            if task_id == 0:
                print(f"  Best Toxicity: MSE={best.fitness.values[0]:.4f}, "
                    f"NumFeat={num_feat}, ScalarFit={best.scalar_fitness:.4f}, "
                    f"Rank={best.factorial_rank:.1f}")
            else:
                print(f"  Best Flammability: MSE={best.fitness.values[0]:.4f}, "
                    f"NumFeat={num_feat}, ScalarFit={best.scalar_fitness:.4f}, "
                    f"Rank={best.factorial_rank:.1f}")

    # Gán rank & scalar_fitness cho toàn bộ quần thể, sau đó đảm bảo best có thuộc tính này
    assign_ranks_and_scalar_fitness(population + [best])

    # best.skill_factor đã gán theo task; chắc chắn có fitness.valid
    return best


# ----------------------------- Optimizer chính (giữ nguyên tên/class) -----------------------------

class GAOptimizer:
    """Bản thay thế: chạy GA thuần cho từng task, giữ nguyên interface đầu ra giống MFEA-II cũ."""

    def __init__(self, data_config: DataConfig, algo_config: AlgorithmConfig):
        self.data_config = data_config
        self.algo_config = algo_config
        self.data_processor = DataProcessor(data_config)
        self.processed_data: Dict[str, Any] | None = None
        self.n_features: int | None = None
        self.feature_names: List[str] | None = None

    def run(self) -> Dict[str, Any]:
        print("--- Bắt đầu GA (thay cho MFEA-II) ---")

        # 1) Xử lý dữ liệu
        self.processed_data = self.data_processor.process_data()
        self.n_features = int(self.processed_data["n_features"])
        self.feature_names = list(self.processed_data["feature_names"])

        # 2) Chạy GA riêng cho từng task (0..n_tasks-1)
        n_tasks = max(1, self.algo_config.n_tasks)
        best_individuals: Dict[int, Individual] = {}

        for task_id in range(n_tasks):
            print(f"\n>>> Chạy GA cho task {task_id} ...")
            best = run_ga_for_task(
                task_id=task_id,
                data=self.processed_data,
                n_features=self.n_features,
                algo=self.algo_config,
            )
            # đảm bảo có .scalar_fitness/.factorial_rank (đã được gán)
            if not np.isfinite(best.scalar_fitness):
                best.scalar_fitness = 1.0
            if not np.isfinite(best.factorial_rank):
                best.factorial_rank = 1.0
            best_individuals[task_id] = best
            print(f"    Done. Best fitness (MSE+penalty): {best.fitness.values[0]:.6f}")

        print("\n--- Hoàn thành GA ---")

        # 3) Trả về interface tương thích main.py
        return {
            "best_individuals": best_individuals,           # dict task_id -> Individual
            "best_fitness": {k: v.fitness.values[0] for k, v in best_individuals.items()},
            "data": self.processed_data,                    # để main.py đánh giá lại RF
            "feature_names": self.feature_names,            # tên features chung
        }
