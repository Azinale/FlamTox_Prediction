# config.py
from dataclasses import dataclass

@dataclass
class RDKitConfig:
    """Cấu hình cho xử lý RDKit"""
    # File input chưa được description
    tox_path: str = "Data/pIGC50.csv"
    flam_path: str = "Data/lflt.csv"
    
    # File output sau khi được description
    tox_descriptor_path: str = "Data/pIGC50_descriptor.csv"
    flam_descriptor_path: str = "Data/lflt_descriptor.csv"

@dataclass
class DataConfig:
    """Cấu hình dữ liệu cho MFEA-II"""
    # File input đã được description
    tox_path: str = "Data/pIGC50_descriptor.csv"
    flam_path: str = "Data/lflt_descriptor.csv"
    
    # Cấu hình xử lý dữ liệu
    test_size: float = 0.2
    random_state: int = 42 # Seed for reproducibility
    n_std: int = 3 # For outlier clipping

@dataclass
class AlgorithmConfig:
    """Cấu hình thuật toán MFEA-II"""
    pop_size: int = 100       # Kích thước quần thể (nên là bội số của số tasks)
    num_generations: int = 200 # Số thế hệ
    mutation_rate: float = 0.1 # Tỷ lệ đột biến gen
    crossover_rate: float = 0.9 # Tỷ lệ lai ghép
    rmp: float = 0.3           # Xác suất lai ghép giữa các tasks (random mating probability)
    tournament_size: int = 3   # Kích thước tournament cho selection
    n_tasks: int = 2           # Số lượng bài toán (toxicity, flammability)
    min_features: int = 5      # Số lượng features tối thiểu
    max_features: int = 150    # Số lượng features tối đa