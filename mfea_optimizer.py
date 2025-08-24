# mfea_optimizer.py
import numpy as np
import random
from deap import base, creator, tools, algorithms
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from typing import List, Tuple, Dict, Any

# Import configurations and DataProcessor
from config import DataConfig, AlgorithmConfig
from data_processor import DataProcessor

# --- DEAP Setup ---
# Xóa các creator cũ nếu có để tránh lỗi khi chạy lại trong cùng session
if 'FitnessMin' in creator.__dict__:
    del creator.FitnessMin
if 'Individual' in creator.__dict__:
    del creator.Individual

# Tạo kiểu Fitness và Individual trong DEAP
# FitnessMin: Mục tiêu là giảm thiểu MSE, nên trọng số là -1.0
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
# Individual: Là một list (gen 0/1), có thêm các thuộc tính MFEA-II
creator.create("Individual", list, fitness=creator.FitnessMin,
               skill_factor=int,       # Task ID (0 or 1)
               factorial_rank=float,   # Rank within its task
               scalar_fitness=float)   # Transformed fitness for selection

# --- MFEA-II Optimizer Class ---

class MFEAIIOptimizer:
    """Triển khai thuật toán MFEA-II để tối ưu hóa việc chọn features đa nhiệm"""
    def __init__(self, data_config: DataConfig, algo_config: AlgorithmConfig):
        self.data_config = data_config
        self.algo_config = algo_config
        self.data_processor = DataProcessor(data_config)
        self.processed_data: Dict[str, Any] | None = None
        self.n_features: int | None = None
        self.feature_names: List[str] | None = None
        self.toolbox = base.Toolbox()

    def _setup_deap(self) -> None:
        """Thiết lập các toán tử và cấu trúc của DEAP"""
        if self.n_features is None:
             raise ValueError("n_features chưa được thiết lập. Cần chạy process_data trước.")

        print("Thiết lập DEAP...")
        self.toolbox.register("attr_bool", random.randint, 0, 1) # Gen là 0 hoặc 1

        # Hàm tạo cá thể cơ bản (chưa có skill factor)
        self.toolbox.register("individual_no_sf", tools.initRepeat, list,
                            self.toolbox.attr_bool, n=self.n_features)

        # Hàm khởi tạo cá thể hoàn chỉnh (có skill factor và đảm bảo số features)
        def init_individual(ind_class, base_ind_func):
            ind = ind_class(base_ind_func()) # Tạo list gen [0,1,...]
            ind.skill_factor = random.randint(0, self.algo_config.n_tasks - 1) # Gán task ngẫu nhiên
            ind.factorial_rank = float('inf') # Khởi tạo rank
            ind.scalar_fitness = float('-inf') # Khởi tạo scalar fitness (selection dùng max)

            # Đảm bảo số features nằm trong khoảng [min, max] khi khởi tạo
            current_features = np.sum(ind)
            target_features = random.randint(self.algo_config.min_features, self.algo_config.max_features)

            # Điều chỉnh số features nếu cần
            indices = list(range(len(ind)))
            random.shuffle(indices) # Xáo trộn để chọn ngẫu nhiên
            
            features_to_add = target_features - current_features
            features_to_remove = current_features - target_features

            if features_to_add > 0: # Cần thêm features
                count = 0
                for i in indices:
                    if ind[i] == 0:
                        ind[i] = 1
                        count += 1
                        if count >= features_to_add:
                            break
            elif features_to_remove > 0: # Cần bớt features
                count = 0
                for i in indices:
                    if ind[i] == 1:
                        ind[i] = 0
                        count += 1
                        if count >= features_to_remove:
                            break
                            
            # Kiểm tra lại lần cuối (phòng trường hợp min/max quá gần nhau hoặc không thể đạt target)
            final_features = np.sum(ind)
            if final_features < self.algo_config.min_features:
                zero_indices = [i for i, bit in enumerate(ind) if bit == 0]
                num_to_add = min(len(zero_indices), self.algo_config.min_features - final_features)
                indices_to_flip = random.sample(zero_indices, num_to_add)
                for i in indices_to_flip: ind[i] = 1
            elif final_features > self.algo_config.max_features:
                 one_indices = [i for i, bit in enumerate(ind) if bit == 1]
                 num_to_remove = min(len(one_indices), final_features - self.algo_config.max_features)
                 indices_to_flip = random.sample(one_indices, num_to_remove)
                 for i in indices_to_flip: ind[i] = 0

            return ind

        self.toolbox.register("individual", init_individual, creator.Individual,
                            self.toolbox.individual_no_sf)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Toán tử lai ghép (Mate) - có kiểm tra skill factor và rmp
        def mate_mfea(ind1, ind2):
            # Luôn clone để tránh thay đổi cá thể gốc
            child1 = self.toolbox.clone(ind1)
            child2 = self.toolbox.clone(ind2)

            perform_crossover = False
            # Lai cùng task hoặc lai khác task nếu rmp cho phép
            # The check `if random.random() < crossover_rate` is REMOVED here.
            # varAnd handles the probability check externally using cxpb.
            if ind1.skill_factor == ind2.skill_factor or random.random() < self.algo_config.rmp:
                 perform_crossover = True # Perform crossover if same task OR rmp allows different task

            if perform_crossover:
                # Dùng cxTwoPoint hoặc cxUniform, etc.
                tools.cxTwoPoint(child1, child2)
                # Gán skill factor cho con (assortative mating)
                child1.skill_factor = random.choice([ind1.skill_factor, ind2.skill_factor])
                child2.skill_factor = random.choice([ind1.skill_factor, ind2.skill_factor])
                # Đặt lại fitness vì con là mới
                del child1.fitness.values
                del child2.fitness.values
                child1.factorial_rank = float('inf')
                child1.scalar_fitness = float('-inf')
                child2.factorial_rank = float('inf')
                child2.scalar_fitness = float('-inf')

                # Đảm bảo số features của con nằm trong khoảng hợp lệ
                for child in [child1, child2]:
                    num_features = np.sum(child)
                    # Điều chỉnh nếu vi phạm ràng buộc min/max
                    indices = list(range(len(child)))
                    random.shuffle(indices)

                    features_to_add = self.algo_config.min_features - num_features
                    features_to_remove = num_features - self.algo_config.max_features

                    if features_to_add > 0: # Cần thêm
                        count = 0
                        for i in indices:
                           if child[i] == 0:
                               child[i] = 1
                               count += 1
                               if count >= features_to_add: break
                        del child.fitness.values # Fitness thay đổi
                    elif features_to_remove > 0: # Cần bớt
                        count = 0
                        for i in indices:
                           if child[i] == 1:
                               child[i] = 0
                               count += 1
                               if count >= features_to_remove: break
                        del child.fitness.values # Fitness thay đổi

            # Trả về con (có thể không thay đổi nếu không lai ghép)
            return child1, child2

        # Toán tử đột biến (Mutate) - đảm bảo số features
        def mutate_mfea(individual, mutation_rate):
            mutant = self.toolbox.clone(individual)
            mutated = False
            indices = list(range(len(mutant)))
            random.shuffle(indices) # Đột biến theo thứ tự ngẫu nhiên

            for i in indices:
                if random.random() < mutation_rate:
                    current_features = np.sum(mutant)
                    original_bit = mutant[i]
                    # Thử đảo bit
                    mutant[i] = 1 - mutant[i]
                    new_features = np.sum(mutant)

                    # Kiểm tra ràng buộc: Nếu vi phạm, hoàn tác đột biến tại vị trí này
                    if not (self.algo_config.min_features <= new_features <= self.algo_config.max_features):
                        mutant[i] = original_bit # Hoàn tác
                    elif original_bit != mutant[i]: # Nếu thực sự có thay đổi hợp lệ
                        mutated = True

            if mutated:
                del mutant.fitness.values # Đặt lại fitness nếu có đột biến
                mutant.factorial_rank = float('inf')
                mutant.scalar_fitness = float('-inf')
            return mutant, # DEAP mutation trả về tuple

        # Đăng ký các toán tử vào toolbox
        # Lưu ý: crossover_rate và mutation_rate được dùng trong hàm varAnd, không cần truyền ở đây
        self.toolbox.register("mate", mate_mfea) # Truyền crossover_rate=... là không cần thiết với varAnd
        self.toolbox.register("mutate", mutate_mfea, mutation_rate=self.algo_config.mutation_rate)
        # Đăng ký hàm selection (class method)
        self.toolbox.register("select", self.selTournamentMFEA)
        print("Đã thiết lập xong các toán tử DEAP.")


    def evaluate_task(self, individual: creator.Individual, task_id: int, data: Dict[str, Any]) -> Tuple[float]:
        """Đánh giá MSE của một cá thể trên một bài toán cụ thể"""
        selected_indices = [i for i, bit in enumerate(individual) if bit == 1]
        num_selected = len(selected_indices)

        # Kiểm tra ràng buộc số lượng features
        if not self.algo_config.min_features <= num_selected <= self.algo_config.max_features:
            return (float('inf'),) # Trả về fitness rất tệ (infinity)

        # Lấy dữ liệu tương ứng với task
        if task_id == 0: # Toxicity
            X_train = data['X_tox_train']
            X_test = data['X_tox_test']
            y_train = data['y_tox_train']
            y_test = data['y_tox_test']
        elif task_id == 1: # Flammability
            X_train = data['X_flam_train']
            X_test = data['X_flam_test']
            y_train = data['y_flam_train']
            y_test = data['y_flam_test']
        else:
            # Trường hợp không mong muốn, nhưng nên có để bắt lỗi
            return (float('inf'),)

        # Chọn các cột features tương ứng
        if num_selected == 0: # Trường hợp không chọn feature nào
             return (float('inf'),)

        X_train_selected = X_train[:, selected_indices]
        X_test_selected = X_test[:, selected_indices]

        try:
            # Sử dụng LinearRegression cho tốc độ trong vòng lặp GA
            model = LinearRegression()
            model.fit(X_train_selected, y_train)
            y_pred = model.predict(X_test_selected)
            # Kiểm tra và xử lý giá trị NaN hoặc Inf trong dự đoán
            if np.isnan(y_pred).any() or np.isinf(y_pred).any():
                 return (float('inf'),) # Fitness tệ nếu dự đoán không hợp lệ
            mse = mean_squared_error(y_test, y_pred)

            # Thêm một penalty nhỏ cho số lượng features (khuyến khích giải pháp gọn hơn)
            feature_penalty_coefficient = 0.05
            feature_range = self.algo_config.max_features - self.algo_config.min_features
            if feature_range > 0:
                 feature_penalty = feature_penalty_coefficient * \
                                   (num_selected - self.algo_config.min_features) / \
                                   feature_range
            else:
                 feature_penalty = 0 # Tránh chia cho 0

            # Tổng fitness là MSE + penalty
            final_fitness = min(mse + feature_penalty, float('inf')) # Clamp giá trị
            # Kiểm tra lại final_fitness
            if np.isnan(final_fitness) or np.isinf(final_fitness):
                 return (float('inf'),)

            return (final_fitness,) # Trả về tuple

        except Exception as e:
            # Ít log hơn để tránh làm chậm vòng lặp
            # print(f"Warning: Error evaluating task {task_id} with {num_selected} features: {e}")
            return (float('inf'),)


    def _setup_evaluation(self, data: Dict[str, Any]) -> None:
        """Thiết lập các hàm đánh giá cho DEAP toolbox"""
        def evaluate_individual(individual):
            # Đánh giá cá thể dựa trên skill_factor của nó
            return self.evaluate_task(individual, individual.skill_factor, data)

        # Đăng ký hàm đánh giá chính
        self.toolbox.register("evaluate", evaluate_individual)


    def calculate_ranks_fitness(self, population: List[creator.Individual]) -> None:
        """Tính toán factorial rank và scalar fitness cho từng cá thể trong quần thể"""
        # Reset ranks and scalar fitness
        for ind in population:
            ind.factorial_rank = float('inf')
            ind.scalar_fitness = float('-inf') # scalar_fitness lớn hơn là tốt hơn

        # Tính toán riêng cho từng task
        for k in range(self.algo_config.n_tasks):
            # Lọc các cá thể theo skill factor của task k và đã có fitness hợp lệ
            task_k_individuals = [ind for ind in population if ind.skill_factor == k and ind.fitness.valid]
            if not task_k_individuals:
                continue # Bỏ qua nếu không có cá thể nào cho task này

            # Sắp xếp các cá thể của task k theo fitness (MSE - nhỏ hơn là tốt hơn)
            task_k_individuals.sort(key=lambda ind: ind.fitness.values[0])

            # Tính toán factorial rank (rank càng nhỏ càng tốt)
            current_rank = 0.0
            last_fitness = float('-inf')
            rank_counter = 0
            for i, ind in enumerate(task_k_individuals):
                current_fitness = ind.fitness.values[0]
                if abs(current_fitness - last_fitness) > 1e-9: # Nếu fitness khác biệt đáng kể
                    current_rank += rank_counter + 1 # Tăng rank lên bằng số lượng cá thể + 1 so với rank trước đó
                    rank_counter = 0 # Reset bộ đếm
                else: # Fitness giống nhau
                    rank_counter += 1

                ind.factorial_rank = current_rank
                last_fitness = current_fitness


            # Tính toán scalar fitness (càng lớn càng tốt cho selection)
            # Lấy rank tệ nhất (lớn nhất) trong task này để chuẩn hóa
            max_rank_k = max((ind.factorial_rank for ind in task_k_individuals if ind.factorial_rank != float('inf')), default=1.0)
            if max_rank_k == 0: max_rank_k = 1.0 # Tránh chia cho 0

            for ind in task_k_individuals:
                 if ind.factorial_rank != float('inf'):
                     # Scalar fitness = 1 / rank_normalized
                     ind.scalar_fitness = 1.0 / (ind.factorial_rank / max_rank_k)
                 else: # Trường hợp rank không hợp lệ
                     ind.scalar_fitness = float('-inf')


    # Hàm selection được đăng ký trong _setup_deap
    def selTournamentMFEA(self, individuals: List[creator.Individual], k: int) -> List[creator.Individual]:
        """Chọn lọc tournament MFEA-II"""
        chosen = []
        n_tasks = self.algo_config.n_tasks
        tourn_size = self.algo_config.tournament_size

        # Tính max_rank cho từng task một lần để chuẩn hóa bonus đa dạng
        max_ranks_by_task = {}
        for task_id in range(n_tasks):
             task_inds = [ind for ind in individuals if ind.skill_factor == task_id and ind.factorial_rank != float('inf')]
             max_ranks_by_task[task_id] = max((p.factorial_rank for p in task_inds), default=1.0)
             if max_ranks_by_task[task_id] == 0: max_ranks_by_task[task_id] = 1.0 # Avoid division by zero


        for _ in range(k): # Chọn k cá thể
            if not individuals: break # Không còn cá thể nào để chọn

            # Chọn các ứng viên cho tournament
            actual_tourn_size = min(tourn_size, len(individuals))
            aspirants = random.sample(individuals, actual_tourn_size)

            # Tính điểm cho từng ứng viên
            def get_score(ind):
                # 1. Điểm từ hiệu năng (Scalar Fitness - lớn hơn là tốt hơn)
                base_score = ind.scalar_fitness if ind.scalar_fitness > float('-inf') else 0.0

                # 2. Bonus đa dạng (Factorial Rank - nhỏ hơn là tốt hơn)
                diversity_bonus = 0.0
                if ind.factorial_rank != float('inf'):
                    max_rank_pool = max_ranks_by_task.get(ind.skill_factor, 1.0)
                    rank_norm = ind.factorial_rank / max_rank_pool
                    diversity_bonus = 0.3 * (1.0 - rank_norm) # Rank thấp -> bonus cao

                # 3. Bonus cho ít features
                num_selected = np.sum(ind)
                max_f = self.algo_config.max_features
                min_f = self.algo_config.min_features
                feature_range = max_f - min_f
                sparsity = 1.0 - (num_selected - min_f) / (feature_range + 1e-9) if feature_range > 0 else 1.0
                feature_bonus = 0.2 * sparsity # Ít feature -> bonus cao

                # Tổng điểm, đảm bảo không âm
                return max(0, base_score + diversity_bonus + feature_bonus)

            # Chọn cá thể tốt nhất từ tournament
            best_aspirant = max(aspirants, key=get_score)
            chosen.append(best_aspirant)
            # (Optional: remove selected aspirant to prevent re-selection in the same generation? Depends on strategy)
            # individuals.remove(best_aspirant) # If using this, need to handle potential empty 'individuals' list

        return chosen
    
    


    def run(self) -> Dict[str, Any]:
        """Chạy thuật toán tối ưu hóa MFEA-II"""
        print("--- Bắt đầu thuật toán MFEA-II ---")
        # 1. Xử lý dữ liệu
        self.processed_data = self.data_processor.process_data()
        self.n_features = self.processed_data['n_features']
        self.feature_names = self.processed_data['feature_names']

        # 2. Thiết lập DEAP (cần n_features)
        self._setup_deap()
        self._setup_evaluation(self.processed_data) # Thiết lập hàm evaluate

        # 3. Khởi tạo quần thể
        population = self.toolbox.population(n=self.algo_config.pop_size)
        print(f"Đã khởi tạo quần thể với {len(population)} cá thể.")

        # 4. Đánh giá quần thể ban đầu
        print("Đánh giá quần thể ban đầu...")
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in tqdm(zip(invalid_ind, fitnesses), total=len(invalid_ind), desc="Đánh giá ban đầu"):
            ind.fitness.values = fit

        # 5. Tính toán rank và scalar fitness ban đầu
        self.calculate_ranks_fitness(population)
        print("Đã tính toán rank và scalar fitness ban đầu.")

        # 6. Lưu trữ best individuals cho mỗi task (Elitism)
        best_individuals = {k: None for k in range(self.algo_config.n_tasks)}
        best_fitness_val = {k: float('inf') for k in range(self.algo_config.n_tasks)}

        # --- Vòng lặp tiến hóa ---
        print("Bắt đầu vòng lặp tiến hóa...")
        for gen in range(self.algo_config.num_generations):
            # Tạo quần thể con bằng lai ghép và đột biến
            # varAnd áp dụng mate và mutate với xác suất đã đăng ký trong toolbox
            offspring = algorithms.varAnd(population, self.toolbox,
                                           cxpb=self.algo_config.crossover_rate,
                                           mutpb=self.algo_config.mutation_rate)

            # Đánh giá các cá thể con mới (chưa có fitness)
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Kết hợp quần thể cha mẹ và con cái
            combined_pop = population + offspring

            # Tính toán rank và scalar fitness cho quần thể kết hợp
            self.calculate_ranks_fitness(combined_pop)

            # Cập nhật best individuals (Elitism)
            current_best_updated = False
            for task_id in range(self.algo_config.n_tasks):
                task_candidates = [ind for ind in combined_pop if ind.skill_factor == task_id and ind.fitness.valid]
                if task_candidates:
                    best_for_task = min(task_candidates, key=lambda x: x.fitness.values[0])
                    if best_for_task.fitness.values[0] < best_fitness_val[task_id]:
                         best_fitness_val[task_id] = best_for_task.fitness.values[0]
                         best_individuals[task_id] = self.toolbox.clone(best_for_task) # Lưu bản sao
                         current_best_updated = True

            # Chọn lọc quần thể mới (kích thước pop_size)
            population[:] = self.toolbox.select(combined_pop, k=self.algo_config.pop_size)

            # In thông tin tiến trình
            if (gen + 1) % 10 == 0 or current_best_updated:
                 print(f"\n--- Thế hệ {gen+1}/{self.algo_config.num_generations} ---")
                 for task_id, task_name in [(0, "Toxicity"), (1, "Flammability")]:
                     best_ind = best_individuals.get(task_id) # Use .get for safety
                     if best_ind and best_ind.fitness.valid:
                         mse_val = best_ind.fitness.values[0]
                         sf = best_ind.scalar_fitness
                         fr = best_ind.factorial_rank
                         n_feat = np.sum(best_ind)
                         print(f"  Best {task_name}: MSE={mse_val:.4f}, NumFeat={n_feat}, ScalarFit={sf:.4f}, Rank={fr:.1f}")
                     else:
                         print(f"  Best {task_name}: Chưa tìm thấy hoặc không hợp lệ.")

        print("--- Hoàn thành thuật toán MFEA-II ---")

        return {
            'best_individuals': best_individuals,
            'best_fitness': best_fitness_val,
            'data': self.processed_data, # Trả về dữ liệu đã xử lý để đánh giá cuối cùng
            'feature_names': self.feature_names
        }