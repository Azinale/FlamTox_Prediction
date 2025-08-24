import random
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import traceback # For detailed error logging
import time

# Import configurations and the optimizer class
from config import DataConfig, AlgorithmConfig
from mfea_optimizer import MFEAIIOptimizer # Import MFEA-II
from ga_optimizer import GAOptimizer       # Import GA

def main():
    """Hàm chính để chạy toàn bộ quy trình"""
    print("===== BẮT ĐẦU QUY TRÌNH CHỌN FEATURES ĐA NHIỆM =====")
    # Cấu hình
    data_config = DataConfig()
    algo_config = AlgorithmConfig()

    # --- Add seeding for reproducibility ---
    seed = data_config.random_state
    random.seed(seed)
    np.random.seed(seed)
    print(f"Đã thiết lập random seed = {seed}")
    # --- End seeding ---

    # Hỏi người dùng chọn thuật toán
    print("\nChọn thuật toán cần chạy:")
    print("  1 - MFEA-II (đa nhiệm)")
    print("  2 - GA (đơn nhiệm)")
    choice = input("Nhập lựa chọn (1/2): ").strip()

    if choice == "1":
        optimizer_class = MFEAIIOptimizer
        algo_name = "MFEA-II"
    elif choice == "2":
        optimizer_class = GAOptimizer
        algo_name = "GA"
    else:
        print("Lựa chọn không hợp lệ! Mặc định dùng MFEA-II.")
        optimizer_class = MFEAIIOptimizer
        algo_name = "MFEA-II"

    # đo thời gian chạy
    start_time = time.time()

    # Khởi tạo và chạy thuật toán
    try:
        print(f"\n===== BẮT ĐẦU CHẠY THUẬT TOÁN {algo_name} =====")
        optimizer = optimizer_class(data_config, algo_config)
        print("Bắt đầu chạy thuật toán tối ưu hóa...")
        results = optimizer.run()
    except FileNotFoundError as e:
        print(f"\n!!! LỖI: {e}")
        print("!!! Vui lòng kiểm tra lại đường dẫn file CSV trong 'config.py'.")
        return
    except ValueError as e:
        print(f"\n!!! LỖI CẤU HÌNH HOẶC DỮ LIỆU: {e}")
        return
    except ImportError as e:
         print(f"\n!!! LỖI IMPORT: {e}")
         print("!!! Đảm bảo các file config.py, data_processor.py, mfea_optimizer.py, ga_optimizer.py nằm cùng thư mục với main.py.")
         return
    except Exception as e:
        print(f"\n!!! ĐÃ XẢY RA LỖI KHÔNG MONG MUỐN:")
        print(f"!!! {e}")
        traceback.print_exc()
        return

    # In kết quả cuối cùng
    print("\n\n===== KẾT QUẢ CUỐI CÙNG =====")
    feature_names = results.get('feature_names', [])

    for task_id, task_name in [(0, "Toxicity (pIGC50)"), (1, "Flammability (sdg)")]:
        print(f"\n--- Best individual cho bài toán {task_name} ---")
        best_ind = results['best_individuals'].get(task_id)

        if best_ind and best_ind.fitness.valid:
            selected_mask = np.array(best_ind, dtype=bool)
            selected_indices = np.where(selected_mask)[0]
            selected_feature_names = [feature_names[i] for i in selected_indices if i < len(feature_names)]
            num_selected = len(selected_indices)

            print(f"  MSE (Linear Regression trong {algo_name}): {best_ind.fitness.values[0]:.5f}")
            print(f"  Scalar Fitness: {best_ind.scalar_fitness:.5f}")
            print(f"  Factorial Rank: {best_ind.factorial_rank:.1f}")
            print(f"  Skill Factor: {best_ind.skill_factor}")
            print(f"  Số features được chọn: {num_selected}")
            print(f"  Tên features được chọn: {selected_feature_names}")

            # Đánh giá cuối cùng với RandomForestRegressor trên tập test
            if num_selected > 0:
                print("\n  Đánh giá cuối cùng với RandomForestRegressor (trên tập test):")
                try:
                    data = results['data']
                    if task_id == 0:
                        X_train = data['X_tox_train'][:, selected_mask]
                        X_test = data['X_tox_test'][:, selected_mask]
                        y_train = data['y_tox_train']
                        y_test = data['y_tox_test']
                    else: # task_id == 1
                        X_train = data['X_flam_train'][:, selected_mask]
                        X_test = data['X_flam_test'][:, selected_mask]
                        y_train = data['y_flam_train']
                        y_test = data['y_flam_test']

                    # Huấn luyện RandomForest
                    print("    Đang huấn luyện RandomForest...")
                    final_model = RandomForestRegressor(n_estimators=100,
                                                        random_state=data_config.random_state,
                                                        n_jobs=-1)
                    final_model.fit(X_train, y_train)
                    y_pred_rf = final_model.predict(X_test)
                    mse_rf = mean_squared_error(y_test, y_pred_rf)
                    r2_rf = r2_score(y_test, y_pred_rf)
                    print(f"    >> MSE (RandomForest): {mse_rf:.5f}")
                    print(f"    >> R^2 (RandomForest): {r2_rf:.5f}")

                except Exception as e:
                    print(f"    Lỗi khi tính toán final MSE/R2 với RandomForest: {e}")
                    traceback.print_exc()
            else:
                 print("    Không thể đánh giá RandomForest vì không có feature nào được chọn.")
        else:
            print(f"  Không tìm thấy individual hợp lệ tốt nhất cho bài toán {task_name}.")

    # Kết thúc và in thời gian chạy
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n===== KẾT THÚC {algo_name} - Tổng thời gian chạy: {elapsed_time:.2f} giây =====")

if __name__ == "__main__":
    main()
