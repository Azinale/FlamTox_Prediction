# evaluation.py
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from config import GAConfig # Sử dụng một config bất kỳ để lấy min/max features

# Giả định cả hai thuật toán có cùng ràng buộc min/max features
algo_config = GAConfig()

def evaluate_features(individual: list, data: dict) -> tuple:
    """
    Hàm đánh giá fitness (MSE) của một tổ hợp features cho một nhiệm vụ.
    """
    selected_indices = [i for i, bit in enumerate(individual) if bit == 1]
    num_selected = len(selected_indices)

    if not (algo_config.min_features <= num_selected <= algo_config.max_features):
        return (float('inf'),)
    
    if num_selected == 0:
        return (float('inf'),)

    X_train = data['X_train'][:, selected_indices]
    y_train = data['y_train']
    X_test = data['X_test'][:, selected_indices]
    y_test = data['y_test']
    
    try:
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        if np.isnan(y_pred).any() or np.isinf(y_pred).any():
            return (float('inf'),)
            
        mse = mean_squared_error(y_test, y_pred)
        
        # Thêm hình phạt nhỏ cho số lượng feature để khuyến khích mô hình đơn giản
        penalty_coef = 0.01
        feature_range = algo_config.max_features - algo_config.min_features
        penalty = penalty_coef * (num_selected - algo_config.min_features) / feature_range if feature_range > 0 else 0
        
        return (mse + penalty,)
        
    except Exception:
        return (float('inf'),)