# MFEA-II for Feature Selection in Chemical Properties Prediction

## Mô tả
Dự án này triển khai thuật toán MFEA-II (Multifactorial Evolutionary Algorithm II) để chọn lọc features cho bài toán dự đoán hai tính chất hóa học:
1. Độc tính (Toxicity - pIGC50)
2. Khả năng cháy (Flammability - sdg)

## Cấu trúc thư mục
```
final_code/
├── config.py                    # Cấu hình cho RDKit và MFEA-II
├── process_rdkit.py             # Xử lý dữ liệu với RDKit
├── data_processor.py            # Xử lý dữ liệu cho MFEA-II
├── mfea_optimizer.py            # Triển khai thuật toán MFEA-II
├── main.py                      # Chạy thuật toán chính
└── Data/                        # Thư mục chứa dữ liệu
    ├── pIGC50.csv               # Dữ liệu độc tính
    ├── lflt.csv                 # Dữ liệu nhiệt độ cháy
    ├── pIGC50_description.csv   # Dữ liệu độc tính đã được giải thích
    └── lflt_description.csv     # Dữ liệu nhiệt độ cháy đã được giải thích
```

## Yêu cầu
- Python 3.7+
- Các thư viện cần thiết:
  ```
  numpy>=1.19.0
  pandas>=1.2.0
  scikit-learn>=0.24.0
  rdkit>=2020.09.1
  deap>=1.3.1
  ```

## Cài đặt
1. Clone repository
2. Cài đặt các thư viện cần thiết:
   ```bash
   pip install -r requirements.txt
   ```

## Cách sử dụng

### Bước 1: Xử lý dữ liệu với RDKit
```bash
python process_rdkit.py
```
- Đọc file `pIGC50.csv` và `lflt.csv`
- Tính toán descriptors cho mỗi SMILES
- Lưu kết quả vào `pIGC50_descriptor.csv` và `lflt_descriptor.csv`

### Bước 2: Chạy thuật toán MFEA-II
```bash
python main.py
```
- Đọc dữ liệu đã được xử lý
- Chạy thuật toán MFEA-II để chọn lọc features
- In kết quả cuối cùng

## Cấu hình
### RDKitConfig (process_rdkit.py)
- `tox_path`: Đường dẫn file dữ liệu độc tính
- `flam_path`: Đường dẫn file dữ liệu khả năng cháy
- `tox_descriptor_path`: Đường dẫn lưu descriptors độc tính
- `flam_descriptor_path`: Đường dẫn lưu descriptors khả năng cháy

### DataConfig (main.py)
- `tox_path`: Đường dẫn file descriptors độc tính
- `flam_path`: Đường dẫn file descriptors khả năng cháy
- `test_size`: Tỷ lệ dữ liệu test
- `random_state`: Seed cho reproducibility
- `n_std`: Số độ lệch chuẩn cho outlier clipping

### AlgorithmConfig (main.py)
- `pop_size`: Kích thước quần thể
- `num_generations`: Số thế hệ
- `mutation_rate`: Tỷ lệ đột biến
- `crossover_rate`: Tỷ lệ lai ghép
- `rmp`: Xác suất lai ghép giữa các tasks
- `tournament_size`: Kích thước tournament
- `n_tasks`: Số lượng bài toán
- `min_features`: Số features tối thiểu
- `max_features`: Số features tối đa

## Kết quả
- In ra các features được chọn cho mỗi bài toán
- MSE và R² trên tập test
- Số lượng features được chọn
- Tên các features được chọn

## Tham khảo
1. Gupta, A., Ong, Y. S., & Feng, L. (2015). Multifactorial evolution: Toward evolutionary multitasking. IEEE Transactions on Evolutionary Computation, 20(3), 343-357.
2. Gupta, A., Ong, Y. S., & Feng, L. (2016). Multifactorial evolution: Toward evolutionary multitasking. IEEE Transactions on Evolutionary Computation, 20(3), 343-357. 