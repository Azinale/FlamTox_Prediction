# data_processor.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from typing import List, Tuple, Dict, Any
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

# Import configuration from the config file
from config import DataConfig

class RDKitProcessor:
    """Xử lý SMILES và tính toán descriptors bằng RDKit"""
    def __init__(self):
        self.desc_names = None
        
    def canonicalize_smiles(self, smiles_list: List[str]) -> List[str]:
        """Chuyển đổi danh sách SMILES sang dạng canonical"""
        canonicalized = []
        for smi in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    canonicalized.append(Chem.MolToSmiles(mol))
                else:
                    canonicalized.append(None)
            except Exception as e:
                print(f"Lỗi khi canonicalize SMILES {smi}: {e}")
                canonicalized.append(None)
        return canonicalized
        
    def calculate_descriptors(self, smiles_list: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        """Tính toán molecular descriptors cho danh sách SMILES"""
        mols = [Chem.MolFromSmiles(smi) for smi in smiles_list if smi is not None]
        calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
        self.desc_names = calc.GetDescriptorNames()

        descriptors = []
        for mol in mols:
            try:
                mol = Chem.AddHs(mol)
                descriptors.append(calc.CalcDescriptors(mol))
            except Exception as e:
                print(f"Lỗi khi tính descriptors: {e}")
                descriptors.append([None] * len(self.desc_names))

        return pd.DataFrame(descriptors, columns=self.desc_names), self.desc_names
        
    def process_smiles_file(self, input_file: str, output_file: str, smiles_col: str = 'Smiles') -> None:
        """Xử lý file chứa SMILES và lưu kết quả với descriptors"""
        try:
            # Đọc file input
            df = pd.read_csv(input_file)
            print(f"Đã đọc file {input_file} với {len(df)} dòng")
            
            # Canonicalize SMILES
            print("Đang canonicalize SMILES...")
            df['Canonical_SMILES'] = self.canonicalize_smiles(df[smiles_col])
            
            # Tính descriptors
            print("Đang tính descriptors...")
            descriptors_df, _ = self.calculate_descriptors(df['Canonical_SMILES'])
            
            # Kết hợp dữ liệu
            result_df = pd.concat([df.reset_index(drop=True), descriptors_df], axis=1)
            
            # Lưu kết quả
            result_df.to_csv(output_file, index=False)
            print(f"Đã lưu kết quả vào {output_file}")
            
        except Exception as e:
            print(f"Lỗi khi xử lý file: {e}")
            raise

class DataProcessor:
    """Xử lý dữ liệu: đọc, làm sạch, chuẩn hóa và chia tập"""
    def __init__(self, config: DataConfig):
        self.config = config
        self.common_features: List[str] | None = None
        self.scaler_tox = StandardScaler()
        self.scaler_flam = StandardScaler()
        self.rdkit_processor = RDKitProcessor()

    def _check_files_exist(self) -> None:
        """Kiểm tra sự tồn tại của file dữ liệu"""
        if not os.path.exists(self.config.tox_path):
            raise FileNotFoundError(f"Không tìm thấy file dữ liệu toxicity: {self.config.tox_path}")
        if not os.path.exists(self.config.flam_path):
            raise FileNotFoundError(f"Không tìm thấy file dữ liệu flammability: {self.config.flam_path}")

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Đọc dữ liệu từ file CSV"""
        self._check_files_exist() # Kiểm tra file trước khi đọc
        try:
            df_tox = pd.read_csv(self.config.tox_path)
            df_flam = pd.read_csv(self.config.flam_path)
            print(f"Đã đọc thành công {self.config.tox_path} ({len(df_tox)} dòng) và {self.config.flam_path} ({len(df_flam)} dòng).")
            return df_tox, df_flam
        except Exception as e:
            print(f"Lỗi khi đọc file CSV: {e}")
            raise # Re-raise the exception after logging

    def prepare_features(self, df_tox: pd.DataFrame, df_flam: pd.DataFrame) -> None:
        """Xác định các features (cột) chung giữa hai bộ dữ liệu"""
        # Các cột cần loại trừ (không phải features mô tả)
        cols_to_exclude = {"Smiles", "Canonical_SMILES", "pIGC50", "sdg"}
        common_cols = set(df_tox.columns) & set(df_flam.columns)
        self.common_features = sorted(list(common_cols - cols_to_exclude)) # Sắp xếp để đảm bảo thứ tự nhất quán
        if not self.common_features:
            raise ValueError("Không tìm thấy features chung nào giữa hai bộ dữ liệu sau khi loại trừ các cột không cần thiết.")
        print(f"Tìm thấy {len(self.common_features)} features chung.")

    def remove_outliers(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Loại bỏ giá trị ngoại lai bằng phương pháp clipping dựa trên độ lệch chuẩn"""
        df_copy = df.copy() # Làm việc trên bản sao để tránh SettingWithCopyWarning
        for col in columns:
            if pd.api.types.is_numeric_dtype(df_copy[col]): # Chỉ xử lý cột số
                mean = df_copy[col].mean()
                std = df_copy[col].std()
                lower_bound = mean - self.config.n_std * std
                upper_bound = mean + self.config.n_std * std
                df_copy[col] = df_copy[col].clip(lower_bound, upper_bound)
        return df_copy

    def process_data(self) -> Dict[str, Any]:
        """Quy trình xử lý dữ liệu hoàn chỉnh"""
        print("Bắt đầu xử lý dữ liệu...")
        df_tox, df_flam = self.load_data()
        self.prepare_features(df_tox, df_flam)

        # Chọn features chung và xử lý dữ liệu thiếu bằng median
        # Chuyển đổi sang float để đảm bảo tính toán số học
        X_tox = df_tox[self.common_features].fillna(df_tox[self.common_features].median()).astype(float)
        X_flam = df_flam[self.common_features].fillna(df_flam[self.common_features].median()).astype(float)
        print("Đã xử lý giá trị thiếu bằng median.")

        # Xử lý outliers
        X_tox = self.remove_outliers(X_tox, self.common_features)
        X_flam = self.remove_outliers(X_flam, self.common_features)
        print(f"Đã loại bỏ outliers (clipping với {self.config.n_std} std).")

        # Chuẩn bị biến mục tiêu (target), xử lý thiếu bằng median
        y_tox = df_tox["pIGC50"].fillna(df_tox["pIGC50"].median())
        y_flam = df_flam["sdg"].fillna(df_flam["sdg"].median())

        # Chia dữ liệu train/test
        X_tox_train, X_tox_test, y_tox_train, y_tox_test = train_test_split(
            X_tox, y_tox, test_size=self.config.test_size, random_state=self.config.random_state
        )
        X_flam_train, X_flam_test, y_flam_train, y_flam_test = train_test_split(
            X_flam, y_flam, test_size=self.config.test_size, random_state=self.config.random_state
        )
        print(f"Đã chia dữ liệu thành train/test (test_size={self.config.test_size}).")

        # Scale features với StandardScaler (fit trên train, transform trên cả train và test)
        X_tox_train_scaled = self.scaler_tox.fit_transform(X_tox_train)
        X_tox_test_scaled = self.scaler_tox.transform(X_tox_test)
        X_flam_train_scaled = self.scaler_flam.fit_transform(X_flam_train)
        X_flam_test_scaled = self.scaler_flam.transform(X_flam_test)
        print("Đã chuẩn hóa features bằng StandardScaler.")
        print("Hoàn tất xử lý dữ liệu.")

        return {
            'X_tox_train': X_tox_train_scaled,
            'X_tox_test': X_tox_test_scaled,
            'y_tox_train': y_tox_train.values, # Trả về numpy array
            'y_tox_test': y_tox_test.values,   # Trả về numpy array
            'X_flam_train': X_flam_train_scaled,
            'X_flam_test': X_flam_test_scaled,
            'y_flam_train': y_flam_train.values, # Trả về numpy array
            'y_flam_test': y_flam_test.values,   # Trả về numpy array
            'n_features': X_tox_train.shape[1], # Số lượng features chung
            'feature_names': self.common_features # Tên các features chung
        }