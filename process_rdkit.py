# process_rdkit.py
import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import traceback
from config import RDKitConfig

def calculate_descriptors(smiles):
    """Tính toán các descriptor cho một SMILES"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Lấy danh sách tất cả các descriptor có sẵn
    descriptor_names = [x[0] for x in Descriptors._descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    
    try:
        descriptors = calculator.CalcDescriptors(mol)
        return dict(zip(descriptor_names, descriptors))
    except:
        return None

def process_smiles_file(input_file, output_file):
    """Xử lý file SMILES và tạo descriptor"""
    print(f"Đang đọc file: {input_file}")
    df = pd.read_csv(input_file)
    
    print("Đang tính toán descriptors...")
    descriptors_list = []
    valid_smiles = []
    
    for idx, row in df.iterrows():
        smiles = row['Smiles']
        descriptors = calculate_descriptors(smiles)
        
        if descriptors is not None:
            descriptors_list.append(descriptors)
            valid_smiles.append(smiles)
    
    print(f"Đã xử lý xong {len(valid_smiles)}/{len(df)} SMILES hợp lệ")
    
    # Tạo DataFrame mới với descriptors
    descriptors_df = pd.DataFrame(descriptors_list)
    descriptors_df['Smiles'] = valid_smiles
    
    # Lưu kết quả
    print(f"Đang lưu kết quả vào: {output_file}")
    descriptors_df.to_csv(output_file, index=False)
    print("Đã hoàn thành!")

def main():
    """Hàm chính để xử lý dữ liệu"""
    print("===== BẮT ĐẦU XỬ LÝ DỮ LIỆU VỚI RDKIT =====")
    
    # Lấy cấu hình
    config = RDKitConfig()
    
    try:
        # Xử lý file toxicity
        if not os.path.exists(config.tox_descriptor_path):
            print(f"\nĐang xử lý file toxicity: {config.tox_path}")
            process_smiles_file(config.tox_path, config.tox_descriptor_path)
        
        # Xử lý file flammability
        if not os.path.exists(config.flam_descriptor_path):
            print(f"\nĐang xử lý file flammability: {config.flam_path}")
            process_smiles_file(config.flam_path, config.flam_descriptor_path)
            
        print("\nĐã hoàn thành xử lý dữ liệu với RDKit!")
    except Exception as e:
        print(f"\n!!! LỖI KHI XỬ LÝ DỮ LIỆU VỚI RDKIT: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 