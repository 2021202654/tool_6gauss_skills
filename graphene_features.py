# graphene_features.py
import numpy as np
import pandas as pd

def calculate_theoretical_k(df, return_components=False):
    """
    计算理论热导率上限 (修正版：增加物理限制，防止理论值虚高)
    """
    T = df.get('temperature', 300.0)
    L = df.get('length_um', 10.0) 
    defect = df.get('defect_ratio', 0.0) 
    
    # 1. 缺陷散射 (保持不变)
    log_D = np.log10(defect + 1e-12)
    norm_D = (log_D - (-8)) / 6.0
    defect_factor = (1.0 - 0.85 * norm_D) 
    
    # 2. 温度因子 (保持不变)
    temp_factor = (300.0 / (T + 1.0)) ** 1.0 
    
    # 3. 尺寸因子 (保持不变)
    size_factor = 1.0 + 0.6 * np.log10(L + 0.1)
    size_factor = np.clip(size_factor, 0.5, 5.0) 
    
    # === 🔥 关键修改点：基础常数与寄生散射 ===
    # 原来是 3200 (纯理想)，现在降级为 2000 (工程级理想)
    base_constant = 4000 
    
    # 计算理想值
    k_ideal = base_constant * temp_factor * size_factor * defect_factor
    
    # === 🔥 引入 "Matthiessen's Rule" 限制 ===
    # 假设无论如何优化，接触热阻和晶界散射让热导率很难超过 8000
    # 1/k_total = 1/k_ideal + 1/k_limit
    k_limit = 8000.0
    
    final_k = (k_ideal * k_limit) / (k_ideal + k_limit)
    final_k = np.maximum(final_k, 10.0)

    if return_components:
        return final_k, {
            "defect_factor": np.mean(defect_factor),
            "temp_factor": np.mean(temp_factor),
            "size_factor": np.mean(size_factor),
            "base_k": base_constant
        }
    
    return final_k

def enhance_features(df):
    """特征工程管道 (保持你的原逻辑，加上一点微调)"""
    df_out = df.copy()
    
    if 'temperature' in df_out.columns:
        df_out['log_temp'] = np.log10(df_out['temperature'] + 1.0)
    if 'length_um' in df_out.columns:
        df_out['log_length'] = np.log10(df_out['length_um'] + 0.001)
    if 'defect_ratio' in df_out.columns:
        df_out['log_defect'] = np.log10(df_out['defect_ratio'] + 1e-9)

    # 简单的物理因子
    df_out['iso_factor'] = 1.0
    df_out['chem_factor'] = 1.0

    # 计算修正后的理论值
    raw_theory_k = calculate_theoretical_k(df_out, return_components=False)
    
    # 处理基底 (Substrate)
    if 'substrate_type' in df_out.columns:
        sub_map = {'Suspended': 1.0, 'hBN': 0.8, 'SiO2': 0.4, 'Au': 0.1, 'Cu': 0.1}
        substrate_factor = df_out['substrate_type'].map(sub_map).fillna(0.4)
    else:
        substrate_factor = 0.4 # 默认认为有基底干扰

    combined_factor = substrate_factor
    
    # 特征里也存一份 log_theory
    df_out['log_theory_k'] = np.log10(raw_theory_k * combined_factor + 1.0)
    

    return df_out
