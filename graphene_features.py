# graphene_features.py (物理升级版)
import numpy as np
import pandas as pd

def calculate_theoretical_k(df, return_components=False):
    """
    升级版：支持返回物理分量 (Components)，用于 Agent 进行深度归因分析。
    """
    # 1. 获取参数
    T = df.get('temperature', 300.0)
    L = df.get('length_um', 10.0) 
    defect = df.get('defect_ratio', 0.0) 
    
    # 2. 归一化缺陷 (Defect Penalty)
    log_D = np.log10(defect + 1e-12)
    norm_D = (log_D - (-8)) / 6.0
    # [分析点] 缺陷因子：越接近 1.0 说明越纯净，越接近 0.0 说明缺陷越严重
    defect_factor = (1.0 - 0.85 * norm_D) 
    
    # 3. 温度因子 (Umklapp Scattering)
    # [分析点] 温度因子：衡量声子-声子散射的强度
    temp_factor = (300.0 / (T + 1.0)) ** 1.0 
    
    # 4. 尺寸因子 (Ballistic Transport)
    # [分析点] 尺寸因子：衡量边界散射的影响 (>1.0 代表弹道输运增益)
    size_factor = 1.0 + 0.6 * np.log10(L + 0.1)
    size_factor = np.clip(size_factor, 0.5, 5.0) 
    
    base_constant = 3200.0 
    
    # 理论估算值
    k_theory = base_constant * temp_factor * size_factor * defect_factor
    final_k = np.maximum(k_theory, 10.0)

    if return_components:
        # 返回详细的物理归因字典 (取平均值以适应 DataFrame)
        return final_k, {
            "defect_factor": np.mean(defect_factor),
            "temp_factor": np.mean(temp_factor),
            "size_factor": np.mean(size_factor),
            "base_k": base_constant
        }
    
    return final_k

def enhance_features(df):
    """
    特征工程管道：原始数据 -> 机器学习可用特征
    """
    df_out = df.copy()
    
    # 1. 基础对数变换
    if 'temperature' in df_out.columns:
        df_out['log_temp'] = np.log10(df_out['temperature'] + 1.0)
    if 'length_um' in df_out.columns:
        df_out['log_length'] = np.log10(df_out['length_um'] + 0.001)
    if 'defect_ratio' in df_out.columns:
        df_out['log_defect'] = np.log10(df_out['defect_ratio'] + 1e-9)
        
    # 2. 处理基底因子
    if 'substrate_type' in df_out.columns:
        sub_map = {
            'Suspended': 1.0, 
            'hBN': 0.8, 
            'SiO2': 0.5, 
            'Au': 0.2, 
            'Cu': 0.2
        }
        df_out['substrate_factor'] = df_out['substrate_type'].map(sub_map).fillna(0.5)
    else:
        df_out['substrate_factor'] = 0.5

    # 3. 注入物理灵魂
    raw_theory_k = calculate_theoretical_k(df_out,return_components=False)
    
    # 最终理论特征 = 修正后的物理上限 * 基底衰减
    # 这样 Suspended 就能跑到 4000-5000，而 SiO2 依然会被拉回 2000 以下
    df_out['log_theory_k'] = np.log10(raw_theory_k * df_out['substrate_factor'])
    
    return df_out