# graphene_tools.py (技能升级版)
import json
import io
import base64
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from langchain.tools import tool
from graphene_features import enhance_features, calculate_theoretical_k

# === 全局配置 ===
MODEL_PATH = "advanced_model.pkl" 
SCALER_PATH = "feature_scaler.pkl"
FEATURE_PATH = "model_features.json"

_gpr_model = None
_scaler = None
_model_features = None

def load_resources():
    """加载资源 (单例模式)"""
    global _gpr_model, _scaler, _model_features
    if _model_features is None:
        try:
            with open(FEATURE_PATH, "r", encoding='utf-8') as f:
                _model_features = json.load(f)
            _scaler = joblib.load(SCALER_PATH)
            _gpr_model = joblib.load(MODEL_PATH)
        except Exception as e:
            return None, None, None, f"资源加载失败: {str(e)}"
    return _gpr_model, _scaler, _model_features, ""

def _predict_core(length_um, temperature_k, defect_ratio, layers=1, doping=0.0, substrate='Suspended'):
    """
    [内部核心函数] 纯粹的预测逻辑，供所有 Tool 复用。
    返回: (预测均值, 标准差)
    """
    model, scaler, features, err = load_resources()
    if err: raise Exception(err)

    # 1. 构造数据
    raw_data = pd.DataFrame([{
        'length_um': length_um,
        'temperature': temperature_k,
        'defect_ratio': defect_ratio,
        'layers': layers,
        'doping_concentration': doping,
        'substrate_type': substrate
    }])
    
    # 2. 特征工程
    enhanced = enhance_features(raw_data)
    final_input = pd.DataFrame(0.0, index=[0], columns=features)
    for col in features:
        if col in enhanced.columns:
            final_input[col] = enhanced[col]
        elif col.startswith('substrate_type_') and substrate == col.replace('substrate_type_', ''):
            final_input[col] = 1.0
            
    # 3. 预测
    X_scaled = scaler.transform(final_input)
    mean_log, std_log = model.predict(X_scaled, return_std=True)
    
    # 4. 还原 (Log -> Real)
    pred_val = 10 ** mean_log[0] - 1.0
    # Log正态分布的近似标准差处理略复杂，这里简化处理用于趋势展示
    pred_std = std_log[0] 
    
    return pred_val, pred_std

@tool
def ml_prediction_tool(length_um: float, temperature_k: float, defect_ratio: float, **kwargs) -> str:
    """[基础预测] 预测指定条件下的石墨烯热导率。"""
    try:
        val, std = _predict_core(length_um, temperature_k, defect_ratio, **kwargs)
        # 计算置信区间
        lower = 10 ** (np.log10(val+1) - 1.96 * std) - 1
        upper = 10 ** (np.log10(val+1) + 1.96 * std) - 1
        return f"{val:.2f} W/mK (95%置信区间: {lower:.0f} ~ {upper:.0f})"
    except Exception as e:
        return f"预测错误: {e}"

@tool
def inverse_design_tool(target_k: float, length_um: float, temperature_k: float) -> str:
    """
    [逆向设计技能] 已知目标热导率，反推需要的‘缺陷浓度’上限。
    如果用户问'怎么做才能达到xxx热导率'，请用此工具。
    """
    try:
        # 定义目标函数：寻找 defect 使得 |预测值 - 目标值| 最小
        def objective(defect):
            # 限制 defect 范围防止物理无意义
            if defect < 0 or defect > 0.1: return 1e6
            pred, _ = _predict_core(length_um, temperature_k, defect)
            return abs(pred - target_k)

        # 使用标量最小化算法 (在 0.0 ~ 0.05 范围内搜索)
        res = minimize_scalar(objective, bounds=(0.0, 0.05), method='bounded')
        
        if res.success:
            found_defect = res.x
            # 验证一下找到的值是多少
            final_k, _ = _predict_core(length_um, temperature_k, found_defect)
            
            if abs(final_k - target_k) > target_k * 0.2:
                return f"难以达到 {target_k} W/mK。在当前温度和尺寸下，即使接近完美晶格(缺陷≈0)，预测值也仅为 {final_k:.1f} W/mK。"
            
            return (f"为了达到 {target_k} W/mK，建议控制缺陷浓度在 {found_defect*100:.4f}% 左右。\n"
                    f"(基于优化算法反推，此时预测值为 {final_k:.1f} W/mK)")
        else:
            return "反推计算未收敛，目标值可能超出物理极限。"
            
    except Exception as e:
        return f"逆向设计出错: {e}"

@tool
def plot_trend_tool(variable: str, fixed_params: str) -> str:
    """
    [可视化技能] 绘制热导率随变量变化的趋势图。
    Args:
        variable: 变化的变量，只能是 'temperature' 或 'defect' 或 'length'。
        fixed_params: JSON字符串，包含其他固定参数。例如 '{"length_um": 10, "defect_ratio": 0.001}'
    Returns:
        Markdown 格式的图片链接。
    """
    try:
        params = json.loads(fixed_params)
        length = params.get('length_um', 10.0)
        temp = params.get('temperature', 300.0)
        defect = params.get('defect_ratio', 0.001)
        
        x_vals = []
        y_vals = []
        x_label = ""
        
        # 生成绘图数据
        if variable == 'temperature':
            x_vals = np.linspace(100, 600, 20)
            x_label = "Temperature (K)"
            for t in x_vals:
                k, _ = _predict_core(length, t, defect)
                y_vals.append(k)
        elif variable == 'defect':
            x_vals = np.linspace(0.0, 0.02, 20)
            x_label = "Defect Ratio"
            for d in x_vals:
                k, _ = _predict_core(length, temp, d)
                y_vals.append(k)
        elif variable == 'length':
            x_vals = np.linspace(1.0, 50.0, 20)
            x_label = "Length (um)"
            for l in x_vals:
                k, _ = _predict_core(l, temp, defect)
                y_vals.append(k)
        else:
            return "不支持的变量类型，仅支持 temperature, defect, length"

        # 绘图
        plt.figure(figsize=(7, 4))
        plt.plot(x_vals, y_vals, 'o-', color='#1f77b4', linewidth=2, markersize=5)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xlabel(x_label)
        plt.ylabel("Thermal Conductivity (W/mK)")
        plt.title(f"Trend Analysis ({variable} varying)")
        plt.tight_layout()

        # 转为 Base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        # 返回 Markdown 图片语法
        return f"![trend_plot](data:image/png;base64,{img_str})"

    except Exception as e:
        return f"绘图失败: {e}"

@tool
def physics_calculation_tool(temperature_k: float, defect_ratio: float, length_um: float = 10.0, **kwargs) -> str:
    """[物理公式工具] 计算理论热导率上限，并返回物理机制拆解分析。"""
    # 保持你原来的代码不变
    try:
        temp_df = pd.DataFrame([{
            'temperature': temperature_k,
            'defect_ratio': defect_ratio,
            'length_um': length_um,
            'substrate_type': 'Suspended' 
        }])
        k_val, components = calculate_theoretical_k(temp_df, return_components=True)
        analysis_data = {
            "理论上限 (W/mK)": round(k_val[0], 2),
            "机制拆解": {
                "声子散射因子": round(components['temp_factor'], 3),
                "边界散射因子": round(components['size_factor'], 3),
                "缺陷散射因子": round(components['defect_factor'], 3)
            }
        }
        return f"计算成功: {json.dumps(analysis_data, ensure_ascii=False)}"
    except Exception as e:
        return f"物理计算出错: {str(e)}"