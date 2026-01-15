# graphene_tools.py (适配 Delta Learning 残差学习版)
import json
import io
import base64
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from langchain.tools import tool

# 🔥 核心引入：需要实时计算物理理论值
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
    [核心推理函数 - Delta Learning 版]
    逻辑：最终预测 = 物理公式理论值 * 模型预测的修正比例
    返回: (预测值, Log空间的标准差)
    """
    model, scaler, features, err = load_resources()
    if err: raise Exception(err)

    # 1. 构造原始数据 (用于计算物理理论值)
    raw_data = pd.DataFrame([{
        'length_um': length_um,
        'temperature': temperature_k,
        'defect_ratio': defect_ratio,
        'layers': layers,
        'doping_concentration': doping,
        'substrate_type': substrate
    }])
    
    # 2. 计算物理基准值 (Theory Baseline)
    # 这一步保证了：即使模型不知道怎么预测，至少有一个符合物理规律的基准
    enhanced = enhance_features(raw_data)
    # 注意：这里我们获取最纯粹的理论计算值
    base_theory_k = calculate_theoretical_k(enhanced).iloc[0]

    # 3. 准备模型输入 (用于预测修正比例)
    final_input = pd.DataFrame(0.0, index=[0], columns=features)
    for col in features:
        if col in enhanced.columns:
            final_input[col] = enhanced[col]
        elif col.startswith('substrate_type_') and substrate == col.replace('substrate_type_', ''):
            final_input[col] = 1.0
            
    # 4. 模型预测 (预测的是 Log10(Ratio))
    X_scaled = scaler.transform(final_input)
    mean_log_ratio, std_log_ratio = model.predict(X_scaled, return_std=True)
    
    # 5. 还原结果
    # 预测的比例系数
    pred_ratio = 10 ** mean_log_ratio[0]
    
    # 最终结果 = 理论值 * 修正比例
    final_pred_val = base_theory_k * pred_ratio
    
    # 返回: 预测值, 以及 Log(Ratio) 的标准差 (用于后续计算置信区间)
    return final_pred_val, std_log_ratio[0], base_theory_k

@tool
def ml_prediction_tool(length_um: float, temperature_k: float, defect_ratio: float, **kwargs) -> str:
    """[基础预测] 预测指定条件下的石墨烯热导率 (基于 物理+AI 混合驱动)。"""
    try:
        # 获取 预测值, 不确定度, 理论基准
        val, std_log, theory_base = _predict_core(length_um, temperature_k, defect_ratio, **kwargs)
        
        # 计算 95% 置信区间
        # 逻辑：先算出 Ratio 的区间，再乘上 Theory
        # mean_log_ratio 隐含在 val 里面，这里我们反推一下或者直接利用 std_log
        # Ratio 的上界 = Ratio_Mean * 10^(1.96 * std)
        # 既然 val = Theory * Ratio_Mean
        # 那么 Val_Upper = val * 10^(1.96 * std)
        
        factor_upper = 10 ** (1.96 * std_log)
        factor_lower = 10 ** (-1.96 * std_log)
        
        upper = val * factor_upper
        lower = val * factor_lower

        return (f"预测热导率: {val:.2f} W/mK\n"
                f"95% 置信区间: {lower:.0f} ~ {upper:.0f} W/mK\n"
                f"(物理理论基准: {theory_base:.1f} W/mK，AI 修正系数: {val/theory_base:.2f}x)")
    except Exception as e:
        return f"预测错误: {e}"

@tool
def inverse_design_tool(target_k: float, length_um: float, temperature_k: float) -> str:
    """
    [逆向设计技能] 已知目标热导率，反推需要的‘缺陷浓度’上限。
    """
    try:
        def objective(defect):
            if defect < 0 or defect > 0.05: return 1e6
            # 注意 _predict_core 现在返回 3 个值，我们要第一个
            pred, _, _ = _predict_core(length_um, temperature_k, defect)
            return abs(pred - target_k)

        res = minimize_scalar(objective, bounds=(0.0, 0.05), method='bounded')
        
        if res.success:
            found_defect = res.x
            final_k, _, _ = _predict_core(length_um, temperature_k, found_defect)
            
            if abs(final_k - target_k) > target_k * 0.2:
                return f"难以达到 {target_k} W/mK。即使接近完美晶格(缺陷≈0)，预测值也仅为 {final_k:.1f} W/mK。"
            
            return (f"为了达到 {target_k} W/mK，建议控制缺陷浓度在 {found_defect*100:.4f}% 左右。\n"
                    f"(预测值: {final_k:.1f} W/mK)")
        else:
            return "反推计算未收敛，目标值可能超出物理极限。"
            
    except Exception as e:
        return f"逆向设计出错: {e}"

@tool
def plot_trend_tool(variable: str, fixed_params: str) -> str:
    """
    [可视化技能] 绘制热导率随变量变化的趋势图。
    """
    try:
        params = json.loads(fixed_params)
        length = params.get('length_um', 10.0)
        temp = params.get('temperature', 300.0)
        defect = params.get('defect_ratio', 0.001)
        
        x_vals = []
        y_vals = []
        theory_vals = [] # 新增：画出纯理论线做对比
        x_label = ""
        
        if variable == 'temperature':
            x_vals = np.linspace(100, 600, 20)
            x_label = "Temperature (K)"
            for t in x_vals:
                k, _, th = _predict_core(length, t, defect)
                y_vals.append(k)
                theory_vals.append(th)
        elif variable == 'defect':
            x_vals = np.linspace(0.0, 0.02, 20)
            x_label = "Defect Ratio"
            for d in x_vals:
                k, _, th = _predict_core(length, temp, d)
                y_vals.append(k)
                theory_vals.append(th)
        elif variable == 'length':
            x_vals = np.linspace(1.0, 50.0, 20)
            x_label = "Length (um)"
            for l in x_vals:
                k, _, th = _predict_core(l, temp, defect)
                y_vals.append(k)
                theory_vals.append(th)
        else:
            return "不支持的变量类型"

        plt.figure(figsize=(7, 4))
        # 绘制最终预测
        plt.plot(x_vals, y_vals, 'o-', color='#1f77b4', linewidth=2, label='AI + Physics Prediction')
        # 绘制纯物理基准
        plt.plot(x_vals, theory_vals, '--', color='gray', alpha=0.6, label='Pure Physics Formula')
        
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xlabel(x_label)
        plt.ylabel("Thermal Conductivity (W/mK)")
        plt.title(f"Trend Analysis ({variable})")
        plt.legend()
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        return f"![trend_plot](data:image/png;base64,{img_str})"

    except Exception as e:
        return f"绘图失败: {e}"

@tool
def physics_calculation_tool(temperature_k: float, defect_ratio: float, length_um: float = 10.0, **kwargs) -> str:
    """[物理公式工具] 计算理论热导率上限及拆解。"""
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