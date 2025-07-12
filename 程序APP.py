
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import pathlib
from pathlib import Path
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib
import sklearn
import xgboost
from xgboost import XGBClassifier

# 设置页面标题
st.set_page_config(
    page_title="Fungal infection prediction system", 
    page_icon="🩺",
    layout="wide"
)
st.title("Fungal infection prediction system")



# 加载模型
model = joblib.load('xgb_model.pkl')

# 侧边栏 - 关于
st.sidebar.header("About")
st.sidebar.info("""
Input the relevant parameters of the patient, and the system will calculate the probability of fungal infection occurrence.
""")

# 风险等级解释
st.sidebar.subheader("Risk level description")
st.sidebar.markdown("""
- **Low risk**: < 20% 
- **Concentration risk**: 20% - 50% 
- **High Risk**: > 50% 
""")



# 主界面
st.header("Please enter the patient parameters")

# 创建输入表单
with st.form("prediction_form"):
    # 使用多列布局
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Laboratory index")
        # 实验室指标输入
        wbc = st.number_input(
            "White blood cell count (10^9/L)",
            min_value=0.0,
            max_value=50.0,
            value=10.0,
            step=0.1,
            help="Normal range: 4.0-10.0 × 10^9/L"
        )
        crp = st.number_input(
            "CRP (mg/L)",
            min_value=0.0,
            max_value=500.0,
            value=5.0,
            step=0.1
        )
        il6 = st.number_input(
            "IL6 (pg/mL)",
            min_value=0.0,
            max_value=1000.0,
            value=10.0,
            step=0.1
        )
        pct = st.number_input(
            "PCT (ng/mL)",
            min_value=0.0,
            max_value=100.0,
            value=0.1,
            step=0.01
        )
        
    with col2:
        st.subheader("Patient characteristics")
        # 是否老龄选择
        elderly = st.selectbox(
            "Elderly(≥65岁)",
            options=[("NO", 0), ("YES", 1)],
            format_func=lambda x: x[0]
        )[1]
        
        # 疾病分类选择
        disease_type = st.selectbox(
            "Disease typre",
            options=[
                ("Others", 0),
                ("Rspiratory disease", 1),
                ("Tumour", 2),
                ("Gynecological disease", 3),
                ("Orthopedics and Trauma", 4),
                ("CVD", 6),
                ("Nerve system disease", 7),
                ("Digestive system disease", 8),
                ("Metabolic diseases", 9),
                ("Urinary system disease", 10),
                ("Infectious disease", 11),
                ("Otorhinolaryngological diseases", 12),
                ("Ophthalmic Diseases", 13),
                ("Dermatological diseases", 14),
                ("Hematological diseases", 15),
                ("Rehabilitation", 16),
                ("ICU", 17),
                ("Gereology", 18),
                ("General medicine", 19),
                ("Traditional Chinese Medicine and General Medicine", 20)
            ],
            format_func=lambda x: x[0]
        )[1]
        
        # 发热状态
        fever_status = st.selectbox(
            "Fever status",
            options=[("NO", 0), ("YES", 1)],
            format_func=lambda x: x[0]
        )[1]
        
        # 细菌感染
        bacterial_infection = st.selectbox(
            "Bacterial infection",
            options=[("NO", 0), ("YES", 1)],
            format_func=lambda x: x[0]
        )[1]
    
    with col3:
        st.subheader("Medical intervention")
        # 抗菌药物使用相关
        antimicrobial_use = st.selectbox(
            "Antimicrobial use",
            options=[("NO", 0), ("YES", 1)],
            format_func=lambda x: x[0]
        )[1]
        
        restricted_antimicrobial = st.selectbox(
            "Restricted antimicrobial use",
            options=[("NO", 0), ("YES", 1)],
            format_func=lambda x: x[0]
        )[1]
        
        special_class_antimicrobial = st.selectbox(
            "Special-class antimicrobial use",
            options=[("NO", 0), ("YES", 1)],
            format_func=lambda x: x[0]
        )[1]
        
        combination_therapy = st.selectbox(
            "Combination antimicrobial therapy",
            options=[("NO", 0), ("YES", 1)],
            format_func=lambda x: x[0]
        )[1]
        
        # 导管相关
        urinary_catheter = st.selectbox(
            "Urinary catheterization",
            options=[("NO", 0), ("YES", 1)],
            format_func=lambda x: x[0]
        )[1]
        
        central_venous_catheter = st.selectbox(
            "Central venous catheter (CVC)",
            options=[("否", 0), ("是", 1)],
            format_func=lambda x: x[0]
        )[1]
    
    # 提交按钮
    submitted = st.form_submit_button("预测真菌感染风险")
    
    if submitted:
        # 创建输入数据框 - 确保顺序与模型训练时一致
        input_data = pd.DataFrame([[
            wbc, crp, il6, pct, elderly, disease_type, fever_status, 
            restricted_antimicrobial, urinary_catheter, 
            special_class_antimicrobial, antimicrobial_use, 
            bacterial_infection, combination_therapy, 
            central_venous_catheter
        ]], columns=[
            'WBC', 'CRP', 'IL6', 'PCT', 'Elderly', 'Disease type', 
            'Fever status', 'Restricted antimicrobial use', 
            'Urinary catheterization', 'Special-class antimicrobial use', 
            'Antimicrobial use', 'Bacterial infection', 
            'Combination antimicrobial therapy', 'Central venous catheter (CVC)'
        ])
        
        # 预测概率
        try:
            proba = model.predict_proba(input_data)[0][1]  # 获取真菌感染的概率
            
            # 显示结果
            st.subheader("Predict the outcome")
            
            # 使用进度条和指标显示概率
            col_res1, col_res2 = st.columns([1, 3])
            with col_res1:
                st.metric(label="Probability of fungal infection", value=f"{proba:.1%}")
            with col_res2:
                st.progress(proba)
            
            # 风险等级评估
            if proba > 0.5:
                st.error("High risk")
            elif proba > 0.2:
                st.warning("Concentration risk")
            else:
                st.success("Low risk")
            
            # 详细解释
            st.markdown("### 临床建议")
            if proba > 0.5:
                st.markdown("""
                - **立即行动**:
                  - 进行真菌培养、G试验和GM试验
                  - 考虑启动经验性抗真菌治疗
                  - 评估免疫状态和基础疾病
                
                - **监测**:
                  - 每日监测感染指标变化
                  - 评估导管使用必要性
                  - 审查抗菌药物使用方案
                """)
            elif proba > 0.2:
                st.markdown("""
                - **进一步检查**:
                  - 进行真菌相关实验室检查
                  - 评估免疫状态和基础疾病
                  - 定期复查感染指标
                
                - **预防措施**:
                  - 评估抗菌药物使用合理性
                  - 评估导管使用必要性
                  - 加强感染监测
                """)
            else:
                st.markdown("""
                - **常规管理**:
                  - 继续当前治疗方案
                  - 监测感染指标变化
                  - 如症状加重及时复查
                
                - **预防建议**:
                  - 合理使用抗菌药物
                  - 避免不必要的导管使用
                  - 加强基础护理
                """)
                
        except Exception as e:
            st.error(f"预测过程中发生错误: {str(e)}")

# 添加使用说明
st.markdown("---")
st.subheader("instructions")
st.markdown("""
1. Enter all the parameters of the patient in the form
2. The system will calculate and display the probability and risk level of fungal infection

**Parameter specification**:
- **Laboratory index**: The current laboratory test values of the patient
- **Patient characteristics**: The basic situation and disease information of the patient
- **Medical intervention**: The medical measures received by the patient and the use of drugs
""")

# 添加页脚
st.markdown("---")
st.caption("© Fungal infection prediction model")
