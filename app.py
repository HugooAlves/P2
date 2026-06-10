
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_page_config(
    page_title='Credit Default Dashboard',
    layout='wide'
)

# ==========================
# CARREGAMENTO
# ==========================

@st.cache_data
def carregar_dados():
    return pd.read_csv('UCI_Credit_Card.csv')

df = carregar_dados()

modelo = joblib.load('modelo_final.joblib')
scaler = joblib.load('scaler.joblib')

# ==========================
# FEATURE ENGINEERING
# ==========================

df['TOTAL_BILL'] = sum(df[f'BILL_AMT{i}'] for i in range(1,7))
df['TOTAL_PAY'] = sum(df[f'PAY_AMT{i}'] for i in range(1,7))
df['PAY_RATIO'] = df['TOTAL_PAY']/(df['TOTAL_BILL']+1)

# ==========================
# MENU
# ==========================

st.title('💳 Credit Default Prediction Dashboard')

aba1, aba2, aba3 = st.tabs([
    'Dashboard',
    'Análise',
    'Predição'
])

# ==========================
# DASHBOARD
# ==========================

with aba1:

    c1,c2,c3 = st.columns(3)

    c1.metric(
        'Registros',
        len(df)
    )

    c2.metric(
        'Variáveis',
        df.shape[1]
    )

    c3.metric(
        'Default (%)',
        round(
            df['default.payment.next.month'].mean()*100,
            2
        )
    )

    st.subheader('Distribuição de Inadimplência')

    fig, ax = plt.subplots()

    sns.countplot(
        x='default.payment.next.month',
        data=df,
        ax=ax
    )

    st.pyplot(fig)

# ==========================
# ANÁLISE
# ==========================

with aba2:

    st.subheader('Heatmap de Correlação')

    fig, ax = plt.subplots(
        figsize=(12,8)
    )

    sns.heatmap(
        df.corr(numeric_only=True),
        cmap='coolwarm',
        ax=ax
    )

    st.pyplot(fig)

# ==========================
# PREDIÇÃO
# ==========================

with aba3:

    st.subheader('Previsão de Inadimplência')

    LIMIT_BAL = st.number_input(
        'LIMIT_BAL',
        value=50000
    )

    SEX = st.selectbox(
        'SEX',
        [1,2]
    )

    EDUCATION = st.selectbox(
        'EDUCATION',
        [1,2,3,4]
    )

    MARRIAGE = st.selectbox(
        'MARRIAGE',
        [1,2,3]
    )

    AGE = st.number_input(
        'AGE',
        value=30
    )

    PAY_0 = st.slider('PAY_0', -2, 8, 0)
    PAY_2 = st.slider('PAY_2', -2, 8, 0)
    PAY_3 = st.slider('PAY_3', -2, 8, 0)
    PAY_4 = st.slider('PAY_4', -2, 8, 0)
    PAY_5 = st.slider('PAY_5', -2, 8, 0)
    PAY_6 = st.slider('PAY_6', -2, 8, 0)

    BILL_AMT1 = st.number_input('BILL_AMT1', value=0)
    BILL_AMT2 = st.number_input('BILL_AMT2', value=0)
    BILL_AMT3 = st.number_input('BILL_AMT3', value=0)
    BILL_AMT4 = st.number_input('BILL_AMT4', value=0)
    BILL_AMT5 = st.number_input('BILL_AMT5', value=0)
    BILL_AMT6 = st.number_input('BILL_AMT6', value=0)

    PAY_AMT1 = st.number_input('PAY_AMT1', value=0)
    PAY_AMT2 = st.number_input('PAY_AMT2', value=0)
    PAY_AMT3 = st.number_input('PAY_AMT3', value=0)
    PAY_AMT4 = st.number_input('PAY_AMT4', value=0)
    PAY_AMT5 = st.number_input('PAY_AMT5', value=0)
    PAY_AMT6 = st.number_input('PAY_AMT6', value=0)

    if st.button('Prever'):

        TOTAL_BILL = (
            BILL_AMT1 + BILL_AMT2 + BILL_AMT3 +
            BILL_AMT4 + BILL_AMT5 + BILL_AMT6
        )

        TOTAL_PAY = (
            PAY_AMT1 + PAY_AMT2 + PAY_AMT3 +
            PAY_AMT4 + PAY_AMT5 + PAY_AMT6
        )

        PAY_RATIO = TOTAL_PAY / (TOTAL_BILL + 1)

        entrada = pd.DataFrame([[
            LIMIT_BAL,
            SEX,
            EDUCATION,
            MARRIAGE,
            AGE,
            PAY_0,
            PAY_2,
            PAY_3,
            PAY_4,
            PAY_5,
            PAY_6,
            BILL_AMT1,
            BILL_AMT2,
            BILL_AMT3,
            BILL_AMT4,
            BILL_AMT5,
            BILL_AMT6,
            PAY_AMT1,
            PAY_AMT2,
            PAY_AMT3,
            PAY_AMT4,
            PAY_AMT5,
            PAY_AMT6,
            TOTAL_BILL,
            TOTAL_PAY,
            PAY_RATIO
        ]])

        entrada.columns = [
            'LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE',
            'PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6',
            'BILL_AMT1','BILL_AMT2','BILL_AMT3',
            'BILL_AMT4','BILL_AMT5','BILL_AMT6',
            'PAY_AMT1','PAY_AMT2','PAY_AMT3',
            'PAY_AMT4','PAY_AMT5','PAY_AMT6',
            'TOTAL_BILL','TOTAL_PAY','PAY_RATIO'
        ]

        num_cols = [
            'LIMIT_BAL',
            'BILL_AMT1','BILL_AMT2','BILL_AMT3',
            'BILL_AMT4','BILL_AMT5','BILL_AMT6',
            'PAY_AMT1','PAY_AMT2','PAY_AMT3',
            'PAY_AMT4','PAY_AMT5','PAY_AMT6',
            'TOTAL_BILL','TOTAL_PAY','PAY_RATIO'
        ]

        entrada[num_cols] = scaler.transform(
            entrada[num_cols]
        )

        prob = modelo.predict_proba(
            entrada
        )[0][1]

        st.metric(
            'Probabilidade de Inadimplência',
            f'{prob*100:.2f}%'
        )

        if prob > 0.5:
            st.error('ALTO RISCO')
        else:
            st.success('BAIXO RISCO')

