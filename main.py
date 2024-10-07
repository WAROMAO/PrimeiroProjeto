import pandas as pd
import streamlit as st
from pandas.core.computation.ops import isnumeric
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

# Carregar os dados
df = pd.read_csv("pizzas.csv")

# Separar features e target
X = df[["diametro"]]
y = df["preco"]

# Criar um imputer para preencher NaNs com a média
imputer = SimpleImputer(strategy='mean')

# Transformar os dados para preencher NaNs
X_imputed = imputer.fit_transform(X)

# Criar e treinar o modelo
modelo = LinearRegression()
modelo.fit(X_imputed, y)

# Interface do Streamlit
st.title('Aplicação web para aprendizado de machine learning com python')
st.divider()

numero = st.number_input('Entre com um nr de qualquer tamanho de diametro da Pizza:')
                         #'Entre com um nr do tamanho da Pizza')

if numero:
    numero_previsto = modelo.predict([[numero]])[0]  # Correção aqui
    st.write(f'O valor calculado com base no número = {int(numero)}\n'
             f' digitado é: {numero_previsto:.2f}')
    st.balloons()