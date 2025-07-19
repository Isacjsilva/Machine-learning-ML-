# ======================================
# 📚 IMPORTANDO BIBLIOTECAS NECESSÁRIAS
# ======================================

import pandas as pd                 # Para manipulação de dados (DataFrame, leitura de CSV etc.)
import numpy as np                  # Para cálculos matemáticos (média, arrays, etc.)
import seaborn as sns               # Para gráficos estáticos (histograma, gráfico de barras, etc.)
import matplotlib.pyplot as plt     # Para visualização de gráficos
import plotly.express as px         # Para gráficos interativos (ex: scatter matrix)

# ======================================
# 📂 CARREGANDO OS DADOS
# ======================================

# Lê os dados de crédito a partir de um arquivo CSV
base_credit = pd.read_csv(r'C:\Users\isacj\OneDrive\Documentos\machine learning\credit_data (1).csv')

# ======================================
# 📊 ANÁLISE EXPLORATÓRIA DOS DADOS (EDA)
# ======================================

# OBS: Esta parte está comentada para evitar execução automática.
# Você pode descomentar para visualizar os dados com gráficos.

'''
# Verificando a distribuição dos valores da coluna 'default'
valores, contagens = np.unique(base_credit['default'], return_counts=True)
print("Valores únicos na coluna 'default':", valores)
print("Contagem de cada valor:", contagens)

# Gráfico de barras da variável alvo (default)
sns.countplot(x=base_credit['default'])
plt.title("Distribuição de inadimplência (default)")
plt.xlabel("Default (0 = não inadimplente, 1 = inadimplente)")
plt.ylabel("Contagem")
plt.show()

# Histogramas das variáveis numéricas
plt.hist(base_credit['age'], bins=20, edgecolor='black')
plt.title("Distribuição de Idade")
plt.xlabel("Idade")
plt.ylabel("Frequência")
plt.show()

plt.hist(base_credit['income'], bins=20, edgecolor='black')
plt.title("Distribuição de Renda")
plt.xlabel("Renda")
plt.ylabel("Frequência")
plt.show()

plt.hist(base_credit['loan'], bins=20, edgecolor='black')
plt.title("Distribuição de Empréstimo")
plt.xlabel("Valor do Empréstimo")
plt.ylabel("Frequência")
plt.show()

# Matriz de dispersão interativa entre variáveis
grafico = px.scatter_matrix(
    base_credit,
    dimensions=['age', 'income', 'loan'],
    color='default',
    title="Dispersão entre Idade, Renda e Empréstimo"
)
grafico.show()
'''

# ======================================
# 🧹 TRATAMENTO DE DADOS INCONSISTENTES
# ======================================

# Verifica se existem idades negativas (o que não faz sentido)
print("Índices com idade negativa:")
print(base_credit.loc[base_credit['age'] < 0].index)

# Calcula a média das idades válidas (idade > 0)
media_idade_valida = base_credit['age'][base_credit['age'] > 0].mean()
print(f"Média das idades válidas: {media_idade_valida:.2f} anos")

# Substitui as idades negativas pela média calculada
base_credit.loc[base_credit['age'] < 0, 'age'] = media_idade_valida

# ======================================
# 🧩 TRATAMENTO DE VALORES FALTANTES (NaN)
# ======================================

# Verifica quantos valores ausentes existem em cada coluna
print("\nValores ausentes por coluna:")
print(base_credit.isnull().sum())

# Mostra os registros com idade ausente
print("\nRegistros com idade ausente:")
print(base_credit.loc[pd.isnull(base_credit['age'])])

# Substitui os valores ausentes na idade pela média válida
base_credit['age'].fillna(media_idade_valida, inplace=True)

# Verifica novamente se ainda existem valores ausentes
print("\nValores ausentes após o preenchimento:")
print(base_credit.isnull().sum())

# Mostra registros corrigidos
print(base_credit.loc[base_credit['clientid'].isin([29, 31, 32])])

# ======================================
# 🤖 DIVISÃO DOS DADOS: ENTRADAS (X) E SAÍDAS (y)
# ======================================

# Entradas (features): idade, renda e valor do empréstimo
X_credit = base_credit.iloc[:, 1:4].values
print("\nEntradas (X):")
print(X_credit)

# Saída (target): inadimplente ou não (0 ou 1)
y_credit = base_credit.iloc[:, 4].values
print("\nSaídas (y):")
print(y_credit)

# ======================================
# ⚖️ ESCALONAMENTO (PADRONIZAÇÃO) DOS DADOS
# ======================================

# Verificando valores mínimo e máximo da variável "income" antes da padronização
print("\nExemplo antes da padronização (coluna 'income'):")
print("Mínimo:", X_credit[:, 1].min())
print("Máximo:", X_credit[:, 1].max())

# Importa o escalonador padrão (z-score): (x - média) / desvio padrão
from sklearn.preprocessing import StandardScaler
scaler_credit = StandardScaler()

# Aplica o escalonamento nas entradas
X_credit = scaler_credit.fit_transform(X_credit)

print("\nEntradas após padronização:")
print(X_credit)


from sklearn.model_selection import train_test_split

# Divisão dos dados em treino e teste (75% treino, 25% teste)
X_credit_treinamento, X_credit_teste, y_credit_treinamento, y_credit_teste = train_test_split(
    X_credit, y_credit, test_size=0.25, random_state=0
)

# Impressão das dimensões dos conjuntos
print("📐 Formato dos conjuntos de dados:")
print("X_credit_treinamento:", X_credit_treinamento.shape)
print("y_credit_treinamento:", y_credit_treinamento.shape)
print("X_credit_teste:", X_credit_teste.shape)
print("y_credit_teste:", y_credit_teste.shape)

import pickle

with open('credit.pkl', mode = 'wb') as f:
  pickle.dump([X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste], f)