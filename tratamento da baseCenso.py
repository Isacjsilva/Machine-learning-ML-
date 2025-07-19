# ====================================================
# 📦 Importação de bibliotecas
# ====================================================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
import pickle

# ====================================================
# 📂 Leitura do dataset
# ====================================================
base_census = pd.read_csv("census.csv")

# Exibe as primeiras linhas
print("📄 Primeiras linhas do dataset:")
print(base_census.head())

# ====================================================
# 📊 Estatísticas e valores faltantes
# ====================================================
print("\n📊 Estatísticas descritivas:")
print(base_census.describe(include='all'))

print("\n❗ Valores faltantes:")
print(base_census.isnull().sum())

# ====================================================
# 📈 Visualização (comente/descomente se desejar)
# ====================================================
"""
np.unique(base_census['income'], return_counts=True)
sns.countplot(x=base_census['income'])
plt.title('Distribuição de Renda')
plt.xlabel('Faixa de Renda')
plt.ylabel('Quantidade')
plt.tight_layout()
plt.show()

plt.hist(base_census['age'], bins=20)
plt.title("Distribuição de Idade")
plt.show()

plt.hist(base_census['education-num'], bins=20)
plt.title("Distribuição da Escolaridade Numérica")
plt.show()

grafico = px.treemap(base_census, path=['workclass', 'age'])
grafico.show()

grafico = px.parallel_categories(base_census, dimensions=['workclass', 'occupation', 'income'])
grafico.show()
"""

# ====================================================
# ✂️ Separação entre X e y
# ====================================================
X_census = base_census.iloc[:, 0:14].values
y_census = base_census.iloc[:, 14].values

# Corrigindo erro de digitação em print anterior (era "Y_census")
print("\n🎯 Amostra de y_census:")
print(y_census[:5])

# ====================================================
# 🔤 Codificação com LabelEncoder (para OneHotEncoder)
# ====================================================
label_encoders = {
    'workclass': LabelEncoder(),
    'education': LabelEncoder(),
    'marital-status': LabelEncoder(),
    'occupation': LabelEncoder(),
    'relationship': LabelEncoder(),
    'race': LabelEncoder(),
    'sex': LabelEncoder(),
    'native-country': LabelEncoder()
}

# Aplica LabelEncoder nas colunas categóricas
for i, coluna in zip([1, 3, 5, 6, 7, 8, 9, 13], label_encoders):
    X_census[:, i] = label_encoders[coluna].fit_transform(X_census[:, i])

# ====================================================
# 🔢 OneHotEncoder (variáveis categóricas)
# ====================================================
onehotencoder = ColumnTransformer(
    transformers=[('OneHot', OneHotEncoder(), [1, 3, 5, 6, 7, 8, 9, 13])],
    remainder='passthrough'
)
X_census = onehotencoder.fit_transform(X_census).toarray()

print("\n✅ OneHotEncoder aplicado. Novo shape de X_census:", X_census.shape)

# ====================================================
# 📏 Escalonamento
# ====================================================
scaler = StandardScaler()
X_census = scaler.fit_transform(X_census)

print("\n✅ Dados escalonados. Exemplo de X[0]:")
print(X_census[0])

# ====================================================
# 🔀 Divisão treino/teste
# ====================================================
X_census_treinamento, X_census_teste, y_census_treinamento, y_census_teste = train_test_split(
    X_census, y_census, test_size=0.15, random_state=0
)

print("\n📐 Formato dos conjuntos:")
print("X_census_treinamento:", X_census_treinamento.shape)
print("y_census_treinamento:", y_census_treinamento.shape)
print("X_census_teste:", X_census_teste.shape)
print("y_census_teste:", y_census_teste.shape)

# ====================================================
# 💾 Salvando os dados com pickle
# ====================================================
with open('census.pkl', mode='wb') as f:
    pickle.dump([X_census_treinamento, y_census_treinamento, X_census_teste, y_census_teste], f)

print("\n✅ Dados salvos com sucesso em 'census.pkl'")
