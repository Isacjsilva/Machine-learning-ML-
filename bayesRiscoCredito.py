# ====================================================
# 📊 Teorema de Naive Bayes (Abordagem Probabilística)
# ====================================================

# 📦 Importação de bibliotecas
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
import pickle

# ======================================
# 📂 Carregando a base de dados
# ======================================
base_risco_credito = pd.read_csv('risco_credito.csv')

# Visualizando os dados originais
print("🔍 Base de dados original:")
print(base_risco_credito)

# ======================================
# 🧼 Pré-processamento dos dados
# ======================================

# Separando os atributos (X) e a classe (y)
X_risco_credito = base_risco_credito.iloc[:, 0:4].values  # colunas: história, dívida, garantia, renda
Y_risco_credito = base_risco_credito.iloc[:, 4].values    # coluna: risco

print("\n📌 Atributos antes da codificação:")
print(X_risco_credito)

# Codificando atributos categóricos (transformando em números)
label_encoder_historia = LabelEncoder()
label_encoder_divida = LabelEncoder()
label_encoder_garantia = LabelEncoder()
label_encoder_renda = LabelEncoder()

X_risco_credito[:, 0] = label_encoder_historia.fit_transform(X_risco_credito[:, 0])
X_risco_credito[:, 1] = label_encoder_divida.fit_transform(X_risco_credito[:, 1])
X_risco_credito[:, 2] = label_encoder_garantia.fit_transform(X_risco_credito[:, 2])
X_risco_credito[:, 3] = label_encoder_renda.fit_transform(X_risco_credito[:, 3])

print("\n✅ Atributos após a codificação (LabelEncoder):")
print(X_risco_credito)

# ======================================
# 💾 Salvando os dados pré-processados
# ======================================
with open('risco_credito.pkl', 'wb') as f:
    pickle.dump([X_risco_credito, Y_risco_credito], f)

# ======================================
# 🧠 Treinando o classificador Naive Bayes
# ======================================
naive_risco_credito = GaussianNB()
naive_risco_credito.fit(X_risco_credito, Y_risco_credito)

# ======================================
# 🔮 Fazendo previsões
# ======================================
# Exemplo 1: história boa, dívida alta, sem garantias, renda > 35
# Exemplo 2: história ruim, dívida alta, garantia adequada, renda < 15
previsao = naive_risco_credito.predict([[0, 0, 1, 2], [2, 0, 0, 0]])

print("\n🧾 Previsões:")
print(previsao)

# ======================================
# 📈 Informações sobre o modelo treinado
# ======================================
print("\n📚 Classes possíveis:")
print(naive_risco_credito.classes_)

print("\n📊 Número de exemplos por classe:")
print(naive_risco_credito.class_count_)

print("\n📊 Probabilidades a priori de cada classe:")
print(naive_risco_credito.class_prior_)
