# =======================================
# 📦 Importação de bibliotecas
# =======================================
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from yellowbrick.classifier import ConfusionMatrix

# =======================================
# 📂 Carregando os dados com pickle
# =======================================
with open('census.pkl', 'rb') as f:
    X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste = pickle.load(f)

# =======================================
# 🤖 Treinamento do modelo Naive Bayes
# =======================================
naive_credit_data = GaussianNB()
naive_credit_data.fit(X_credit_treinamento, y_credit_treinamento)

# =======================================
# 📈 Realizando previsões
# =======================================
previsoes = naive_credit_data.predict(X_credit_teste)
print(previsoes)

# =======================================
# 📊 Avaliação do modelo
# =======================================
print("Acurácia:", accuracy_score(y_credit_teste, previsoes))
print("Matriz de Confusão:")
print(confusion_matrix(y_credit_teste, previsoes))
print("\nRelatório de Classificação:")
print(classification_report(y_credit_teste, previsoes))

# =======================================
# 🎨 Visualização da Matriz de Confusão
# =======================================
cm = ConfusionMatrix(naive_credit_data)
cm.fit(X_credit_treinamento, y_credit_treinamento)
cm.score(X_credit_teste, y_credit_teste)
cm.show()
