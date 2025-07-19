# 🤖 Curso Introdutório de Machine Learning - Projeto 1: Tratamento de Dados

Este repositório faz parte do meu aprendizado no **curso introdutório de Machine Learning**, onde estou construindo passo a passo os fundamentos dessa área incrível da Inteligência Artificial. Este primeiro projeto tem foco na **etapa de preparação e tratamento de dados**, essencial para o sucesso de qualquer modelo de machine learning.

---

## 📚 O que aprendi até aqui

Este projeto abordou os seguintes conceitos práticos:

### 1. 🔍 Análise Exploratória (EDA)
- Carregamento de bases de dados reais (`census.csv` e `credit_data.csv`)
- Exibição de estatísticas e verificação de valores faltantes
- Visualizações com `matplotlib`, `seaborn` e `plotly`

### 2. 🧹 Tratamento de Dados
- Substituição de valores inconsistentes (como idades negativas)
- Preenchimento de valores ausentes com médias
- Codificação de variáveis categóricas:
  - `LabelEncoder` para transformar rótulos em números
  - `OneHotEncoder` para evitar ordens falsas
- Padronização de variáveis com `StandardScaler`

### 3. 📂 Separação dos Dados
- Divisão entre variáveis de entrada (X) e saída (y)
- Separação dos dados em **conjuntos de treino e teste** com `train_test_split`

### 4. 💾 Salvamento de dados processados
- Serialização dos dados tratados usando `pickle` para uso futuro em modelos

---

## 📁 Estrutura dos Arquivos
├── credit_data.csv
├── census.csv
├── tratamento_credit.py
├── tratamento_census.py
├── credit.pkl
├── census.pkl
└── README.md


---

## 🛠 Tecnologias e Bibliotecas Usadas

- **Linguagem**: Python 3.11  
- **Bibliotecas**:
  - `pandas`, `numpy`
  - `matplotlib`, `seaborn`, `plotly.express`
  - `sklearn.preprocessing`, `sklearn.model_selection`, `sklearn.compose`
  - `pickle` para salvar os dados tratados

---

## 🚀 Próximos Passos no Curso

Agora que finalizei a etapa de pré-processamento, o próximo módulo do curso será:

- **Construção de modelos supervisionados**
- Avaliação com métricas como accuracy, precision, recall e F1-score
- Comparação entre algoritmos como Decision Tree, Naive Bayes, KNN, etc.

---

## 👨‍🎓 Sobre mim

Graduando em Engenharia de Controle e Automação pela Universidade Federal de Lavras (UFLA), com experiência em projetos acadêmicos, pesquisa e extensão. Tenho interesse nas áreas de inteligência artificial, automação e desenvolvimento de sistemas, buscando constantemente aprimorar meus conhecimentos e aplicá-los em soluções inovadoras. Possuo habilidades em C/C++ e Python, e atualmente estudo machine learning, MATLAB.

Participei de um projeto de extensão no Centro de Inovação em Automação e Inteligência Artificial (AIA), atuando na área de marketing e ministrando cursos de Arduino para crianças de escolas municipais. Também integrei o projeto Jovens Makers, vinculado ao Departamento de Física da UFLA, ensinando Arduino para crianças do CEDET, em Lavras-MG.

Fui representante discente no colegiado do Departamento de Automática da UFLA, colaborando nas decisões acadêmicas do curso e representando os interesses dos estudantes. Atualmente, sou bolsista da EPAMIG/FAPEMIG, desenvolvendo uma pesquisa voltada à criação de um sensor para medir o potencial hídrico do café, unindo tecnologia e inovação para otimizar processos na agricultura.

Busco constantemente novos desafios para expandir minhas habilidades técnicas e contribuir para o avanço da tecnologia e da automação.
---
