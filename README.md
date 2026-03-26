---
title: Classificação da Qualidade Ambiental

# Classificação da Qualidade Ambiental

Este projeto utiliza modelos de **Machine Learning**, atualmente **SVM**, para classificar a qualidade ambiental com base em dados coletados de sensores e indicadores ambientais (**temperatura, pressão atmosférica, umidade, poluentes, entre outros**).

---

## Funcionalidades
- Pré-processamento completo:
  - Remoção de outliers por **IQR**;
  - Tratamento de valores nulos (mediana);
  - Normalização e codificação de variáveis categóricas;
  - Balanceamento de classes com mapeamento simplificado.
- Treinamento e avaliação do modelo **SVM**.
- Geração de métricas de desempenho detalhadas:
  - Accuracy, F1-score, Precision, Recall;
  - Matriz de confusão;
  - Curvas de aprendizado.
- Salvamento de artefatos:
  - Modelo treinado (`.pkl`);
  - **Scaler**, **LabelEncoder** e **OneHotEncoder**.
- Plotagem automática da matriz de confusão.

---

## Estrutura do Projeto
Projeto/
│── scripts/
│   ├── Analise_Modelos.py       # Comparação de modelos e métricas
│   ├── Treinamento.py           # Treinamento do modelo final
│   ├── Figuras/                 # Gráficos de análise exploratória e métricas
│   ├── Modelo/                  # Modelos e artefatos salvos
│   ├── Compara.py               # Funções para exibir métricas e escolher o melhor modelo
│   ├── Transform.py             # Funções de pré-processamento (remoção de outliers, tratamento de nulos)
│── app.py                       # Script para deploy
│── requirements.txt             # Dependências do projeto
│── README.txt                   # Este arquivo

---

## Instalação

Clone o repositório e crie um ambiente virtual:

git clone https://github.com/Rafael281292/air-quality-classifier
cd seu-projeto

# Criar ambiente virtual
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows

# Instalar dependências
pip install -r requirements.txt

---

## Treinamento do Modelo
Para rodar a análise e comparação de métricas:

python scripts/Analise_Modelos.py

Isso gera:

- gráficos da análise exploratória
- gráficos comparativos de métricas entre modelos;
- indicação do melhor modelo;
- arquivo CSV com resultados intermediários.

---

## Treinar apenas o modelo final

python scripts/Treinamento.py

Gera:

- Modelo/SVM/SVM_modelo_final.pkl
- Modelo/SVM_onehot_encoder.pkl
- Modelo/SVM_scaler.pkl
- Modelo/SVM_label_encoder.pkl
- gráficos de matriz de confusão e curva de aprendizado

---

## Avaliação e Métricas
As métricas calculadas incluem:

- **Accuracy (Treino, Validação, Teste)**
- **F1-score (Treino, Validação, Teste)**
- **Precision e Recall (Validação e Teste)**
- **Cross-validation F1 (Treino)**
- **Matriz de confusão**

---

## Como rodar o projeto

A sequência recomendada:

1. Pré-processar e analisar os dados:

python scripts/Analise_Modelos.py

2. Treinar o modelo final:

python scripts/Treinamento.py

3. Rodar inferência:

python app.py

O script `main.py` carregará o modelo final, encoder e scaler, permitindo que você faça previsões diretamente com novos dados ambientais.

---

## Artefatos gerados

Após o treinamento, os seguintes arquivos são salvos:

Modelo/
└── SVM/
    ├── SVM_modelo_final.pkl
    ├── SVM_onehot_encoder.pkl
    ├── SVM_scaler.pkl
    ├── SVM_label_encoder.pkl
Figuras/
└── Confusion_Matrix_SVM.png
└── Curva_Aprendizado_SVM.png
processed_data.csv

---

## Autor
Rafael de Oliveira Lima