import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, ParameterGrid, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.svm import SVC
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
import joblib
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from transform import preprocessing_pipeline
from compara import exibir_resultado_classificacao, escolher_melhor_modelo_multicriterio

import matplotlib
matplotlib.use("Agg")

# ===============================
# CONFIGURAÇÃO
# ===============================
MLRUNS_DIR = os.path.abspath("mlruns")
mlflow.set_tracking_uri(f"file:///{MLRUNS_DIR.replace(os.sep, '/')}")
base_dir = os.path.dirname(os.path.abspath(__file__))
pasta = os.path.join(base_dir, "Figuras")
os.makedirs("Figuras", exist_ok=True)

# ===============================
# FUNÇÕES
# ===============================
def carregar_dados():
    df = pd.read_csv('C:/Users/rafae/OneDrive/Desktop/Desafio_final_IA/dataset_ambiental.csv')
    df.columns = df.columns.str.strip()
    print("Dados carregados:", df.shape)
    return df

def codificar_classes(y):
    le = LabelEncoder()
    y_num = le.fit_transform(y)
    return y_num, le

def dividir_dados(df):
    X = df.drop(columns=["Qualidade_Ambiental"])
    y = df["Qualidade_Ambiental"]
    y_num, le = codificar_classes(y)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y_num, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test, le

def codificar(X_train, X_val, X_test):
    categorical = X_train.select_dtypes(include="object").columns
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(X_train[categorical])

    def transform(X):
        if len(categorical) == 0:
            return X.copy()
        cat = encoder.transform(X[categorical])
        num = X.drop(columns=categorical)
        return pd.concat([num.reset_index(drop=True), pd.DataFrame(cat)], axis=1)

    return transform(X_train), transform(X_val), transform(X_test), encoder

def normalizar(X_train, X_val, X_test):
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_val), scaler.transform(X_test), scaler

def treinar_modelo(nome, modelo, grid, X_train, y_train):
    pipe = Pipeline([
        ("under", RandomUnderSampler(random_state=42)),
        ("model", modelo)
    ])
    resultados = []
    for params in ParameterGrid(grid):
        model = pipe.set_params(**params)
        try:
            score = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1).mean()
        except Exception as e:
            print(f"Erro com parâmetros {params}: {e}")
            continue
        resultados.append({**params, "acc_cv": score})
    df_res = pd.DataFrame(resultados).sort_values("acc_cv", ascending=False)
    best_params = {k: v for k, v in df_res.iloc[0].to_dict().items() if "model__" in k}
    modelo_final = pipe.set_params(**best_params)
    modelo_final.fit(X_train, y_train)
    return modelo_final, best_params

def salvar_curva_aprendizado(modelo, X, y, nome):
    train_sizes, train_scores, val_scores = learning_curve(
        modelo, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1
    )
    plt.figure(figsize=(6,4))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label="Train")
    plt.plot(train_sizes, np.mean(val_scores, axis=1), 'o-', label="Validation")
    plt.title(f"Curva de Aprendizado - {nome}")
    plt.xlabel("Número de amostras")
    plt.ylabel("Acurácia")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{pasta}/Curva_Aprendizado_{nome}.png", dpi=300)
    plt.close()

def avaliar_modelo_final_classificacao(modelo, X_final, y_final, X_test, y_test, modelo_nome):
    modelo.fit(X_final, y_final)
    y_pred_train = modelo.predict(X_final)
    y_pred_test = modelo.predict(X_test)
    metricas = {
        "acc_train": accuracy_score(y_final, y_pred_train),
        "f1_train": f1_score(y_final, y_pred_train, average="weighted"),
        "acc_test": accuracy_score(y_test, y_pred_test),
        "f1_test": f1_score(y_test, y_pred_test, average="weighted"),
        "precision_test": precision_score(y_test, y_pred_test, average="weighted"),
        "recall_test": recall_score(y_test, y_pred_test, average="weighted"),
        "confusion_matrix": confusion_matrix(y_test, y_pred_test)
    }
    return metricas

# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    df = carregar_dados()
    df = preprocessing_pipeline(df)

    X_train, X_val, X_test, y_train, y_val, y_test, le = dividir_dados(df)
    X_train, X_val, X_test, encoder = codificar(X_train, X_val, X_test)
    X_train_s, X_val_s, X_test_s, scaler = normalizar(X_train, X_val, X_test)

    # ===============================
    # TREINAR APENAS SVM
    # ===============================
    grid = {
        "model__C": [100],
        "model__kernel": ["linear"],
        "model__gamma": ["scale"]
    }
    modelo_name = "SVM"
    modelo, params = treinar_modelo(modelo_name, SVC(), grid, X_train_s, y_train)

    X_final = X_train_s
    y_final = y_train
    X_test_model = X_test_s
    scaler_to_save = scaler

    # =========================================================
    # Segurança
    # =========================================================
    if modelo is None or modelo_name is None:
        raise ValueError("Nenhum modelo foi selecionado. Descomente exatamente um bloco de modelo.")

    # =========================================================
    # Avaliação final
    # =========================================================
    metricas = avaliar_modelo_final_classificacao(
        modelo=modelo,
        X_final=X_final,
        y_final=y_final,
        X_test=X_test_model,
        y_test=y_test,
        modelo_nome=modelo_name
    )

    print("\nMétricas finais:")
    for k, v in metricas.items():
        if k != "confusion_matrix":
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}:\n{v}")

    print(f"\nHiperparâmetros usados: {params}")

    # =========================================================
    # Salvar modelo e artefatos (Windows-safe)
    # =========================================================
    modelo_dir = os.path.join("Modelo", modelo_name)
    os.makedirs(modelo_dir, exist_ok=True)

    modelo_path = os.path.join(modelo_dir, f"{modelo_name}_modelo_final.pkl")
    encoder_path = os.path.join(modelo_dir, f"{modelo_name}_onehot_encoder.pkl")
    scaler_path = os.path.join(modelo_dir, f"{modelo_name}_scaler.pkl")

    joblib.dump(modelo, modelo_path)
    joblib.dump(encoder, encoder_path)
    joblib.dump(scaler_to_save, scaler_path)

    print(f"\nModelo salvo em: {modelo_path}")
    print(f"Encoder salvo em: {encoder_path}")
    print(f"Scaler salvo em: {scaler_path}")

    # =========================================================
    # Plotar matriz de confusão
    # =========================================================
    cm = metricas["confusion_matrix"]
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Matriz de Confusão - {modelo_name}")
    plt.xlabel("Predito")
    plt.ylabel("Verdadeiro")
    plt.tight_layout()
    cm_path = os.path.join(modelo_dir, f"{modelo_name}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Matriz de confusão salva em: {cm_path}")