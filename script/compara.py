from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
import json


def exibir_resultado_classificacao(nome, modelo, hiperparams,
                                   X_train, y_train,
                                   X_val, y_val,
                                   X_test, y_test):
    """
    Avalia um modelo de classificação em treino, validação e teste.
    Retorna dicionário de métricas.
    """

    # Predições
    y_pred_train = modelo.predict(X_train)
    y_pred_val   = modelo.predict(X_val)
    y_pred_test  = modelo.predict(X_test)

    # =========================
    # MÉTRICAS
    # =========================
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_val   = accuracy_score(y_val, y_pred_val)
    acc_test  = accuracy_score(y_test, y_pred_test)

    f1_train = f1_score(y_train, y_pred_train, average="weighted")
    f1_val   = f1_score(y_val, y_pred_val, average="weighted")
    f1_test  = f1_score(y_test, y_pred_test, average="weighted")

    precision_val = precision_score(y_val, y_pred_val, average="weighted")
    recall_val    = recall_score(y_val, y_pred_val, average="weighted")

    # Cross-validation
    cv_scores = cross_val_score(
        modelo, X_train, y_train,
        cv=5,
        scoring="f1_weighted",
        n_jobs=-1
    )
    f1_cv = cv_scores.mean()

    # =========================
    # LOG
    # =========================
    print(f"\n{nome}")

    print("Treino:")
    print(f" - Accuracy: {acc_train:.3f}")
    print(f" - F1:       {f1_train:.3f}")

    print("Validação:")
    print(f" - Accuracy:  {acc_val:.3f}")
    print(f" - F1:        {f1_val:.3f}")
    print(f" - Precision: {precision_val:.3f}")
    print(f" - Recall:    {recall_val:.3f}")

    print("Teste:")
    print(f" - Accuracy: {acc_test:.3f}")
    print(f" - F1:       {f1_test:.3f}")

    print("Cross-validation (treino):")
    print(f" - F1 CV: {f1_cv:.3f}")

    print(f" - Hiperparâmetros: {hiperparams}")

    # =========================
    # RETORNO
    # =========================
    return {
        "acc_train": acc_train,
        "acc_val": acc_val,
        "acc_test": acc_test,
        "f1_train": f1_train,
        "f1_val": f1_val,
        "f1_test": f1_test,
        "precision_val": precision_val,
        "recall_val": recall_val,
        "f1_cv": f1_cv,
        "params": hiperparams
    }


# ===============================
# ESCOLHA MULTICRITÉRIO
# ===============================
def escolher_melhor_modelo_multicriterio(df_resultados):
    """
    Ranking baseado em:
    - f1_val (maior melhor)
    - accuracy_val (maior melhor)
    - f1_cv (maior melhor)
    """

    df_rank = pd.DataFrame(index=df_resultados.index)

    # Rankings
    df_rank["rank_f1_val"] = df_resultados["f1_val"].rank(ascending=False)
    df_rank["rank_acc_val"] = df_resultados["acc_val"].rank(ascending=False)
    df_rank["rank_f1_cv"] = df_resultados["f1_cv"].rank(ascending=False)

    # Score total
    df_rank["score_total"] = (
        df_rank["rank_f1_val"] +
        df_rank["rank_acc_val"] +
        df_rank["rank_f1_cv"]
    )

    # Melhor modelo
    melhor = df_rank["score_total"].idxmin()

    print("\n🏆 Ranking dos modelos:")
    print(df_rank.sort_values("score_total"))

    print(f"\n🥇 Melhor modelo: {melhor}")
    print(json.dumps(df_resultados.loc[melhor]["params"], indent=4))

    return melhor, df_rank, df_resultados.loc[melhor]