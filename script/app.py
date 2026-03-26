import os
import gradio as gr
import joblib
import pandas as pd

# =========================================================
# ESCOLHA DO MODELO
# =========================================================
MODELO_ESCOLHIDO = "SVM"  # Altere para o modelo que você treinou

# =========================================================
# CARREGAR MODELO, ENCODER, SCALER E LABEL ENCODER
# =========================================================
def carregar_artefatos():
    base_dir = os.path.dirname(os.path.abspath(__file__))  # pasta do script atual
    pasta_modelo = os.path.join(base_dir, "Modelo", MODELO_ESCOLHIDO)
    os.makedirs(pasta_modelo, exist_ok=True)  # garante que a pasta existe

    caminho_modelo = os.path.join(pasta_modelo, f"{MODELO_ESCOLHIDO}_modelo_final.pkl")
    caminho_encoder = os.path.join(pasta_modelo, f"{MODELO_ESCOLHIDO}_onehot_encoder.pkl")
    caminho_scaler = os.path.join(pasta_modelo, f"{MODELO_ESCOLHIDO}_scaler.pkl")
    caminho_label = os.path.join(pasta_modelo, f"{MODELO_ESCOLHIDO}_label_encoder.pkl")

    # Checagem de arquivos
    if not os.path.exists(caminho_modelo):
        raise FileNotFoundError(f"Modelo não encontrado: {caminho_modelo}")
    if not os.path.exists(caminho_encoder):
        raise FileNotFoundError(f"Encoder não encontrado: {caminho_encoder}")
    if not os.path.exists(caminho_label):
        raise FileNotFoundError(f"LabelEncoder não encontrado: {caminho_label}")

    # Carrega arquivos
    modelo = joblib.load(caminho_modelo)
    encoder = joblib.load(caminho_encoder)
    scaler = joblib.load(caminho_scaler) if os.path.exists(caminho_scaler) else None
    label_encoder = joblib.load(caminho_label)

    return modelo, encoder, scaler, label_encoder

# =========================================================
# Carrega artefatos
# =========================================================
modelo, encoder, scaler, label_encoder = carregar_artefatos()

# =========================================================
# FUNÇÃO DE PREVISÃO
# =========================================================
def prever(Temperatura, Umidade, CO2, CO, Pressao_Atm, NO2, SO2, O3):
    dados = pd.DataFrame([{
        "Temperatura": float(Temperatura),
        "Umidade": float(Umidade),
        "CO2": float(CO2),
        "CO": float(CO),
        "Pressao_Atm": float(Pressao_Atm),
        "NO2": float(NO2),
        "SO2": float(SO2),
        "O3": float(O3)
    }])

    # Aplica encoder se houver colunas categóricas
    cat_cols = [col for col in dados.columns if str(dados[col].dtype) == "object"]
    if cat_cols:
        dados_cat = encoder.transform(dados[cat_cols])
        if hasattr(dados_cat, "toarray"):
            dados_cat = dados_cat.toarray()
        dados_cat = pd.DataFrame(dados_cat, columns=encoder.get_feature_names_out(cat_cols), index=dados.index)
        num_cols = [c for c in dados.columns if c not in cat_cols]
        dados_final = pd.concat([dados[num_cols], dados_cat], axis=1)
    else:
        dados_final = dados.copy()

    # Garante colunas iguais ao treino
    if hasattr(modelo, "feature_names_in_"):
        dados_final = dados_final.reindex(columns=modelo.feature_names_in_, fill_value=0)

    # Aplica scaler
    if scaler is not None:
        dados_final = pd.DataFrame(
            scaler.transform(dados_final),
            columns=dados_final.columns,
            index=dados_final.index
        )

    # Predição
    pred_num = modelo.predict(dados_final)[0]

    # Converte número para rótulo legível
    pred_classe = label_encoder.inverse_transform([pred_num])[0]
    
    # Textos explicativos para cada classe de qualidade do ar
    TEXTOS_CLASSES = {
    "Boa": "Ar saudável. Pode-se realizar atividades ao ar livre sem restrições.",
    "Moderada": "Ar com qualidade aceitável. Pessoas sensíveis devem reduzir esforço físico intenso.",
    "Ruim": "Ar poluído. Evite exercícios ao ar livre e proteja grupos sensíveis.",
    "Muito Ruim": "Ar muito poluído. Evite sair de casa e siga recomendações médicas.",
    "Péssima": "Ar extremamente poluído. Tome todas as precauções e evite exposição."
}
    texto_explicativo = TEXTOS_CLASSES.get(pred_classe, "")
    return f"🌿 Previsão de Qualidade Ambiental: {pred_classe}\n📖 Informação: {texto_explicativo}"

# =========================================================
# INTERFACE GRADIO
# =========================================================
inputs = [
    gr.Number(label="Temperatura (°C)"),
    gr.Number(label="Umidade (%)"),
    gr.Number(label="CO2 (ppm)"),
    gr.Number(label="CO (ppm)"),
    gr.Number(label="Pressão Atmosférica (hPa)"),
    gr.Number(label="NO2 (ppm)"),
    gr.Number(label="SO2 (ppm)"),
    gr.Number(label="O3 (ppm)")
]

output = gr.Textbox(label="Qualidade Ambiental Prevista", lines=2)

css = """
* { font-size: 18px !important; font-family: Arial, sans-serif; }
h1, h2, h3 { font-size: 28px !important; color: #2e7d32; }
button { 
    font-size: 18px !important; 
    background-color: #4caf50 !important; 
    color: white !important; 
    border-radius: 8px !important;
    padding: 10px 20px !important;
}
label { font-weight: bold; color: #1b5e20; }
.gr-number input { border: 1px solid #4caf50 !important; border-radius: 6px !important; padding: 5px; }
"""

app = gr.Interface(
    fn=prever,
    inputs=inputs,
    outputs=output,
    title="🌿 Previsão de Qualidade Ambiental",
    description="Insira os valores de poluentes e parâmetros ambientais para prever a qualidade do ar. Este conteúdo é destinado apenas para fins educacionais. Os dados exibidos são ilustrativos e podem não corresponder a situações reais.",
    css=css,
    theme="grass"
)

if __name__ == "__main__":
    app.launch()