# =====================================================================
# preparar_bases.py
# Converte os arquivos ORIGINAIS baixados do UCI (ver FONTES_E_LICENCAS.md)
# em CSVs compatíveis com o pipeline.py, dentro da pasta dados/.
#
# Uso (coloque os arquivos baixados em dados_brutos/):
#   python preparar_bases.py
# =====================================================================
import csv
import os
import pandas as pd

BRUTOS, SAIDA = "dados_brutos", "dados"
os.makedirs(SAIDA, exist_ok=True)


# ---- 1) SMS Spam Collection (texto). Arquivo: SMSSpamCollection (separado por TAB, sem cabeçalho)
def sms_spam():
    arq = os.path.join(BRUTOS, "SMSSpamCollection")
    df = pd.read_csv(arq, sep="\t", header=None, names=["label", "mensagem"],
                     quoting=csv.QUOTE_NONE, encoding="utf-8", encoding_errors="replace")
    df = df.dropna().drop_duplicates()  # remove mensagens duplicadas (evita o mesmo texto em treino e teste)
    df.to_csv(os.path.join(SAIDA, "spam_sms.csv"), index=False)
    print("spam_sms.csv:", df.shape, df["label"].value_counts().to_dict())


# ---- 2) Default of Credit Card Clients (tabular). Arquivo: default of credit card clients.xls
def credito_cartao():
    arq = os.path.join(BRUTOS, "default of credit card clients.xls")
    df = pd.read_excel(arq, header=1)  # a 1ª linha do arquivo é só um rótulo X1..X23; o cabeçalho real é a 2ª
    df = df.rename(columns={"default payment next month": "inadimplente"})
    # códigos -> categorias legíveis, para que o One-Hot Encoding seja aplicado (conforme documentação da base)
    df["SEX"] = df["SEX"].map({1: "masculino", 2: "feminino"})
    df["EDUCATION"] = df["EDUCATION"].map({1: "pos_graduacao", 2: "universidade", 3: "ensino_medio"}).fillna("outros")
    df["MARRIAGE"] = df["MARRIAGE"].map({1: "casado", 2: "solteiro"}).fillna("outros")
    df.to_csv(os.path.join(SAIDA, "credito_cartao.csv"), index=False)
    print("credito_cartao.csv:", df.shape, df["inadimplente"].value_counts().to_dict())


# ---- 3) Breast Cancer Wisconsin (Diagnostic) (tabular). Arquivo: wdbc.data (sem cabeçalho)
def cancer_mama():
    arq = os.path.join(BRUTOS, "wdbc.data")
    base = ["raio", "textura", "perimetro", "area", "suavidade", "compacidade",
            "concavidade", "pontos_concavos", "simetria", "dimensao_fractal"]
    nomes = ["id", "diagnostico"] + [f"{b}_{s}" for s in ("media", "erro_padrao", "pior") for b in base]
    df = pd.read_csv(arq, header=None, names=nomes)
    df.to_csv(os.path.join(SAIDA, "cancer_mama.csv"), index=False)
    print("cancer_mama.csv:", df.shape, df["diagnostico"].value_counts().to_dict())


if __name__ == "__main__":
    for f in (sms_spam, credito_cartao, cancer_mama):
        try:
            f()
        except FileNotFoundError as e:
            print(f"[PULADO] {f.__name__}: arquivo não encontrado ({e.filename})")
