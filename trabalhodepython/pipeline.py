import argparse, glob, json, os, warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, auc, confusion_matrix,
                             roc_curve, precision_recall_curve, ConfusionMatrixDisplay)
import shap
from cortes import escolher_cortes, avaliar_faixas, plotar_histogramas

warnings.filterwarnings("ignore")
SEED = 42
LIMIAR = 0.5

PASTA = os.path.dirname(os.path.abspath(__file__))


def achar_csv(csv):
    if csv:
        for c in (csv, os.path.join(PASTA, csv)):
            if os.path.isfile(c):
                return c
        raise SystemExit(f"\nArquivo nao encontrado: {csv}\n"
                         f"Copie o CSV para a pasta:\n  {PASTA}\n"
                         f"e rode de novo (ou passe o caminho completo em --csv).")
    achados = sorted(glob.glob(os.path.join(PASTA, "*.csv")) + glob.glob(os.path.join(PASTA, "dados", "*.csv")))
    if len(achados) == 1:
        print(f"CSV encontrado automaticamente: {achados[0]}")
        return achados[0]
    if not achados:
        raise SystemExit(f"\nNenhum arquivo .csv encontrado.\nCopie o CSV do trabalho para a pasta:\n  {PASTA}\n"
                         f"e rode de novo.")
    lista = "\n  ".join(os.path.basename(a) for a in achados)
    raise SystemExit(f"\nHa varios CSVs na pasta:\n  {lista}\nEscolha um com --csv NOME.csv")


def achar_alvo(df, alvo):
    if alvo:
        if alvo not in df.columns:
            raise SystemExit(f"\nColuna alvo '{alvo}' nao existe. Colunas do CSV:\n  {list(df.columns)}")
        return alvo
    for c in df.columns:
        if str(c).strip().lower() in ("churn", "target", "alvo", "label", "classe", "class", "y"):
            print(f"Coluna alvo detectada: {c}")
            return c
    print(f"Coluna alvo nao informada: usando a ultima coluna ({df.columns[-1]})")
    return df.columns[-1]


def achar_positivo(serie, positivo):
    vals = serie.dropna().astype(str)
    if positivo is not None:
        if str(positivo) not in set(vals):
            raise SystemExit(f"\nValor positivo '{positivo}' nao existe no alvo. Valores: {sorted(set(vals))}")
        return str(positivo)
    for v in ("Yes", "yes", "YES", "1", "True", "true", "Churn", "Sim", "sim", "S"):
        if v in set(vals):
            print(f"Classe positiva detectada: {v}")
            return v
    v = vals.value_counts().idxmin()
    print(f"Classe positiva nao informada: usando a classe minoritaria ({v})")
    return str(v)


def limpar_features(X, texto):
    """Remove colunas identificadoras (valor unico por linha) e converte numeros gravados como texto."""
    X = X.copy()
    for c in list(X.columns):
        if c == texto or pd.api.types.is_numeric_dtype(X[c]):
            continue
        conv = pd.to_numeric(X[c], errors="coerce")
        if conv.notna().mean() > 0.95:          # ex.: TotalCharges vem como texto
            X[c] = conv
        elif X[c].nunique() == len(X):          # ex.: customerID
            print(f"Coluna identificadora removida: {c}")
            X = X.drop(columns=c)
    return X


def para_denso(M):
    return M.toarray() if hasattr(M, "toarray") else np.asarray(M)


def montar_prep(X, texto):
    if texto:
        return ColumnTransformer([("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1, 2),
                                                            sublinear_tf=True), texto)])
    num = X.select_dtypes(include="number").columns.tolist()
    cat = [c for c in X.columns if c not in num]
    partes = []
    if num:
        partes.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                                        ("sc", StandardScaler())]), num))
    if cat:
        partes.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                        ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=5))]), cat))
    return ColumnTransformer(partes)


def candidatos(X, texto):
    def pipe(clf, denso=False):
        passos = [("prep", montar_prep(X, texto))]

        if denso:
            passos.append(("dens", FunctionTransformer(para_denso, accept_sparse=True)))
        return Pipeline(passos + [("clf", clf)])
    return {
        "Regressão Logística": (pipe(LogisticRegression(max_iter=2000, random_state=SEED)),
                                {"clf__C": [0.01, 0.1, 1, 10], "clf__class_weight": [None, "balanced"]}),
        "Random Forest": (pipe(RandomForestClassifier(random_state=SEED, n_jobs=-1)),
                          {"clf__n_estimators": [100, 300], "clf__max_depth": [None, 10],
                           "clf__min_samples_leaf": [1, 3]}),
        "Gradient Boosting": (pipe(HistGradientBoostingClassifier(random_state=SEED), denso=True),
                              {"clf__learning_rate": [0.05, 0.1], "clf__max_depth": [3, None],
                               "clf__l2_regularization": [0.0, 1.0]}),
    }


def metricas(y, p, limiar=LIMIAR):
    pred = (p >= limiar).astype(int)
    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    return {
        "Acurácia": accuracy_score(y, pred),
        "Precisão": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred),
        "F1": f1_score(y, pred),
        "AUC-ROC": roc_auc_score(y, p),
        "AUC-PR (trapezoidal)": auc(rec, prec),
        "Average Precision (AP)": average_precision_score(y, p),
    }


def shap_beeswarm(modelo, Xtr, Xte, texto, arq, nome_modelo):

    prep, clf = modelo.named_steps["prep"], modelo.named_steps["clf"]
    Ttr, Tte = para_denso(prep.transform(Xtr)), para_denso(prep.transform(Xte))
    nomes = np.array([n.split("__", 1)[-1] for n in prep.get_feature_names_out()])
    rng = np.random.RandomState(SEED)
    Tte_s = Tte[rng.choice(len(Tte), min(len(Tte), 500), replace=False)]
    bg = Ttr[rng.choice(len(Ttr), min(len(Ttr), 200), replace=False)]
    if isinstance(clf, LogisticRegression):
        expl = shap.LinearExplainer(clf, shap.maskers.Independent(bg, max_samples=len(bg)))
        sv, saida = expl.shap_values(Tte_s), "log-odds da classe positiva"
    elif isinstance(clf, RandomForestClassifier):
        expl = shap.TreeExplainer(clf)
        sv = expl.shap_values(Tte_s)
        sv = sv[..., 1] if getattr(sv, "ndim", 2) == 3 else sv[1]
        saida = "probabilidade da classe positiva"
    else:
        f = lambda Z: clf.decision_function(Z)

        resumo = shap.kmeans(bg, 20)
        masker = shap.maskers.Independent(resumo.data)
        expl = shap.Explainer(f, masker, algorithm="permutation", max_evals=2 * Tte.shape[1] + 1)
        sv, saida = expl(Tte_s[:150]).values, "log-odds da classe positiva"
        Tte_s = Tte_s[:150]
    plt.figure()
    shap.summary_plot(sv, Tte_s, feature_names=nomes, max_display=15, show=False)
    plt.title(f"SHAP beeswarm - {nome_modelo}\nSaída explicada: {saida}")
    plt.tight_layout(); plt.savefig(arq, dpi=150, bbox_inches="tight"); plt.close()
    imp = pd.Series(np.abs(sv).mean(0), index=nomes).sort_values(ascending=False).head(15)
    return saida, imp, len(Tte_s)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--csv", default=None, help="CSV (padrao: o unico .csv da pasta do projeto)")
    a.add_argument("--alvo", default=None, help="coluna alvo (padrao: detecta 'Churn' ou usa a ultima)")
    a.add_argument("--positivo", default=None, help="valor do alvo que define a classe positiva")
    a.add_argument("--nome", default=None, help="nome da pasta de saida em resultados/")
    a.add_argument("--texto", default=None, help="coluna de texto (usa TF-IDF)")
    a.add_argument("--largura", type=float, default=10)
    a.add_argument("--max-fn-auto", type=float, default=0.02,
                   help="máx. fração de positivos entre as negativas automáticas (validação)")
    a.add_argument("--max-fp-auto", type=float, default=0.02,
                   help="máx. fração de negativos entre as positivas automáticas (validação)")
    g = a.parse_args()

    caminho = achar_csv(g.csv)
    df = pd.read_csv(caminho)
    g.alvo = achar_alvo(df, g.alvo)
    df = df.dropna(subset=[g.alvo])
    g.positivo = achar_positivo(df[g.alvo], g.positivo)
    g.nome = g.nome or os.path.splitext(os.path.basename(caminho))[0]
    out = os.path.join(PASTA, "resultados", g.nome); os.makedirs(out, exist_ok=True)
    y = (df[g.alvo].astype(str) == str(g.positivo)).astype(int).values
    X = limpar_features(df.drop(columns=[g.alvo]), g.texto)

    dist = pd.DataFrame({"Classe": ["Negativa (0)", f"Positiva (1) = {g.positivo}"],
                         "Quantidade": [(y == 0).sum(), (y == 1).sum()],
                         "Percentual": [100 * (y == 0).mean(), 100 * (y == 1).mean()]})
    dist.to_csv(f"{out}/distribuicao_classes.csv", index=False)
    print(dist.round(2).to_string(index=False))

    Xdev, Xte, ydev, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    Xtr, Xva, ytr, yva = train_test_split(Xdev, ydev, test_size=0.25, stratify=ydev, random_state=SEED)
    print(f"treino={len(ytr)} validação={len(yva)} teste={len(yte)}")

    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    achados, val_ap = {}, {}
    for nome, (pipe, grade) in candidatos(X, g.texto).items():
        gs = GridSearchCV(pipe, grade, scoring="average_precision", cv=cv, n_jobs=-1, refit=True)
        gs.fit(Xtr, ytr)
        achados[nome] = gs
        val_ap[nome] = average_precision_score(yva, gs.predict_proba(Xva)[:, 1])
        print(f"{nome}: AP(CV)={gs.best_score_:.4f} AP(val)={val_ap[nome]:.4f} {gs.best_params_}")
    melhor = max(val_ap, key=val_ap.get)

    lin_val, lin_te = [], []
    for nome, gs in achados.items():
        lin_val.append({"Modelo": nome, **metricas(yva, gs.predict_proba(Xva)[:, 1])})
        lin_te.append({"Modelo": nome, **metricas(yte, gs.predict_proba(Xte)[:, 1])})
    tv, tt = pd.DataFrame(lin_val), pd.DataFrame(lin_te)
    tv.to_csv(f"{out}/metricas_validacao.csv", index=False)
    tt.to_csv(f"{out}/metricas_teste.csv", index=False)
    pd.DataFrame([{"Modelo": n, "Hiperparâmetros": str(gs.best_params_), "AP médio no CV (treino)": gs.best_score_,
                   "AP na validação": val_ap[n]} for n, gs in achados.items()]).to_csv(
        f"{out}/hiperparametros.csv", index=False)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (nome, gs) in zip(axs, achados.items()):
        pr = (gs.predict_proba(Xte)[:, 1] >= LIMIAR).astype(int)
        ConfusionMatrixDisplay(confusion_matrix(yte, pr), display_labels=["Neg", "Pos"]).plot(ax=ax, colorbar=False)
        ax.set_title(nome)
    plt.tight_layout(); plt.savefig(f"{out}/matrizes_confusao.png", dpi=150); plt.close()

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    for nome, gs in achados.items():
        p = gs.predict_proba(Xte)[:, 1]
        fpr, tpr, _ = roc_curve(yte, p); pr_, rc_, _ = precision_recall_curve(yte, p)
        a1.plot(fpr, tpr, label=f"{nome} (AUC={roc_auc_score(yte, p):.3f})")
        a2.plot(rc_, pr_, label=f"{nome} (AP={average_precision_score(yte, p):.3f})")
    a1.plot([0, 1], [0, 1], "k--"); a1.set(xlabel="Taxa de falsos positivos", ylabel="Taxa de verdadeiros positivos", title="Curva ROC (teste)")
    a2.axhline(yte.mean(), color="k", ls="--", label="Baseline (prevalência)")
    a2.set(xlabel="Recall", ylabel="Precisão", title="Curva Precision-Recall (teste)")
    a1.legend(); a2.legend(); plt.tight_layout(); plt.savefig(f"{out}/curvas_roc_pr.png", dpi=150); plt.close()

    best = achados[melhor].best_estimator_
    saida, imp, n_shap = shap_beeswarm(best, Xtr, Xte, g.texto, f"{out}/shap_beeswarm.png", melhor)
    imp.to_csv(f"{out}/shap_importancia.csv", header=["media_abs_shap"])

    pva, pte = best.predict_proba(Xva)[:, 1], best.predict_proba(Xte)[:, 1]
    t1, t2 = escolher_cortes(yva, pva, g.max_fn_auto, g.max_fp_auto)
    tab_v, err_v = avaliar_faixas(yva, pva, t1, t2)
    tab_t, err_t = avaliar_faixas(yte, pte, t1, t2)
    tab_v.to_csv(f"{out}/faixas_validacao.csv", index=False)
    tab_t.to_csv(f"{out}/faixas_teste.csv", index=False)
    pd.Series(err_v).to_csv(f"{out}/erros_faixas_validacao.csv", header=["valor"])
    pd.Series(err_t).to_csv(f"{out}/erros_faixas_teste.csv", header=["valor"])
    pd.DataFrame({"y_true": yte, "proba": pte}).to_csv(f"{out}/teste_probabilidades.csv", index=False)
    plotar_histogramas(yte, pte, g.largura, t1, t2, f"{out}/histograma_cortes_teste.png",
                       rotulo_positivo=str(g.positivo), titulo=f"{g.nome}: {melhor} - conjunto de teste")

    resumo = {"dominio": g.nome, "classe_positiva": g.positivo, "melhor_modelo": melhor,
              "hiperparametros": {k: str(v) for k, v in achados[melhor].best_params_.items()},
              "limiar_metricas": LIMIAR, "t1": t1, "t2": t2, "shap_saida_explicada": saida,
              "shap_n_instancias_teste": int(n_shap), "n_treino": len(ytr), "n_val": len(yva), "n_teste": len(yte)}
    json.dump(resumo, open(f"{out}/resumo.json", "w"), ensure_ascii=False, indent=2)

    pd.set_option("display.width", 250, "display.max_columns", 20)
    print("\n== TESTE ==\n", tt.round(4).to_string(index=False))
    print(f"\nMelhor: {melhor} | t1={t1:.2f} t2={t2:.2f}\n", tab_t.round(2).to_string(index=False))
    for k, v in err_t.items():
        print(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")


if __name__ == "__main__":
    main()