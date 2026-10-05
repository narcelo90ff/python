
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _validar(y, p):
    try:
        y = np.asarray(y).astype(int)
    except (ValueError, TypeError):
        raise ValueError("y_true deve ser numérico com apenas 0 e 1 (1 = classe positiva). "
                         "Converta os rótulos de texto (ex.: 'yes'/'no') para 1/0 antes.")
    try:
        p = np.asarray(p).astype(float)
    except (ValueError, TypeError):
        raise ValueError("proba deve ser numérica (probabilidades entre 0 e 1).")
    if y.shape != p.shape or y.ndim != 1:
        raise ValueError("y_true e proba devem ser vetores de mesmo tamanho.")
    if len(y) == 0:
        raise ValueError("y_true e proba estão vazios.")
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("y_true deve conter apenas 0 e 1 (1 = classe positiva).")
    if np.isnan(p).any() or p.min() < 0 or p.max() > 1:
        raise ValueError("proba deve estar em [0, 1] e sem NaN.")
    return y, p


def _validar_cortes(t1, t2):
    if not (0 <= t1 < t2 <= 1):
        raise ValueError("É necessário 0 <= t1 < t2 <= 1.")


def _bordas(largura_pp):
    if largura_pp <= 0 or largura_pp > 100:
        raise ValueError("A largura (em pontos percentuais) deve estar em (0, 100].")
    n = 100 / largura_pp
    if abs(n - round(n)) > 1e-9:
        raise ValueError("A largura (em pontos percentuais) deve dividir 100.")
    return np.linspace(0, 1, int(round(n)) + 1)


def histograma_percentual(y, p, largura_pp=10):

    y, p = _validar(y, p)
    bordas = _bordas(largura_pp)
    idx = np.clip(np.digitize(p, bordas[1:-1], right=False), 0, len(bordas) - 2)
    n = len(p)
    todos = np.bincount(idx, minlength=len(bordas) - 1) / n * 100
    pos = np.bincount(idx[y == 1], minlength=len(bordas) - 1) / n * 100
    assert abs(todos.sum() - 100) < 1e-9
    return bordas, todos, pos


def avaliar_faixas(y, p, t1, t2):
    y, p = _validar(y, p)
    _validar_cortes(t1, t2)
    n, n_pos, n_neg = len(y), int(y.sum()), int((1 - y).sum())
    faixas = {
        "Negativa automática (p < t1)": p < t1,
        "Análise manual (t1 <= p < t2)": (p >= t1) & (p < t2),
        "Positiva automática (p >= t2)": p >= t2,
    }
    linhas = []
    for nome, m in faixas.items():
        q = int(m.sum())
        qp, qn = int(y[m].sum()), int((1 - y[m]).sum())
        linhas.append({
            "Faixa": nome, "Instâncias": q, "% da população (de N)": 100 * q / n,
            "Positivos": qp, "Negativos": qn,
            "% positivos na faixa (de instâncias da faixa)": 100 * qp / q if q else np.nan,
            "% negativos na faixa (de instâncias da faixa)": 100 * qn / q if q else np.nan,
        })
    tab = pd.DataFrame(linhas)
    fn = int(y[p < t1].sum())
    fp = int((1 - y[p >= t2]).sum())
    erros = {
        "N total": n, "Positivos reais": n_pos, "Negativos reais": n_neg,
        "Positivos classificados como negativos (FN automáticos)": fn,
        "  % sobre todos os positivos reais": 100 * fn / n_pos if n_pos else np.nan,
        "  % sobre os classificados automaticamente como negativos":
            100 * fn / int((p < t1).sum()) if (p < t1).any() else np.nan,
        "Negativos classificados como positivos (FP automáticos)": fp,
        "  % sobre todos os negativos reais": 100 * fp / n_neg if n_neg else np.nan,
        "  % sobre os classificados automaticamente como positivos":
            100 * fp / int((p >= t2).sum()) if (p >= t2).any() else np.nan,
        "Cobertura automática (% de N)": 100 * (1 - faixas["Análise manual (t1 <= p < t2)"].mean()),
    }
    return tab, erros


def escolher_cortes(y_val, p_val, max_fn_auto=0.02, max_fp_auto=0.02, passo=0.01):

    y, p = _validar(y_val, p_val)
    if not (0 < passo < 1):
        raise ValueError("passo deve estar em (0, 1).")
    if not (0 <= max_fn_auto <= 1 and 0 <= max_fp_auto <= 1):
        raise ValueError("As tolerâncias devem estar em [0, 1].")
    n = len(p)
    ordem = np.argsort(p, kind="mergesort")
    ps, ys = p[ordem], y[ordem]
    cum_pos = np.concatenate([[0], np.cumsum(ys)])
    cum_neg = np.concatenate([[0], np.cumsum(1 - ys)])
    grade = np.round(np.arange(passo, 1.0, passo), 10)
    k = np.searchsorted(ps, grade, side="left")  # nº de instâncias com p < t
    # t1: positivos entre as p < t / nº de instâncias com p < t
    A = [(t, kk) for t, kk in zip(grade, k) if kk == 0 or cum_pos[kk] / kk <= max_fn_auto]
    # t2: negativos entre as p >= t / nº de instâncias com p >= t
    B = [(t, kk) for t, kk in zip(grade, k)
         if n - kk == 0 or (cum_neg[n] - cum_neg[kk]) / (n - kk) <= max_fp_auto]
    melhor = None  # minimiza o volume da análise manual; desempate: faixa manual mais estreita
    for t1, k1 in A:
        for t2, k2 in B:
            if t1 < t2:
                chave = (int(k2 - k1), t2 - t1)
                if melhor is None or chave < melhor[0]:
                    melhor = (chave, t1, t2)
    if melhor is None:
        raise ValueError("Tolerâncias inviáveis: não há t1 < t2 que as respeitem. Relaxe-as.")
    return float(melhor[1]), float(melhor[2])


def plotar_histogramas(y, p, largura_pp=10, t1=None, t2=None, arquivo=None,
                       rotulo_positivo="classe positiva", titulo=None):
    bordas, todos, pos = histograma_percentual(y, p, largura_pp)
    x = bordas[:-1] * 100
    w = largura_pp
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x, todos, width=w, align="edge", color="blue", alpha=0.6,
           edgecolor="navy", label="Todas as instâncias")
    ax.bar(x, pos, width=w, align="edge", color="red", alpha=0.9,
           edgecolor="darkred", label=f"Rótulo real positivo ({rotulo_positivo})")
    for i in range(len(x)):
        if todos[i] > 0:
            ax.text(x[i] + w / 2, todos[i] + 0.3, f"{todos[i]:.1f}", ha="center", fontsize=8, color="navy")
    ax.set_ylim(0, max(todos.max() * 1.35, 1))
    if t1 is not None and t2 is not None:
        for t in (t1, t2):
            ax.axvline(t * 100, color="k", ls="--", lw=1.2)
        tab, _ = avaliar_faixas(y, p, t1, t2)
        topo = ax.get_ylim()[1]
        centros = [(0 + t1 * 100) / 2, (t1 + t2) * 50, (t2 * 100 + 100) / 2]
        nomes = ["Negativa\nautomática", "Análise\nmanual", "Positiva\nautomática"]
        for c, nm, (_, r) in zip(centros, nomes, tab.iterrows()):
            ax.text(c, topo * 0.97, f"{nm}\n{r['% da população (de N)']:.1f}% da pop.",
                    ha="center", va="top", fontsize=9, bbox=dict(fc="white", ec="gray", alpha=0.8))
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, max(largura_pp, 10)))
    ax.set_xlabel(f"Probabilidade estimada da classe positiva ({rotulo_positivo}) (%)")
    ax.set_ylabel("% de instâncias (sobre o total do conjunto)")
    ax.set_title(titulo or "Distribuição das probabilidades previstas (conjunto de teste)")
    ax.legend(loc="center right")
    fig.tight_layout()
    if arquivo:
        fig.savefig(arquivo, dpi=150)
        plt.close(fig)  # evita acumular figuras abertas na memória
    return fig


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--csv", required=True)
    a.add_argument("--col-y", default="y_true")
    a.add_argument("--col-proba", default="proba")
    a.add_argument("--largura", type=float, default=10, help="largura em pontos percentuais")
    a.add_argument("--t1", type=float, required=True)
    a.add_argument("--t2", type=float, required=True)
    a.add_argument("--rotulo-positivo", default="classe positiva")
    a.add_argument("--saida", default="grafico_cortes.png")
    g = a.parse_args()
    df = pd.read_csv(g.csv)
    for col in (g.col_y, g.col_proba):
        if col not in df.columns:
            raise SystemExit(f"Coluna '{col}' não encontrada. Colunas disponíveis: {list(df.columns)}")
    plotar_histogramas(df[g.col_y], df[g.col_proba], g.largura, g.t1, g.t2, g.saida, g.rotulo_positivo)
    tab, err = avaliar_faixas(df[g.col_y], df[g.col_proba], g.t1, g.t2)
    pd.set_option("display.width", 250, "display.max_columns", 20)
    print(tab.round(2).to_string(index=False))
    for k, v in err.items():
        print(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")


if __name__ == "__main__":
    main()