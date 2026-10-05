# Trabalho A2 – Classificação binária e faixas de decisão

Dois programas em Python:

| Arquivo | Função |
|---|---|
| `cortes.py` | **Parte 1.** Histograma das probabilidades previstas, escolha dos cortes `t1 < t2` (na validação) e avaliação das 3 faixas. Funciona com qualquer modelo/base que forneça a probabilidade da classe positiva. |
| `pipeline.py` | **Parte 2.** Pipeline completo: separação treino/validação/teste, 3 modelos com busca de hiperparâmetros, métricas, curvas, SHAP e aplicação do `cortes.py` ao melhor modelo. |

## 1. Instalação

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Preparar as bases

Coloque um CSV por domínio (ex.: na pasta `dados/`). Cada CSV precisa de:
- uma coluna com a **variável-alvo**;
- as demais colunas como atributos (numéricos, categóricos ou uma coluna de texto);
- remova ou ignore (`--ignorar`) colunas de identificação (ex.: `customerID`).

Registre fonte e licença de cada base em `FONTES_E_LICENCAS.md`.

## 3. Executar o pipeline (Parte 2)

```bash
python pipeline.py --csv dados/churn.csv --alvo Churn --positivo Yes --nome churn --ignorar customerID
python pipeline.py --csv dados/spam.csv  --alvo label --positivo spam --nome spam --texto mensagem
python pipeline.py --csv dados/outra.csv --alvo alvo  --positivo 1    --nome outra
```

| Argumento | Obrigatório | Descrição |
|---|---|---|
| `--csv` | sim | caminho do CSV |
| `--alvo` | sim | nome da coluna-alvo |
| `--positivo` | sim | valor do alvo que define a **classe positiva** (ex.: `Yes`, `spam`, `1`) |
| `--nome` | sim | nome do domínio; os resultados vão para `resultados/<nome>/` |
| `--texto` | não | coluna de texto (usa TF-IDF; as demais colunas são ignoradas) |
| `--ignorar` | não | colunas a remover (ex.: identificadores) |
| `--largura` | não | largura dos intervalos do histograma em pontos percentuais (padrão 10; deve dividir 100) |
| `--max-fn-auto` | não | tolerância de positivos entre as negativas automáticas (padrão 0,02) |
| `--max-fp-auto` | não | tolerância de negativos entre as positivas automáticas (padrão 0,02) |

O programa avisa se alguma coluna parece um identificador e converte para número colunas
numéricas lidas como texto (ex.: valores em branco).

### Como os dados são tratados
- Separação estratificada: **60% treino / 20% validação / 20% teste** (semente fixa 42).
- Pré-processamento (imputação, padronização, One-Hot, TF-IDF) dentro do `Pipeline`:
  é ajustado só com o treino, inclusive dentro de cada divisão da validação cruzada (5 folds).
- Modelos: Regressão Logística, Random Forest e Gradient Boosting (`HistGradientBoostingClassifier`),
  com `GridSearchCV` usando **Average Precision** como métrica-alvo (somente no treino).
- O melhor modelo é escolhido pela Average Precision **na validação**.
- Métricas que dependem de decisão (acurácia, precisão, recall, F1, matriz de confusão) usam o
  limiar definido pela constante `LIMIAR` no início do `pipeline.py`.
- `t1` e `t2` são escolhidos na validação e aplicados ao teste sem novos ajustes.

### Arquivos gerados em `resultados/<nome>/`

| Arquivo | Conteúdo |
|---|---|
| `distribuicao_classes.csv` | quantidade e % de cada classe |
| `hiperparametros.csv` | melhores hiperparâmetros, AP no CV e AP na validação |
| `metricas_validacao.csv`, `metricas_teste.csv` | tabelas comparativas dos 3 modelos (AUC-PR trapezoidal e Average Precision em colunas separadas) |
| `matrizes_confusao.png` | matriz de confusão dos 3 modelos (teste) |
| `curvas_roc_pr.png` | curvas ROC e Precision-Recall (teste) |
| `shap_beeswarm.png`, `shap_importancia.csv` | SHAP do melhor modelo (dados de teste) |
| `faixas_validacao.csv`, `faixas_teste.csv` | tabela das 3 faixas |
| `erros_faixas_validacao.csv`, `erros_faixas_teste.csv` | erros automáticos (FN e FP) com os denominadores |
| `histograma_cortes_teste.png` | gráfico da Parte 1 com os cortes (teste) |
| `teste_probabilidades.csv`, `validacao_probabilidades.csv` | `y_true` e `proba` de cada conjunto |
| `resumo.json` | melhor modelo, hiperparâmetros, t1, t2 e tamanhos dos conjuntos |

## 4. Usar o programa da Parte 1 isoladamente (`cortes.py`)

Entrada: CSV com as colunas `y_true` (0/1; 1 = classe positiva) e `proba` (probabilidade da classe positiva, entre 0 e 1).

```bash
# Cortes informados manualmente
python cortes.py --csv resultados/churn/teste_probabilidades.csv --t1 0.1 --t2 0.7 \
       --rotulo-positivo Yes --largura 10 --saida grafico.png

# Cortes escolhidos automaticamente na validação e aplicados ao teste
python cortes.py --csv resultados/churn/teste_probabilidades.csv \
       --csv-val resultados/churn/validacao_probabilidades.csv \
       --max-fn-auto 0.02 --max-fp-auto 0.02 --rotulo-positivo Yes
```

Também pode ser importado: `from cortes import escolher_cortes, avaliar_faixas, plotar_histogramas`.

No gráfico, o eixo X é a probabilidade estimada da classe positiva (0–100%) e o eixo Y é o % de
instâncias de cada intervalo sobre o total do conjunto. **Azul** = todas as instâncias;
**vermelho** = instâncias de rótulo real positivo (mesmo denominador). As linhas tracejadas
marcam `t1` e `t2`, e cada faixa mostra o % da população e o % de positivos.

Faixas: `p < t1` negativa automática · `t1 <= p < t2` análise manual · `p >= t2` positiva automática.

## 5. Bases sugeridas (3 domínios, todas CC BY 4.0)

Detalhes, links e referências em `FONTES_E_LICENCAS.md`.

1. Baixe os arquivos originais do UCI e coloque em `dados_brutos/`.
2. Gere os CSVs: `python preparar_bases.py`
3. Rode o pipeline em cada domínio:

```bash
python pipeline.py --csv dados/spam_sms.csv      --alvo label        --positivo spam --nome spam_sms --texto mensagem
python pipeline.py --csv dados/credito_cartao.csv --alvo inadimplente --positivo 1    --nome credito  --ignorar ID
python pipeline.py --csv dados/cancer_mama.csv   --alvo diagnostico  --positivo M    --nome cancer_mama --ignorar id
```

Observação: a base de câncer de mama é pequena (569 exames), então o conjunto de teste terá ~114 casos
e os cortes `t1`/`t2` ficam mais sensíveis a pequenas variações; comente isso no relatório.

## 6. Estrutura sugerida da entrega

```
trabalhodepython/
├── cortes.py
├── pipeline.py
├── requirements.txt
├── README.md
├── FONTES_E_LICENCAS.md
├── preparar_bases.py
├── dados_brutos/     (arquivos originais baixados)
├── dados/            (um CSV por domínio)
└── resultados/       (saídas de cada domínio)
```
