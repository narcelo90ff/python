# Fontes e licenças das bases de dados

Licenças conferidas nas páginas oficiais do UCI Machine Learning Repository em 04/10/2026.
As três bases estão sob **Creative Commons Attribution 4.0 International (CC BY 4.0)**, que permite
compartilhar e adaptar os dados para qualquer finalidade (inclusive acadêmica), **desde que se dê o crédito**.
Licença: https://creativecommons.org/licenses/by/4.0/

> Atenção: cópias dessas bases em outros sites (Kaggle, Hugging Face etc.) às vezes trazem licença
> diferente ou "desconhecida". Baixe e cite sempre a página do UCI indicada abaixo.

| # | Domínio | Base | Arquivo CSV | Variável-alvo | Classe positiva | Instâncias | Fonte | Licença |
|---|---------|------|-------------|---------------|-----------------|-----------|-------|---------|
| 1 | Mensagens (texto) | SMS Spam Collection | `dados/spam_sms.csv` | `label` | `spam` | 5.574 mensagens (antes de remover duplicadas) | https://archive.ics.uci.edu/dataset/228/sms+spam+collection | CC BY 4.0 |
| 2 | Finanças / crédito | Default of Credit Card Clients | `dados/credito_cartao.csv` | `inadimplente` | `1` (inadimplente no mês seguinte) | 30.000 clientes | https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients | CC BY 4.0 |
| 3 | Saúde | Breast Cancer Wisconsin (Diagnostic) | `dados/cancer_mama.csv` | `diagnostico` | `M` (maligno) | 569 exames | https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic | CC BY 4.0 |

## Arquivos originais a baixar (colocar em `dados_brutos/`)
1. SMS Spam Collection: arquivo `SMSSpamCollection` (dentro do .zip da página).
2. Default of Credit Card Clients: arquivo `default of credit card clients.xls`.
3. Breast Cancer Wisconsin (Diagnostic): arquivo `wdbc.data`.

Depois rode `python preparar_bases.py` para gerar os CSVs em `dados/`.

## Referências (citar no relatório)
- ALMEIDA, T. A.; GÓMEZ HIDALGO, J. M.; SILVA, T. P. Towards SMS Spam Filtering: Results under a New Dataset.
  International Journal of Information Security Science, v. 2, n. 1, p. 1-18, 2013.
  SMS Spam Collection [Dataset]. UCI Machine Learning Repository.
- YEH, I-C.; LIEN, C-H. The comparisons of data mining techniques for the predictive accuracy of probability of
  default of credit card clients. Expert Systems with Applications, v. 36, n. 2, p. 2473-2480, 2009.
  Default of Credit Card Clients [Dataset]. UCI Machine Learning Repository.
- WOLBERG, W. H.; STREET, W. N.; MANGASARIAN, O. L. Breast Cancer Wisconsin (Diagnostic) [Dataset].
  UCI Machine Learning Repository, 1995.

## Tratamentos aplicados pelo `preparar_bases.py`
- **SMS:** leitura do arquivo separado por tabulação; remoção de mensagens duplicadas (evita o mesmo texto em treino e teste).
- **Crédito:** cabeçalho lido da 2ª linha; coluna-alvo renomeada para `inadimplente`; `SEX`, `EDUCATION` e `MARRIAGE`
  convertidas de códigos para categorias (conforme a documentação da base), para aplicar One-Hot Encoding. Coluna `ID` ignorada na execução.
- **Câncer de mama:** inclusão dos nomes das colunas (10 medidas x média, erro padrão e pior valor). Coluna `id` ignorada na execução.
