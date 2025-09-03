# IMDB Rating Prediction
## Descrição do Desafio

O desafio consiste em analisar um conjunto de dados de filmes e, a partir dele:
1. Explorar estatisticamente as variáveis e levantar hipóteses (EDA).
2. Responder perguntas norteadoras, como fatores que influenciam bilheteria e notas do IMDb.
3. Construir um modelo de Machine Learning capaz de prever a nota do IMDb de um filme.
4. Salvar o modelo final em `.pkl` para ser reutilizado em novas previsões.

### Definições importantes:
- Blockbusters: filmes com faturamento muito acima da média, geralmente ultrapassando centenas de milhões de dólares, que acabam distorcendo estatísticas como média de bilheteria.
- Meta_score: nota crítica especializada.
- Certificate: classificação indicativa.
- Gross: faturamento da bilheteria, em dólares.
- Series_Title: título do filme.
- Released_Year: ano de lançamento.
- Runtime: duração do filme (em minutos).
- Genre: gênero ou combinação de gêneros.
- IMDB_Rating: nota média do IMDb, fornecida pelos usuários.
- Overview: breve descrição/sinopse do filme.
- Director: diretor do filme.
- Star1, Star2, Star3, Star4: Principais atores/atrizes do elenco.
- No_of_Votes: quantidade de votos recebidos no IMDb.

---

## Instalação

O repositório precisa ser clonado:

```bash
git clone https://github.com/barbaraalvx/imdb-rating-prediction.git
```

As dependências também precisam ser instaladas:

```bash
bash
cd imdb-rating-prediction
pip install -r requirements.txt
```

---

## Execução

### Executar o notebook:

Precisamos abrir o Jupyter e executar o notebook principal:
```bash
jupyter notebook notebooks/LH_CD_BARBARA.ipynb
```
Ele contém toda a análise exploratória, hipóteses levantadas e os experimentos com modelos.

### Fazer uma nova previsão com novo filme a partir do modelo treinado:
Use a função `predict_one` disponível em `src/predict.py`.

---

## Estrutura do Repositório:
- `data/`: dados brutos
- `models/`: modelos salvos (.pkl)
- `notebooks/`: notebook principal (EDA + Modelagem)
- `reports/`: figuras
- `src/`: scripts Python
- `src/preprocess.py`: limpeza e transformação de dados
- `src/train.py`: função de treino e avaliação dos modelos
- `src/predict.py`: função para prever nota de novos filmes
- `requirements.txt`: pacotes e versões utilizadas

---

Bárbara G. A. Cavalcante