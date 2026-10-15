# Explicação dos algoritmos e dos ajustes

Este arquivo tem duas partes:
- **Parte 1, ajustes nos dados:** média, desvio, z-score, log(1 + runtime) e one-hot.
- **Parte 2, algoritmos:** distâncias, kNN, Bayes e regressão linear.

Em cada item: o que faz, a fórmula, cada símbolo, onde está no código, nuances e como falar.

| Item | Classificação (aço) | Regressão (contêineres) |
|---|---|---|
| Média e desvio | sim (z-score e Bayes) | sim (z-score) |
| Z-score | sim, no X | sim, no X |
| log(1 + runtime) | não | sim, no Y |
| One-hot | não | sim, no algoritmo de busca |
| Euclidiana / Manhattan / kNN | sim (votação) | sim (média) |
| Bayes univariado / multivariado | sim | não |
| Regressão linear | não | sim |

---

# PARTE 1: AJUSTES NOS DADOS

## 1. Média e desvio padrão

### O que fazem
A **média** diz o "centro" de um atributo. O **desvio padrão** diz o quanto os valores costumam se afastar desse centro.

### As fórmulas
```
         Σ xᵢ                          ┌──────────────────┐
μ   =  ──────              σ   =    √  │  Σ (xᵢ − μ)²     │
           n                           │  ────────────    │
                                       │       n          │
                                       └──────────────────┘
```

### Cada símbolo
- **xᵢ** = o valor do atributo na amostra i
- **n** = o número de amostras do treino
- **μ** = a média
- **(xᵢ − μ)** = o quanto cada valor se afasta da média
- **²** = eleva ao quadrado, para o sinal sumir
- **√** = tira a raiz, para voltar à unidade original
- **σ** = o desvio padrão

### Onde entram
- No **z-score**, nos dois datasets.
- No **Bayes**, onde a média e o desvio são calculados **por classe**.
- No resultado final, para dizer **média ± desvio dos 5 folds**.

### 💻 No código
- `scripts/comum.py`, em `PadronizadorZScore.ajustar`: `X.mean(axis=0)` e `X.std(axis=0)`.
- `scripts/comum.py`, em `resumir`: `valores.mean()` e `valores.std()`.

### Nuances
O `np.std` divide por n, e o `np.cov` do Bayes multivariado divide por n − 1. Com cerca de 1000 amostras, a diferença é desprezível.
No resultado ± dos folds, o desvio diz se o modelo é estável: um desvio pequeno quer dizer que ele foi parecido nos 5 folds.

---

## 2. Z-score (padronização do X)

### O que faz
Coloca **todos os atributos na mesma régua**: cada um fica com média 0 e desvio 1. Ele não muda a forma dos dados, só a escala.

### A fórmula
```
          x − μ
z   =   ───────
            σ
  ↑       ↑   ↑
valor   quanto  dividido pelo
novo    se afasta  desvio
        da média
```

### Cada símbolo
- **x** = o valor original do atributo (ex.: a área da chapa)
- **μ** = a média desse atributo **no treino**
- **σ** = o desvio desse atributo **no treino**
- **z** = quantos desvios o valor está acima (+) ou abaixo (−) da média

### Exemplo
```
área média no treino = 1000,  desvio = 200
chapa com área 1400  →  z = (1400 − 1000) / 200 = +2   (2 desvios acima)
chapa com área 900   →  z = (900 − 1000) / 200  = −0,5 (meio desvio abaixo)
```

### O que ele devolve e o que guarda
- **Devolve** a tabela inteira com os valores em z.
- **Guarda** o μ e o σ de cada atributo, para aplicar os mesmos números no teste.

### 💻 No código
`scripts/comum.py`:
- `PadronizadorZScore`: `ajustar` guarda a média e o desvio, e `transformar` faz `(X - media) / desvio`;
- em `validacao_cruzada`, ele é ajustado com `X[idx_treino]` e aplicado no treino e no teste.

### Por que escolhemos o z-score
O kNN mede distância. Sem z-score, a área (milhares) mandaria sozinha, e atributos entre 0 e 1 não pesariam nada. **Sem ele, o kNN da classificação caía de 79% para 63%.** Foi escolhido em vez do min-max porque não depende do menor e do maior valor, que podem ser extremos.

### Nuances
Ele é ajustado só no treino de cada fold, para o modelo não espiar o teste.
Se um atributo tem desvio 0 no treino, o código troca por 1 para não dividir por zero.
O Bayes e a regressão linear praticamente não mudam com ele; quem depende é o kNN.

---

## 3. log(1 + runtime) (ajuste do Y da regressão)

### O que faz
**Encolhe os tempos grandes**, para os casos de 3600 s não dominarem o modelo. Ele só é usado no Y da regressão.

### A fórmula
```
y  =  log ( 1  +  runtime )
↑      ↑    ↑        ↑
alvo  logaritmo  evita log(0)  tempo em segundos
usado natural    quando o      que o algoritmo de
no treino        tempo é 0     busca levou
```

### Cada símbolo
- **runtime** = o tempo, em segundos, que o algoritmo de busca levou para organizar o pátio. É um dado da tabela, de 0 a 3600 s
- **1 +** = evita o log de 0, que não existe
- **log** = o logaritmo natural. Ele transforma multiplicações em somas: de 1 para 10 s e de 100 para 1000 s viram o mesmo "passo"
- **y** = o valor que o modelo aprende e prevê

### Exemplo
```
runtime =    0 s  →  y = 0,0
runtime =   10 s  →  y = 2,4
runtime =  100 s  →  y = 4,6
runtime = 3600 s  →  y = 8,2
```
Repare que de 0 a 3600 s, o y vai só de 0 a 8,2.

### Para voltar a segundos
```
runtime = e^y − 1        (np.expm1)
```

### 💻 No código
- `scripts/regressao_41700.py`, em `carregar`: `y = np.log1p(runtime)`.
- Em `metricas`: `mae_segundos` usa `np.expm1` para desfazer o log.

### Por que usamos
Os tempos vão de menos de 1 s até 3600 s, e 582 execuções pararam no limite. **Sem o log, a regressão linear errava 1221 s em média e previa tempo negativo. Com o log, o erro caiu para 1052 s e todas as previsões ficaram positivas.**

### Nuances
O log não acelera nada: ele só muda a escala do Y para o modelo aprender melhor.
O R² e o MAE da tabela estão em log; um MAE de 2,1 em log quer dizer errar por um fator de cerca de 8 vezes.
Desfazer o log da previsão tende a dar um valor mais perto da mediana do que da média em segundos.

---

## 4. One-hot (algoritmo de busca na regressão)

### O que faz
Transforma o **nome do algoritmo de busca** (texto) em **colunas de 0 e 1**, porque o modelo só entende número.

### Como fica
```
algoritmo                      alg=2   alg=3   alg=4
astar.symmulgt  (o 1º)    →      0       0       0
astar.symmullt            →      1       0       0
idastar.symmulgt          →      0       1       0
idastar.symmullt          →      0       0       1
```

### 💻 No código
`scripts/regressao_41700.py`, em `carregar`: `algoritmos[1:]` cria só 3 colunas.

### Nuances
São só 3 colunas para 4 algoritmos porque, se as três estão em 0, já se sabe que é o primeiro.
Com 4 colunas, a soma delas daria sempre 1, igual à coluna do β₀, e o XᵗX da regressão linear ficaria sem inversa.
O algoritmo de busca está no X, não no Y: o Y é só o tempo.

---

# PARTE 2: ALGORITMOS

## 5. Distância Euclidiana

### O que faz
Mede a **distância em linha reta** entre duas chapas, como uma régua esticada de um ponto ao outro.

### A fórmula
```
                 ┌──────────────────────┐
d(x, y)  =    √  │  Σ  (xᵢ − yᵢ)²       │
                 └──────────────────────┘
```

### Cada símbolo
- **x** = a amostra nova, que o modelo quer prever (já em z-score)
- **y** = uma amostra do treino (também em z-score)
- **xᵢ, yᵢ** = o valor do atributo i em cada uma
- **(xᵢ − yᵢ)** = a diferença naquele atributo
- **²** = eleva ao quadrado, para o sinal sumir e as diferenças grandes pesarem mais
- **Σ** = soma as diferenças de todos os atributos (26 na classificação, 25 na regressão)
- **√** = tira a raiz

### Exemplo (só 3 atributos)
```
nova    x = ( 0,5 ;  1,0 ; −0,2)
treino  y = ( 0,1 ; −1,0 ;  0,3)
diferenças = ( 0,4 ;  2,0 ; −0,5)
ao quadrado = (0,16 ; 4,00 ; 0,25)  → soma 4,41 → √ = 2,10
```

### 💻 No código
`scripts/comum.py`, em `distancia_euclidiana`. Ela calcula a distância até todas as amostras do treino, em blocos de 256.

### Nuances
Uma diferença grande num único atributo domina, porque o quadrado aumenta muito o valor (no exemplo, o 2,0 virou 4,00).
Sem z-score, o atributo de número maior mandaria sozinho.

---

## 6. Distância Manhattan

### O que faz
Mede a distância **andando pelos quarteirões**, como um táxi: só na horizontal e na vertical, sem cortar caminho.

### A fórmula
```
d(x, y)  =   Σ   | xᵢ − yᵢ |
```

### Cada símbolo
- **x, y, xᵢ, yᵢ** = os mesmos da Euclidiana
- **| |** = valor absoluto: tira o sinal
- **Σ** = soma as diferenças
- Não tem quadrado nem raiz

### O mesmo exemplo
```
diferenças = ( 0,4 ; 2,0 ; −0,5)
sem sinal  = ( 0,4 ; 2,0 ;  0,5)  → soma = 2,90
```

### 💻 No código
`scripts/comum.py`, em `distancia_manhattan`.

### Nuances
Como não eleva ao quadrado, um atributo muito diferente pesa menos, e a distância sofre menos com valores extremos.
Na classificação, as duas empataram (79,0% contra 78,8%); na regressão, a Manhattan foi melhor (R² 0,30 contra 0,26).

---

## 7. kNN (k vizinhos mais próximos)

### O que faz
"Diga-me com quem andas e te direi quem és." Para prever uma amostra nova, ele procura as **5 mais parecidas** do treino e copia a resposta delas.

### Passo a passo
```
1. Treinar  = só guardar os dados (nenhuma conta)
2. Prever   = medir a distância da amostra nova até TODAS do treino
3.          = pegar as 5 mais próximas (k = 5)
4a. Classificação → votação: a classe que mais aparece nas 5
4b. Regressão     → média:   a média dos tempos (em log) das 5
```

### As fórmulas
```
Classificação:   ŷ = classe mais votada entre os k vizinhos

Regressão:       ŷ =  1/k · Σ yⱼ
```

### Cada símbolo
- **k** = quantos vizinhos olhar. Aqui, **5**
- **ŷ** = a previsão
- **yⱼ** = a resposta de cada vizinho: 0 ou 1 na classificação, log do tempo na regressão
- **1/k · Σ** = a média dos 5

### Exemplo
```
5 vizinhos: comum, outro, comum, comum, outro
votos: comum = 3, outro = 2   →   previsão: comum
```

### 💻 No código
`scripts/comum.py`:
- `_KNNBase.treinar` só guarda os dados;
- `_vizinhos` usa o `argpartition` para achar os 5 menores sem ordenar tudo;
- `KNNClassificador.prever` conta os votos;
- `KNNRegressor.prever` tira a média com `.mean`.

### Nuances
Ele não aprende uma fórmula: guarda tudo e decide na hora, por isso treina em 0,003 ms e testa em cerca de 60 ms.
Ele não supõe nada sobre o formato dos dados, e por isso foi o melhor na classificação (79%).
Com k = 5 não há empate entre duas classes, porque um lado sempre tem 3 ou mais votos.
Com k = 1 ele copia o ruído de um único vizinho, e na regressão o R² chegou a ficar negativo.

---

## 8. Regra de Bayes (base dos dois Bayes)

### O que faz
Calcula **qual classe é mais provável** para aquela chapa e escolhe a maior.

### A fórmula
```
              p(x | c) · P(c)
P(c | x) =  ─────────────────
                  p(x)
```

### Cada símbolo
- **c** = a classe: 0 (defeito comum) ou 1 (outro defeito)
- **x** = a chapa nova (26 atributos)
- **P(c | x)** = a **posteriori**: a chance de ser da classe c, depois de ver a chapa
- **p(x | c)** = a **verossimilhança**: o quanto essa chapa se parece com as chapas da classe c (vem da normal)
- **P(c)** = a **priori**: a frequência da classe no treino. **0,65** para comum e **0,35** para outro defeito
- **p(x)** = igual para as duas classes, então não muda quem ganha

### Na prática (em log)
```
ŷ = a classe com maior [ log p(x | c) + log P(c) ]
```
O log é usado porque multiplicar 26 números pequenos daria praticamente zero. Com log, o produto vira soma.

### 💻 No código
`scripts/classificacao_1504.py`: `self.log_priori` e `scores.argmax(axis=1)`, nas duas classes Bayes.

---

## 9. Bayes univariado (Naive Bayes)

### O que faz
Faz um "sino" (distribuição normal) **para cada atributo, separado**, e multiplica os 26.

### A fórmula (para um atributo)
```
                1                ┌      1   ( x − μ )² ┐
p(x) =  ─────────────── · exp    │  −  ─── (───────)   │
          √(2π) · σ              └      2   (   σ   )   ┘
```

### Cada símbolo
- **x** = o valor de um atributo da chapa nova
- **μ** = a média desse atributo **dentro da classe**
- **σ** = o desvio desse atributo dentro da classe
- **π** ≈ 3,1416
- **exp** = "e elevado a". Quanto mais longe da média, menor o valor
- Depois, **p(x | c) = p(x₁|c) · p(x₂|c) · … · p(x₂₆|c)**. Essa multiplicação é a suposição "ingênua": os atributos seriam independentes

### 💻 No código
`scripts/classificacao_1504.py`, em `BayesUnivariado`: `treinar` guarda a média, o desvio e o `log_priori`; `prever` soma o log da fórmula dos 26 atributos.

### Nuances
Ele supõe que cada atributo tem formato de sino, mas 10 dos 26 são bem tortos.
Ele supõe que os atributos são independentes, mas 15 pares andam muito juntos (como área e perímetro), e ele conta a mesma informação duas vezes.
Por isso ele chamou 904 defeitos comuns de outro defeito e ficou com 51%, abaixo do chute fixo de 65%.
É o mais rápido no teste: 0,09 ms.

---

## 10. Bayes multivariado (com λI)

### O que faz
Faz **um único sino de 26 dimensões**, que enxerga como os atributos andam juntos.

### A fórmula
```
                    1                    ┌    1                     ┐
p(x) = ───────────────────────── · exp   │ − ─── (x−μ)ᵗ Σ⁻¹ (x−μ)   │
        (2π)^(d/2) · |Σ|^(1/2)           └    2                     ┘
```

### Cada símbolo
- **x** = a chapa nova inteira (vetor com 26 valores)
- **μ** = o vetor das 26 médias da classe
- **Σ** = a matriz de covariância 26 × 26 da classe (`np.cov`). A diagonal tem a variância de cada atributo, e fora dela fica o quanto cada par anda junto
- **Σ⁻¹** = a inversa de Σ
- **|Σ|** = o determinante de Σ
- **d** = o número de atributos: **26**
- **(x−μ)ᵗ Σ⁻¹ (x−μ)** = a distância de Mahalanobis: uma distância que leva em conta a correlação
- **λI** = soma **0,001** na diagonal de Σ, para garantir a inversa (alguns atributos são quase cópias, como Y_Minimum e Y_Maximum)

### 💻 No código
`scripts/classificacao_1504.py`, em `BayesMultivariado`:
- `np.cov(...) + self.lam * np.eye(d)` é o Σ + λI;
- `np.linalg.inv` calcula a inversa;
- `slogdet` calcula o log do determinante;
- `einsum` calcula a distância de Mahalanobis.

### Nuances
Ele resolve o problema da independência, porque enxerga a correlação, e por isso sobe de 51% para 61%.
Ele continua supondo o formato de sino, por isso ainda fica atrás do kNN.
Ele estima 351 números por classe, e isso deixa o resultado menos estável entre os folds (±0,050).
Como os dados estão em z-score, o λ = 0,001 é só 0,1% da variância e quase não mexe nos dados.

---

## 11. Regressão linear múltipla

### O que faz
Encontra **o peso de cada atributo** que, somados, chegam o mais perto possível do tempo real (em log).

### As fórmulas
```
ŷ  =  β₀ + β₁·x₁ + β₂·x₂ + … + β₂₅·x₂₅          (forma matricial: ŷ = Xβ)

β  =  (XᵗX)⁻¹ Xᵗ y                              (equação normal)
```

### Cada símbolo
- **ŷ** = o tempo previsto (em log)
- **β₀** = o intercepto: o valor de ŷ quando todos os x são 0, ou seja, um pátio "médio", já que os x estão em z-score
- **βⱼ** = o peso do atributo j: quanto ŷ muda se x_j subir 1 desvio
- **xⱼ** = o valor do atributo j (25 atributos)
- **X** = a tabela do treino com uma coluna de 1s na frente (para o β₀)
- **Xᵗ** = a transposta de X
- **(XᵗX)⁻¹** = a inversa de XᵗX
- **y** = os tempos reais (em log) do treino

### 💻 No código
`scripts/regressao_41700.py`, em `RegressaoLinearMultipla`:
- `treinar`: `np.linalg.pinv(X1.T @ X1) @ X1.T @ y`;
- `prever`: `X1 @ self.beta`.

### Nuances
Os β são os que deixam a soma dos erros ao quadrado a menor possível.
Ela usa o `pinv` em vez do `inv`, porque ele dá o mesmo resultado e não quebra se houver atributos colineares.
Ela foi a melhor e a mais rápida na regressão: R² 0,31, prevendo em 0,02 ms.
Ela é a mais estável (±0,024 no R², contra ±0,061 do kNN Manhattan).

---

# 🆚 Resumo para falar

> "Antes de treinar, usei o **z-score** para colocar todos os atributos na mesma escala, com média 0 e desvio 1, e o **log** para encolher os tempos grandes do Y da regressão. O **kNN** guarda os dados e, para prever, procura as 5 amostras mais parecidas. A distância pode ser a **Euclidiana**, em linha reta, ou a **Manhattan**, somando as diferenças de cada atributo. O **Bayes** calcula a probabilidade de cada classe usando a distribuição normal: o **univariado** faz um sino por atributo e supõe que eles são independentes, e o **multivariado** faz um sino só, com a matriz de covariância. A **regressão linear** encontra o peso de cada atributo que minimiza o erro ao quadrado. Na classificação, o kNN foi o melhor porque não supõe nada sobre os dados. Na regressão, a linear foi a melhor e a mais rápida."
