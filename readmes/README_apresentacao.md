# Roteiro da apresentação

São 14 slides, cerca de 1 minuto cada. A ordem deixa a classificação junta (slides 7 a 10), depois a regressão (11), os tempos (12) e as conclusões (13).

| # | Slide | O que falar | Palavras-chave |
|---|---|---|---|
| 1 | Capa | Nome, disciplina e tema. | dois datasets · implementação própria |
| 2 | Introdução | Comparar algoritmos em acerto e tempo, só com NumPy. Z-score ajusta o X; log ajusta o Y da regressão. | objetivo · acerto x tempo · z-score · log |
| 3 | Dataset 1504 | Defeitos em chapas de aço: defeito comum x outro defeito (65/35). Tirei V28–V33 porque entregam a resposta. | 1941 · 26 atributos · vazamento |
| 4 | Dataset 41700 | Tempo para organizar pátios de contêineres. Tirei o runstatus (vazamento). Muitos batem 3600 s, então usei log. | 2108 · 527 × 4 · limite 3600 s · log |
| 5 | Algoritmos | kNN vota pelos vizinhos; Bayes usa a normal; a linear usa a equação normal. | vizinhos · normal · covariância · β |
| 6 | Protocolo e métricas | 5 folds; estratificado na classificação, agrupado na regressão; z-score só no treino. Para que serve cada métrica. | 5 folds · média ± desvio · métricas |
| 7 | Resultados: classificação | kNN Manhattan foi o melhor (79%). O Bayes univariado ficou abaixo do chute fixo (65%). | kNN 79% · Bayes 51% |
| 8 | Matriz de confusão | O kNN deixa passar 268 outros defeitos; o Bayes chama 904 defeitos comuns de outro defeito. | tipo de erro · rede gigante |
| 9 | Acurácia e F1 | Os dados não seguem uma normal e são correlacionados, o que derruba o Bayes. | dados assimétricos · correlação |
| 10 | Efeito de k (classificação) | Análise extra: acurácia estável entre 77% e 79%; k = 5 fica no topo. | k = 5 adequado |
| 11 | Resultados: regressão | Linear e kNN Manhattan com R² ≈ 0,31; a linear é mais estável. Na curva de k, k = 1 dá R² negativo. | R² 0,31 · estável · k |
| 12 | Tempo | O kNN não treina, mas é lento para prever. Bayes e linear são rápidos. | kNN lento no teste |
| 13 | Conclusões | Classificação: vale o kNN. Regressão: a linear é a melhor e a mais rápida. Depois, as limitações. | melhor · mais rápido · vale a pena · limitações |
| 14 | Referências | Não precisa ler. | |

---

## Falas por slide

### Slide 1: Capa
"Boa tarde. Meu nome é Márcio Lima, e vou apresentar a avaliação de algoritmos de aprendizado supervisionado em dois datasets do OpenML: um de classificação e um de regressão."

### Slide 2: Introdução
"O objetivo deste trabalho é comparar algoritmos de aprendizado supervisionado em duas tarefas: uma de classificação e uma de regressão. Na classificação, comparei o kNN com distância Euclidiana e Manhattan e o Bayes univariado e multivariado. Na regressão, comparei o kNN com as duas distâncias e a regressão linear múltipla. Tudo foi implementado do zero, só com NumPy.

A comparação olha duas coisas: quanto o modelo acerta e quanto tempo ele leva. Na prática, escolher um modelo é equilibrar essas duas.

Antes de treinar, usei duas fórmulas para preparar os dados e facilitar o aprendizado. A primeira é o **z-score**, que ajusta o X, as entradas. Ele subtrai a média e divide pelo desvio padrão, e assim todos os atributos ficam na mesma escala. A segunda é o **log**, que ajusta o Y da regressão, a saída. Ele encolhe os tempos muito grandes para o modelo não ficar dominado por eles."

### Slide 3: Dataset de classificação
"O dataset de classificação é de defeitos em chapas de aço. Cada linha é um defeito encontrado numa chapa. Existem 6 tipos conhecidos de defeito, como riscos, manchas, sujeira e saliências. Tudo o que não se encaixa em nenhum deles fica numa sétima categoria, chamada *Other_Faults*, que quer dizer "outros defeitos". O modelo precisa responder uma pergunta: esse defeito é de um dos 6 tipos conhecidos, que eu chamo de defeito comum, ou é um outro defeito?

O problema é que seis colunas, da V28 à V33, marcam justamente qual dos 6 tipos conhecidos é o defeito. Se todas estão zeradas, é um outro defeito. Ou seja, a resposta já estava na tabela. Isso se chama vazamento, então eu tirei essas colunas. Também tirei o TypeOfSteel_A400, que é sempre o oposto do A300 e não traz informação nova. Saí de 33 para 26 atributos.

No fim, o **X** são as 26 medidas do defeito: posição, área, perímetro, luminosidade, espessura e tipo de aço. O **Y** é a resposta: 0 para defeito comum e 1 para outro defeito.

O gráfico mostra que são 1941 chapas: 1268 com defeito comum, que são 65%, e 673 com outro defeito, que são 35%. Essas classes são desbalanceadas, e isso vai importar na hora de escolher as métricas."

### Slide 4: Dataset de regressão
"O dataset de regressão é sobre organizar pátios de contêineres. Num porto, os contêineres ficam empilhados, e os que vão sair primeiro precisam estar por cima. Reorganizar o pátio antes de o navio chegar se chama pré-empilhamento. Quatro algoritmos de busca, que são variações do A* e do IDA*, tentaram organizar 527 pátios diferentes, com limite de uma hora, ou seja, 3600 segundos. Cada linha é um pátio resolvido por um desses algoritmos: 527 vezes 4, o que dá 2108 linhas.

O **X** são as características do pátio: quantas pilhas ele tem, qual a altura máxima, quão cheio ele está, quantas pilhas têm contêiner fora de ordem, e assim por diante. São 22 atributos, mais 3 colunas de 0 e 1 que dizem qual algoritmo de busca foi usado, porque o modelo só entende número. Total: 25 atributos. O **Y** é o tempo, em segundos, que levou para organizar aquele pátio.

Aqui também havia vazamento. A coluna *runstatus* diz se a execução deu certo ou estourou o tempo ou a memória. Quando ela não diz "deu certo", o tempo é sempre exatamente 3600. Ou seja, ela entrega o Y, então eu tirei. Também tirei o *repetition*, que tem o mesmo valor em todas as linhas, e o *instance_id*, que é só o nome do pátio.

O gráfico mostra o Y. Os tempos vão de menos de 1 segundo até uma hora, e 582 execuções pararam no limite de 3600. Como os valores são muito desiguais, eu treinei com o log do tempo, que deixa essa escala equilibrada."

#### Apoio: one-hot do algoritmo de busca (se perguntarem)
A coluna de texto `algorithm` vira **3 colunas de 0 e 1**. Se as três estão em 0, é o 1º algoritmo. Por isso são 3 colunas, e não 4.

```
algoritmo usado        alg=2   alg=3   alg=4
astar.symmulgt    →      0       0       0     ← tudo zero = o 1º
astar.symmullt    →      1       0       0
idastar.symmulgt  →      0       1       0
idastar.symmullt  →      0       0       1
```

Código: `scripts/regressao_41700.py`, em `carregar`. O `algoritmos[1:]` pula o 1º nome, e o `1.0 if ... else 0.0` marca 1 na coluna do algoritmo daquela linha.

**Depois, essas colunas passam pelo z-score**, junto com as outras 22 (`PadronizadorZScore`, a mesma conta para todas as colunas). Valores reais, iguais nos 5 folds:
```
cada algoritmo aparece em 25% das linhas
μ = 0,25        σ = √(0,25 · 0,75) = 0,433

0  →  (0 − 0,25) / 0,433 = −0,58     ← "não foi esse algoritmo"
1  →  (1 − 0,25) / 0,433 = +1,73     ← "foi esse algoritmo"
```

O que o modelo recebe:
```
           alg=2    alg=3    alg=4
linha A   +1,73    −0,58    −0,58   ← algoritmo 2
linha B   −0,58    −0,58    +1,73   ← algoritmo 4
linha C   −0,58    −0,58    −0,58   ← nenhum valor alto: é o 1º
```

- O valor muda, mas o significado não: a tabela que vai para o modelo tem −0,58 e +1,73, e não volta a ser 0 e 1.
- Quem diz qual algoritmo é a coluna: o valor alto (+1,73) marca a coluna do algoritmo usado, como o 1 fazia antes.
- Não existe valor no meio: a coluna só tinha 0 ou 1, então depois só tem −0,58 ou +1,73.
- O 0 sempre vira negativo e o 1 sempre vira positivo, porque a média de uma coluna 0/1 fica entre 0 e 1.
- Não dá divisão por zero, porque o desvio é 0,433. Ele só seria 0 se a coluna fosse toda igual, e aí o código troca por 1.
- O modelo não sabe o nome do algoritmo: ele aprende que +1,73 naquela coluna está ligado a tempos maiores ou menores.
- Os valores são iguais nos 5 folds porque cada pátio tem os 4 algoritmos e os folds separam pátios inteiros.
- Não se usa 1, 2, 3, 4 porque isso criaria uma ordem falsa entre os algoritmos.

### Slide 5: Algoritmos
"O **kNN** não treina nada, só guarda os dados. Para prever, ele busca os 5 vizinhos mais próximos. Na classificação, faz votação; na regressão, tira a média. A distância pode ser a Euclidiana, que é a linha reta, ou a Manhattan, que soma as diferenças de cada atributo.

O **Bayes univariado** ajusta uma distribuição normal para cada atributo, separadamente, e supõe que eles são independentes. O **Bayes multivariado** ajusta uma normal de várias dimensões, com a matriz de covariância, e assim enxerga a relação entre os atributos.

A **regressão linear múltipla** procura os coeficientes β que minimizam a soma dos erros ao quadrado, e prevê com ŷ = Xβ."

### Slide 6: Protocolo e métricas
"Para comparar os modelos de forma justa, usei **validação cruzada de 5 folds**. Eu divido os dados em 5 partes, treino com 4 e testo com 1, e repito 5 vezes, trocando a parte de teste. Assim, cada amostra é testada uma vez, e o resultado aparece como **média ± desvio padrão**: a média diz o desempenho, e o desvio diz se ele é estável.

Na classificação, os folds são **estratificados**: cada parte mantém os 65% e 35% das classes. Na regressão, eles são **agrupados por pátio**: as 4 linhas do mesmo pátio ficam sempre juntas, senão o modelo veria o pátio no treino e reconheceria no teste.

O **z-score** é calculado só com o treino de cada fold, para o modelo não espiar o teste.

A limpeza dos dados também foi testada. Com as colunas de vazamento, o kNN chegava a 96% e o Bayes multivariado a 100% na classificação, e a regressão linear pulava de R² 0,31 para 0,69. Ou seja, eles estavam lendo a resposta, não aprendendo. Sem o z-score, o kNN caía de 79% para 63%.

Nas métricas da classificação:
- A **acurácia** diz, de todas as chapas, quantas o modelo acertou.
- A **precisão** diz, das que ele chamou de outro defeito, quantas eram mesmo.
- O **recall** diz, de todos os outros defeitos, quantos ele encontrou.
- O **F1** junta precisão e recall e só fica alto se as duas forem altas.

Precisão, recall e F1 são calculados em **média macro**, que dá o mesmo peso às duas classes.

Nas métricas da regressão:
- O **R²** diz quanto da variação do tempo o modelo consegue explicar. O valor 1 é perfeito, e o 0 equivale a chutar sempre a média.
- O **R² ajustado** faz a mesma coisa, mas desconta pela quantidade de atributos.
- O **MAE** é o erro médio, quanto o modelo erra em média."

### Slide 7: Tabela da classificação
"Esta é a tabela pedida no enunciado, com média e desvio dos 5 folds. O melhor foi o **kNN Manhattan**, com 79% de acurácia e F1 de 0,76. A Euclidiana ficou praticamente empatada, com 78,8%.

Para ter uma referência: como 65% das chapas são de defeito comum, um modelo que chutasse sempre "defeito comum" já acertaria 65%. O kNN passa bem disso. O **Bayes univariado** ficou com 51%, abaixo do chute. O **multivariado** chegou a 61%, ainda abaixo.

Repare também nos desvios. Os do kNN ficam perto de ±0,02, então o resultado é estável entre os folds."

### Slide 8: Matriz de confusão
"A matriz mostra o tipo de erro, que a acurácia esconde. As linhas são o que a chapa era de verdade, e as colunas são o que o modelo disse. A diagonal azul são os acertos.

O kNN acerta 1129 defeitos comuns e 405 outros defeitos. O erro principal dele é deixar passar 268 outros defeitos, chamando-os de comuns.

O Bayes univariado erra do jeito oposto: ele chama 904 defeitos comuns de outro defeito, cerca de 71% deles. Ele encontra quase todos os outros defeitos (628 de 673), mas porque sai chamando quase tudo de outro defeito. É como jogar uma rede gigante: pega quase todos os peixes, mas vem muito lixo junto."

### Slide 9: Acurácia e F1
"O gráfico compara a acurácia e o F1 de cada modelo. Os dois kNN ficam na frente, perto de 0,79 de acurácia e 0,76 de F1.

Por que o Bayes foi mal? Ele supõe que cada atributo tem formato de sino, a distribuição normal. O univariado ainda supõe que os atributos são independentes entre si. Os dados quebram as duas regras: muitos atributos são tortos, e vários são quase cópias um do outro, como área e perímetro. O multivariado melhora porque enxerga a correlação, mas continua supondo o formato de sino.

O kNN não supõe nada: ele só procura as 5 chapas mais parecidas e faz votação. Por isso ele se adapta melhor a esses dados."

### Slide 10: Efeito de k na classificação
"Esta é uma análise extra, que não foi pedida. Eu fixei k = 5 no trabalho e quis conferir se essa escolha era boa. Testei k de 1 a 21.

Na classificação, a acurácia fica entre 77% e 79% em toda a faixa. Com k = 1 cai um pouco, porque o modelo segue um vizinho só e copia o ruído. O k = 5 é o melhor da Manhattan e fica perto do melhor da Euclidiana, que chega a 79,1% com k = 7. Então a escolha foi adequada.

As duas distâncias andam juntas, porque, com os dados em z-score, elas acham praticamente os mesmos vizinhos."

### Slide 11: Resultados da regressão
"Na regressão, a **regressão linear** teve o maior R², 0,31, seguida de perto pelo **kNN Manhattan**, com 0,30. A Euclidiana ficou atrás, com 0,26.

Um R² de 0,31 quer dizer que os modelos explicam cerca de 31% da variação do tempo. É um valor modesto, porque prever o tempo de um algoritmo de busca é difícil, e 28% das execuções foram cortadas no limite de uma hora.

A linear e o kNN Manhattan empatam na média, mas a linear é mais estável: o desvio dela é ±0,024, contra ±0,061 do kNN.

Na curva de k, a regressão se comporta diferente da classificação. Com k = 1, o R² fica negativo, pior do que chutar a média, porque um vizinho só pode ser um pátio que demorou muito ou pouco por acaso. O R² vai melhorando até perto de k = 21. Então, na regressão, o k = 5 não é o melhor, e essa é uma das limitações."

### Slide 12: Tempo de treino e teste
"O enunciado pede os tempos de treino e teste. Eles mostram um padrão claro.

O **kNN** quase não treina, porque ele só guarda os dados: são 0,003 milissegundos. Em compensação, é o mais lento no teste, cerca de 60 a 70 milissegundos, porque precisa medir a distância de cada amostra nova até todas as amostras do treino.

O **Bayes** e a **regressão linear** fazem o contrário: gastam um pouco mais no treino, calculando médias, covariâncias ou os coeficientes, e depois preveem quase instantaneamente. O Bayes univariado testa em 0,09 milissegundos, e a linear em 0,02.

Ou seja, o kNN deixa todo o trabalho para a hora de prever, e os outros fazem o trabalho antes."

### Slide 13: Conclusões
"Na classificação, o melhor foi o **kNN Manhattan**, com 79% de acurácia e F1 de 0,76. O mais rápido foi o **Bayes univariado**, mas com 51% de acurácia, abaixo do chute fixo. Vale a pena o kNN: ele demora cerca de 60 milissegundos a mais por fold, e em troca acerta 18 pontos a mais que o melhor Bayes.

Na regressão, a **regressão linear** foi a melhor e a mais rápida ao mesmo tempo: R² de 0,31, prevendo em 0,02 milissegundos, contra 69 do kNN. Aqui não há troca: ela vence nos dois.

As limitações foram três. Os atributos não seguem a distribuição normal, o que prejudica o Bayes. Na regressão, 28% dos tempos foram cortados em 3600 segundos, então o modelo não sabe o tempo real desses casos. E o k e o λ ficaram fixos, sem busca de melhor valor; a curva de k mostrou que, na regressão, um k maior ajudaria."

### Slide 14: Referências
Não precisa ler. Se perguntarem, os datasets são do OpenML (ids 1504 e 41700).

---

**Perguntas prováveis**
- *Por que tirar V28–V33?* Elas marcam os 6 tipos conhecidos de defeito, ou seja, entregam a resposta. Com elas, o kNN chegava a 96% e o Bayes multivariado a 100% sem aprender nada.
- *Por que tirar o runstatus?* Quando ele não é "ok", o tempo é sempre 3600 s. Com ele, o R² da linear pulava de 0,31 para 0,69.
- *Por que log no tempo?* O tempo vai de 0 a 3600 s e é muito desigual. O log reduz o peso dos valores extremos. Sem ele, a linear errava 1221 s em média (contra 1052 s) e previa tempo negativo.
- *Por que z-score?* O kNN mede distância, e um atributo de número grande dominaria. Sem ele, o kNN caía de 79% para 63%.
- *O algoritmo de busca é o Y?* Não. Ele está no X (3 colunas de 0 e 1). O Y é só o tempo.
- *Por que o Bayes foi mal?* Ele supõe uma distribuição normal, e os atributos são assimétricos e correlacionados.
- *Por que o kNN é lento?* Para cada previsão, ele mede a distância até todos os pontos de treino.
- *O tempo foi pedido?* Sim: na Tabela 1 do enunciado, na seção 3.3 e na análise crítica. A curva de k é análise extra.
