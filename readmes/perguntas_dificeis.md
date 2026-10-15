# Perguntas difíceis sobre a implementação

A rubrica (item 2.2) prevê **4 perguntas: 2 de teoria e 2 de código**. Por isso as perguntas estão separadas nesses dois grupos.

As mais perigosas são **T7**, **T11**, **T13**, **C9**, porque mexem em limitações reais do trabalho. Vale treinar essas em voz alta.

## Resultados da regressão (referência)

### Regressão - OpenML 41700 CPMP-2015-regression

- Amostras: 2108 | Atributos usados: 25
- Alvo: log(1 + runtime) | mediana do runtime 65.9 s
- runstatus: {'memout': 371, 'ok': 1526, 'timeout': 211}

| Modelo | R² | R² ajustado | MAE (log) | RMSE (log) | MAE (s) | Tempo Treino (s) | Tempo Teste (s) |
|---|---|---|---|---|---|---|---|
| kNN (Dist. Euclidiana) | 0.2560 ± 0.0250 | 0.2090 ± 0.0265 | 2.2063 ± 0.0300 | 2.7868 ± 0.0423 | 959.1251 ± 69.6545 | 0.000004 ± 0.000001 | 0.069285 ± 0.001122 |
| kNN (Dist. Manhattan) | 0.3041 ± 0.0609 | 0.2601 ± 0.0648 | 2.0935 ± 0.0691 | 2.6918 ± 0.0735 | 952.6392 ± 77.3953 | 0.000003 ± 0.000000 | 0.068598 ± 0.002005 |
| Regressão Linear Múltipla | 0.3085 ± 0.0244 | 0.2648 ± 0.0259 | 2.2542 ± 0.0532 | 2.6864 ± 0.0304 | 1052.3287 ± 98.1035 | 0.000360 ± 0.000129 | 0.000018 ± 0.000005 |

---

## 📚 Perguntas de teoria

**T1. Por que o z-score é ajustado só no treino? Que mal faria ajustar em tudo?**
Se a média e o desvio usassem o teste, uma informação do teste entraria no treino. Isso é vazamento, mesmo que pequeno. Na vida real, o modelo não conhece os dados novos de antemão.
`comum.py`, em `validacao_cruzada`: `PadronizadorZScore().ajustar(X[idx_treino])`.

**T2. Por que, na regressão, os folds são agrupados e não aleatórios?**
Cada pátio aparece 4 vezes, uma por algoritmo, e as 4 linhas têm o mesmo X do pátio. Se uma linha fosse para o treino e outra para o teste, o kNN acharia o "gêmeo" e o resultado ficaria otimista demais.
`comum.py`, em `gerar_folds(..., grupos=grupos)`.

**T3. Com k = 5 pode dar empate na votação?**
Entre as duas classes, não: com 5 votos, um lado sempre tem 3 ou mais. Um k ímpar evita empate em problema binário.

**T4. Por que o kNN tem tempo de treino quase zero? Isso não é injusto na comparação?**
Porque o treino dele é só guardar os dados. Não é injusto: o custo existe, mas vai todo para o teste (cerca de 60 ms contra 0,09 ms do Bayes). O enunciado pede os dois tempos justamente para mostrar essa troca.

**T5. Por que a Manhattan foi melhor que a Euclidiana na regressão (0,30 contra 0,26)?**
A Euclidiana eleva as diferenças ao quadrado, então um atributo muito diferente pesa demais. A Manhattan soma as diferenças sem elevar, e por isso sofre menos com atributos tortos e valores extremos. É uma explicação provável, não algo que eu tenha provado.

**T6. Por que trabalhar com logaritmo no Bayes?**
O univariado multiplica 26 densidades, e o resultado fica tão pequeno que o computador arredonda para zero (underflow). Com log, o produto vira soma, e o argmax não muda, porque o log é crescente.

**T7. Se o Bayes usa a priori, que favorece o defeito comum, por que ele chamou 904 comuns de outro defeito?**
Porque a verossimilhança pesou muito mais do que a priori. Como as normais não se ajustam bem a dados tortos e correlacionados, a soma de 26 termos errados empurra para o outro defeito. A priori é só um termo, e o log(0,65/0,35) dá cerca de 0,6.

**T8. Para que serve o λI no multivariado? Por que 1e-3?**
Alguns atributos são quase cópias um do outro (como Y_Minimum e Y_Maximum), e isso deixa Σ quase sem inversa. Somar λ na diagonal garante a inversa. Como os dados estão em z-score (variância 1), 1e-3 é 0,1% da variância, ou seja, mexe muito pouco nos dados. O valor foi fixo, sem busca.

**T9. Por que o multivariado tem desvio maior entre os folds (±0,050 contra ±0,003 do univariado)?**
Ele estima uma matriz 26 × 26 por classe, ou seja, 351 números, e alguns atributos são quase colineares. Pequenas mudanças nos dados do fold mudam bastante a inversa de Σ.

**T10. O R² pode ser negativo? Aconteceu no seu trabalho?**
Pode: R² negativo quer dizer que o modelo é pior do que chutar sempre a média. Aconteceu no kNN com k = 1 na regressão (−0,23 na Euclidiana e −0,12 na Manhattan), na curva de k.

**T11. O seu R² de 0,31 está em log. Ele vale para os segundos?**
Não. Ele mede o acerto na escala log, e em segundos o valor seria outro. Por isso também calculei o MAE em segundos, desfazendo o log com `expm1`.

**T12. O que quer dizer um MAE de 2,1 na escala log?**
Em log, errar 2,1 equivale a errar por um fator de cerca de e^2,1 ≈ 8 vezes. Por exemplo, prever 10 s quando foi 80 s. É um erro relativo, não em segundos.

**T13. Desfazer o log da previsão com `expm1` dá a média do tempo?**
Não exatamente. A média em log, desfeita, fica mais perto da mediana e tende a subestimar a média em segundos. É uma limitação conhecida da transformação.

**T14. Para que serve o R² ajustado, se ele quase não muda aqui?**
Ele penaliza o número de atributos (p = 25). Com cerca de 420 amostras de teste por fold, a penalização é pequena: cai cerca de 0,04 (0,309 → 0,265). Ele seria decisivo se houvesse muitos atributos para poucas amostras.

**T15. Por que média macro, e não micro ou ponderada?**
No problema binário, a micro é igual à acurácia, e a ponderada daria mais peso à classe maior (65%). A macro dá o mesmo peso às duas classes, e assim os erros na classe menor aparecem.

---

## 💻 Perguntas de código

**C1. O que acontece se um atributo tiver desvio zero no treino?**
A divisão daria erro (divisão por zero). O código troca o desvio 0 por 1, então a coluna vira só zeros e não atrapalha.
`comum.py`, em `PadronizadorZScore.ajustar`.

**C2. Você aplicou z-score nas colunas de 0 e 1 do algoritmo. Isso é um problema?**
Não. Elas continuam com dois valores só, apenas em outra escala. E isso deixa o peso delas na distância do kNN comparável ao dos outros atributos.

**C3. Na one-hot do algoritmo, por que só 3 colunas para 4 algoritmos?**
A quarta seria redundante: se as três estão em 0, já se sabe qual é o algoritmo. Com as 4, a soma delas daria sempre 1, igual à coluna de 1s do β₀, e o XᵗX ficaria sem inversa. Isso é a "armadilha da variável dummy".
`regressao_41700.py`, em `carregar`: `algoritmos[1:]`.

**C4. O que faz o `argpartition`, e por que não ordenar tudo?**
Ele só separa os k menores, sem ordenar a linha inteira, e isso é mais rápido. A ordem entre os 5 vizinhos não importa para votação nem para média.
`comum.py`, em `_KNNBase._vizinhos`.

**C5. O seu Bayes usa a probabilidade a priori? Qual é o valor dela?**
Usa. P(c) é a frequência da classe no treino: cerca de 0,65 para defeito comum e 0,35 para outro defeito. Ela entra em log, somada à verossimilhança.
`classificacao_1504.py`: `self.log_priori`.

**C6. O univariado usa `std` com divisão por n, e o `np.cov` divide por n−1. Isso é inconsistente?**
É, mas com cerca de 1000 amostras por classe a diferença é de 0,1%, sem efeito no resultado. O enunciado pede o `np.cov` para o multivariado, e o `std` por n é o estimador de máxima verossimilhança da normal.

**C7. Por que usar `slogdet` e não `det`?**
O determinante de uma matriz 26 × 26 pode ser minúsculo ou enorme e estourar o limite do computador. O `slogdet` devolve direto o log do determinante, que é o que a fórmula em log precisa.

**C8. Por que `pinv` e não `inv` na regressão linear?**
Se XᵗX não tiver inversa, por causa de atributos colineares, o `inv` quebra ou dá números absurdos. O `pinv` dá o mesmo resultado quando a inversa existe e ainda funciona quando ela não existe.

**C9. O R² ajustado não deveria usar o n do treino?**
Ponto delicado. No código, ele usa o n do fold de teste, e a fórmula clássica é definida para o ajuste no treino. Se ela perguntar, admita que é uma escolha: aplicar ao teste dá uma penalização maior e mais conservadora.

**C10. A matriz do slide 8 é de qual fold?**
De nenhum fold isolado: é a soma dos 5. Como cada amostra é testada exatamente uma vez, a soma tem as 1941 chapas.

**C11. A acurácia de 79,0% do slide é a mesma que sai da matriz somada?**
Quase. O slide mostra a média das acurácias dos 5 folds, e a matriz dá (1129 + 405) / 1941 = 79,03%. Como os folds têm tamanhos quase iguais, os dois valores coincidem.
