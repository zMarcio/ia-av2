"""
classificacao_1504.py
=====================
CLASSIFICAÇÃO - OpenML 1504 "steel-plates-fault" (defeitos em chapas de aço)

O que este script faz:
  1. Lê o dataset e prepara X (atributos) e y (classe).
  2. Mostra a análise do dataset (amostras, atributos, classes, correlações).
  3. Avalia 4 classificadores com validação cruzada estratificada de 5 folds:
       - kNN (Distância Euclidiana), k = 5
       - kNN (Distância Manhattan),  k = 5
       - Bayesiano Univariado
       - Bayesiano Multivariado
  4. Mostra a tabela com média ± desvio de Acurácia, Precisão, Recall,
     F1-Score, Tempo de treino e Tempo de teste.
  5. Salva tudo em resultados/classificacao_1504.json e .md

    python src/classificacao_1504.py
"""

import numpy as np

from scripts.comum import (KNNClassificador, gerar_folds, ler_arff, resumir, salvar,
                   tabela_markdown, top_correlacoes, validacao_cruzada)

ARQUIVO = "steel-plates-fault.arff"
K_FOLDS = 5
K_VIZINHOS = 5
SEMENTE = 42

# Nomes reais das colunas V1..V33 (descrição do dataset no OpenML)
NOMES = [
    "X_Minimum", "X_Maximum", "Y_Minimum", "Y_Maximum", "Pixels_Areas",
    "X_Perimeter", "Y_Perimeter", "Sum_of_Luminosity", "Minimum_of_Luminosity",
    "Maximum_of_Luminosity", "Length_of_Conveyer", "TypeOfSteel_A300",
    "TypeOfSteel_A400", "Steel_Plate_Thickness", "Edges_Index", "Empty_Index",
    "Square_Index", "Outside_X_Index", "Edges_X_Index", "Edges_Y_Index",
    "Outside_Global_Index", "LogOfAreas", "Log_X_Index", "Log_Y_Index",
    "Orientation_Index", "Luminosity_Index", "SigmoidOfAreas",
    "Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps",
]


# ===========================================================================
# 1. CARREGAR E PREPARAR OS DADOS
#
# Decisões de pré-processamento:
#   * V28..V33 (Pastry, Z_Scratch, ...) são as OUTRAS 6 classes de defeito
#     em formato one-hot. A classe "Other_Faults" é exatamente "todas elas = 0".
#     Mantê-las seria VAZAMENTO DE DADOS (100% de acerto sem aprender nada),
#     por isso são removidas.
#   * V13 (TypeOfSteel_A400) = 1 - V12 (TypeOfSteel_A300) sempre. É redundante
#     e deixaria a matriz de covariância singular no Bayes multivariado.
#   * Alvo: Class 1 = defeito comum (um dos 6 tipos) -> 0
#           Class 2 = Other_Faults                   -> 1
# ===========================================================================
def carregar():
    _, linhas = ler_arff(ARQUIVO)
    M = np.array(linhas, dtype=float)
    X_completo, y = M[:, :33], (M[:, 33] == 2).astype(int)

    removidas = list(range(27, 33)) + [12]  # índices de V28..V33 e V13
    manter = [j for j in range(33) if j not in removidas]
    return X_completo, X_completo[:, manter], y, [NOMES[j] for j in manter], [NOMES[j] for j in removidas]


# ===========================================================================
# 2. CLASSIFICADOR BAYESIANO - REGRA DE BAYES
#
#                    p(x | c) · P(c)
#      P(c | x) = ---------------------
#                        p(x)
#
#   P(c | x) : probabilidade "a posteriori" da classe c dado o exemplo x
#   p(x | c) : verossimilhança - densidade de x se ele fosse da classe c
#   P(c)     : probabilidade "a priori" da classe c (frequência no treino)
#   p(x)     : evidência - igual para todas as classes, então pode ser ignorada
#
# Decisão: escolher a classe com maior P(c | x). Para evitar números muito
# pequenos (underflow), trabalhamos com LOGARITMO:
#
#      ŷ = argmax_c [ log p(x | c) + log P(c) ]
#
# A diferença entre os dois classificadores abaixo é COMO p(x | c) é calculada.
# ===========================================================================
class BayesUnivariado:
    """
    CASO UNIVARIADO (Naive Bayes Gaussiano)

    Cada atributo é modelado por uma Normal de UMA variável:

                     1              ┌     1  ( x - μ )² ┐
        p(x) =  ───────────  · exp  │ -  ─── (───────)  │
                 √(2π) · σ          └     2  (   σ   )  ┘

        x  : valor de um atributo
        μ  : média desse atributo dentro da classe c
        σ  : desvio padrão desse atributo dentro da classe c
        π  : constante pi (≈ 3,1416)
        exp: função exponencial (e elevado a ...)

    Hipótese "ingênua": os d atributos são independentes dada a classe, então

        p(x | c) = p(x_1 | c) · p(x_2 | c) · ... · p(x_d | c)

    e, em log, o produto vira soma:

        log p(x | c) = Σ_j [ -log(√(2π)·σ_jc) - ½ ((x_j - μ_jc)/σ_jc)² ]
    """

    def treinar(self, X, y):
        self.classes = np.unique(y)
        self.media = np.array([X[y == c].mean(axis=0) for c in self.classes])       # μ_jc
        self.desvio = np.array([X[y == c].std(axis=0) for c in self.classes])       # σ_jc
        self.desvio = np.maximum(self.desvio, 1e-9)                                 # evita σ = 0
        self.log_priori = np.log([np.mean(y == c) for c in self.classes])           # log P(c)
        return self

    def prever(self, X):
        scores = np.zeros((X.shape[0], len(self.classes)))
        for i in range(len(self.classes)):
            mu, sigma = self.media[i], self.desvio[i]
            # log da fórmula p(x) acima, para cada atributo, somado entre atributos
            log_vero = (-np.log(np.sqrt(2 * np.pi) * sigma) - 0.5 * ((X - mu) / sigma) ** 2).sum(axis=1)
            scores[:, i] = log_vero + self.log_priori[i]
        return self.classes[scores.argmax(axis=1)]


class BayesMultivariado:
    """
    CASO MULTIVARIADO (Gaussiano com covariância completa)

    Os d atributos são modelados JUNTOS por uma Normal de d dimensões:

                         1                 ┌    1                      ┐
        p(x) = ──────────────────── · exp  │ - ─── (x-μ)ᵗ Σ⁻¹ (x-μ)    │
                (2π)^(d/2) |Σ|^(1/2)       └    2                      ┘

        x        : vetor coluna com os d atributos do exemplo
        μ        : vetor de médias (d componentes) da classe c
        Σ        : matriz de covariâncias d × d da classe c   (np.cov)
                     diagonal  -> variância de cada atributo (σ_j²)
                     fora dela -> covariância entre pares de atributos (σ_jk)
        |Σ|      : determinante de Σ
        Σ⁻¹      : inversa de Σ
        (x-μ)ᵗ   : transposta de (x - μ)
        d        : número de atributos

    O termo (x-μ)ᵗ Σ⁻¹ (x-μ) é a distância de MAHALANOBIS: considera a
    correlação entre atributos, ao contrário do caso univariado.

    Em log (para estabilidade numérica):

        log p(x|c) = -(d/2)·log(2π) - ½·log|Σ| - ½·(x-μ)ᵗ Σ⁻¹ (x-μ)

    Regularização: somamos λ·I à diagonal de Σ (Σ + λI). Isso garante que Σ
    seja inversível mesmo com atributos quase iguais (Y_Minimum e Y_Maximum).
    """

    def __init__(self, regularizacao=1e-3):
        self.lam = regularizacao

    def treinar(self, X, y):
        self.classes = np.unique(y)
        self.d = X.shape[1]
        self.media, self.inv_cov, self.log_det = [], [], []
        for c in self.classes:
            Xc = X[y == c]
            sigma = np.cov(Xc, rowvar=False) + self.lam * np.eye(self.d)   # Σ + λI
            self.media.append(Xc.mean(axis=0))                             # μ
            self.inv_cov.append(np.linalg.inv(sigma))                      # Σ⁻¹
            self.log_det.append(np.linalg.slogdet(sigma)[1])               # log|Σ|
        self.log_priori = np.log([np.mean(y == c) for c in self.classes])
        return self

    def prever(self, X):
        scores = np.zeros((X.shape[0], len(self.classes)))
        for i in range(len(self.classes)):
            dif = X - self.media[i]                                        # (x - μ)
            mahalanobis = np.einsum("ij,jk,ik->i", dif, self.inv_cov[i], dif)  # (x-μ)ᵗ Σ⁻¹ (x-μ)
            log_vero = -0.5 * self.d * np.log(2 * np.pi) - 0.5 * self.log_det[i] - 0.5 * mahalanobis
            scores[:, i] = log_vero + self.log_priori[i]
        return self.classes[scores.argmax(axis=1)]


# ===========================================================================
# 3. MÉTRICAS DE CLASSIFICAÇÃO
#
# MATRIZ DE CONFUSÃO (classe positiva = 1)
#
#                       Previsto 0   Previsto 1
#        Real 0  ->        VN           FP
#        Real 1  ->        FN           VP
#
#   VP (Verdadeiro Positivo): era 1 e o modelo disse 1
#   VN (Verdadeiro Negativo): era 0 e o modelo disse 0
#   FP (Falso Positivo)     : era 0 e o modelo disse 1  ("alarme falso")
#   FN (Falso Negativo)     : era 1 e o modelo disse 0  ("deixou passar")
#
# ACURÁCIA
#                     VP + VN
#     Acurácia = -----------------        fração de acertos sobre o total
#                VP + VN + FP + FN
#
# PRECISÃO (por classe c)
#                  VP_c
#     Precisão = ---------                dos que o modelo disse "c", quantos são c
#                VP_c + FP_c
#
# RECALL (por classe c)
#                 VP_c
#     Recall = ---------                  dos que são c, quantos o modelo achou
#              VP_c + FN_c
#
# F1-SCORE (por classe c)
#               2 · Precisão · Recall
#     F1 = ---------------------------    média harmônica: só é alta se as
#               Precisão + Recall         duas forem altas
#
# MÉDIA MACRO
#                1   C
#     Macro  = ---   Σ  métrica_c         C = número de classes (2)
#                C  c=1
#
#   Dá o mesmo peso às duas classes. Como o dataset é desbalanceado (65% x 35%),
#   evita que a classe maior esconda os erros na menor.
# ===========================================================================
def matriz_confusao(y_real, y_prev, n_classes=2):
    M = np.zeros((n_classes, n_classes), dtype=int)
    for r, p in zip(y_real, y_prev):
        M[r, p] += 1
    return M


def metricas(y_real, y_prev):
    M = matriz_confusao(y_real, y_prev)
    precisoes, recalls, f1s = [], [], []
    for c in range(2):
        vp = M[c, c]
        fp = M[:, c].sum() - vp
        fn = M[c, :].sum() - vp
        p = vp / (vp + fp) if vp + fp > 0 else 0.0
        r = vp / (vp + fn) if vp + fn > 0 else 0.0
        precisoes.append(p)
        recalls.append(r)
        f1s.append(2 * p * r / (p + r) if p + r > 0 else 0.0)
    return {
        "acuracia": float(np.mean(y_real == y_prev)),
        "precisao": float(np.mean(precisoes)),
        "recall": float(np.mean(recalls)),
        "f1": float(np.mean(f1s)),
    }


# ===========================================================================
# 4. EXECUÇÃO
# ===========================================================================
MODELOS = {
    "kNN (Dist. Euclidiana)": lambda: KNNClassificador(k=K_VIZINHOS, distancia="euclidiana"),
    "kNN (Dist. Manhattan)": lambda: KNNClassificador(k=K_VIZINHOS, distancia="manhattan"),
    "Bayesiano (Univariado)": lambda: BayesUnivariado(),
    "Bayesiano (Multivariado)": lambda: BayesMultivariado(),
}

COLUNAS = [
    ("acuracia", "Acurácia"), ("precisao", "Precisão"), ("recall", "Recall"),
    ("f1", "F1-Score"), ("tempo_treino", "Tempo Treino (s)"), ("tempo_teste", "Tempo Teste (s)"),
]

if __name__ == "__main__":
    X_completo, X, y, nomes, removidas = carregar()
    n0, n1 = int((y == 0).sum()), int((y == 1).sum())
    vazamento = float(np.mean((X_completo[:, 27:33].sum(axis=1) == 0) == (y == 1)))
    correlacoes = top_correlacoes(X, nomes, y)

    print("=" * 70)
    print("CLASSIFICAÇÃO - OpenML 1504 steel-plates-fault")
    print("=" * 70)
    print(f"Amostras: {X.shape[0]}")
    print(f"Atributos: {X_completo.shape[1]} originais -> {X.shape[1]} usados")
    print(f"Removidos: {', '.join(removidas)}")
    print(f"Classes: 0 = defeito comum ({n0}, {n0/len(y):.0%}) | 1 = Other_Faults ({n1}, {n1/len(y):.0%})")
    print(f"Vazamento: V28..V33 definem a classe em {vazamento:.0%} das amostras")
    print("Atributos mais correlacionados com a classe:")
    for c in correlacoes:
        print(f"   {c['atributo']:<25} r = {c['r']:+.3f}")

    # --- validação cruzada ---
    folds = gerar_folds(len(y), k=K_FOLDS, semente=SEMENTE, y=y, estratificado=True)
    resumos, confusoes = {}, {}
    for nome, criar in MODELOS.items():
        resultados, previsoes = validacao_cruzada(criar, X, y, folds, metricas)
        resumos[nome] = resumir(resultados)
        # matriz de confusão somando os 5 folds (cada amostra aparece uma vez)
        confusoes[nome] = sum(matriz_confusao(y[te], p) for (_, te), p in zip(folds, previsoes)).tolist()

    # --- curva de k (análise complementar: k de 1 a 21) ---
    curva_k = {}
    for dist in ("euclidiana", "manhattan"):
        curva_k[dist] = []
        for k in (1, 3, 5, 7, 9, 11, 15, 21):
            res, _ = validacao_cruzada(lambda: KNNClassificador(k=k, distancia=dist), X, y, folds, metricas)
            curva_k[dist].append({"k": k, "acuracia": resumir(res)["acuracia"]["media"]})

    tabela = tabela_markdown(COLUNAS, resumos)
    print(f"\nResultados ({K_FOLDS}-fold estratificado, k = {K_VIZINHOS}):\n")
    print(tabela)
    print("\nMatriz de confusão (soma dos folds) [[VN, FP], [FN, VP]]:")
    for nome, m in confusoes.items():
        print(f"   {nome:<26} {m}")

    md = (f"# Classificação - OpenML 1504 steel-plates-fault\n\n"
          f"- Amostras: {X.shape[0]} | Atributos usados: {X.shape[1]} de {X_completo.shape[1]}\n"
          f"- Removidos: {', '.join(removidas)}\n"
          f"- Classes: defeito comum {n0} | Other_Faults {n1}\n\n{tabela}\n")
    salvar("classificacao_1504", {
        "amostras": int(X.shape[0]), "atributos_originais": int(X_completo.shape[1]),
        "atributos_usados": int(X.shape[1]), "removidos": removidas,
        "classes": {"defeito_comum": n0, "other_faults": n1}, "vazamento": vazamento,
        "correlacoes": correlacoes, "resumo": resumos, "matriz_confusao": confusoes, "curva_k": curva_k,
    }, md)
    print("\nSalvo em resultados/classificacao_1504.json e .md")
