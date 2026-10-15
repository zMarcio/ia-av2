"""
comum.py
========
Funções usadas pelos DOIS datasets:

  1. Leitura do arquivo ARFF (formato do OpenML), feita manualmente.
  2. Padronização z-score.
  3. Divisão em folds para a validação cruzada.
  4. Distâncias Euclidiana e Manhattan.
  5. kNN (classificação e regressão).
  6. Correlação de Pearson (análise dos atributos).
  7. Validação cruzada com tempos, média ± desvio e gravação dos resultados.

Os arquivos classificacao_1504.py e regressao_41700.py importam daqui.
"""

import json
import os
import time

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # pasta principal do projeto
PASTA_DADOS = os.path.join(RAIZ, "dados")              # onde o código procura os arquivos .arff
PASTA_RESULTADOS = os.path.join(RAIZ, "resultados")    # onde os .json e .md de resultado são gravados


# ===========================================================================
# 1. LEITURA DO ARFF
#
# Formato ARFF (resumo):
#     @relation nome
#     @attribute nome_coluna tipo      <- uma linha por coluna
#     @data
#     valor1,valor2,...                <- uma linha por amostra
# ===========================================================================
def ler_arff(nome_arquivo):
    """Devolve (nomes_das_colunas, linhas). Cada linha é uma lista de strings."""
    nomes, linhas = [], []
    lendo_dados = False
    with open(os.path.join(PASTA_DADOS, nome_arquivo), encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if not linha or linha.startswith("%"):  # linha vazia ou comentário
                continue
            if not lendo_dados:
                minuscula = linha.lower()
                if minuscula.startswith("@attribute"):           # cabeçalho: guarda o nome da coluna
                    nomes.append(linha.split()[1].strip("'\""))
                elif minuscula.startswith("@data"):              # daqui para baixo vêm as amostras
                    lendo_dados = True
                continue
            linhas.append([v.strip().strip("'\"") for v in linha.split(",")])   # uma amostra = valores separados por vírgula
    return nomes, linhas


# ===========================================================================
# 2. PADRONIZAÇÃO Z-SCORE
#
#              x - μ
#        z = -------
#               σ
#
#   x : valor original de um atributo
#   μ : média desse atributo, calculada SOMENTE no conjunto de treino
#   σ : desvio padrão desse atributo, também só no treino
#   z : valor padronizado (média 0 e desvio 1 no treino)
#
# Por que? O kNN mede distâncias: um atributo em milhões (ex.: Sum_of_Luminosity)
# dominaria outro entre 0 e 1 (ex.: Edges_Index). Com z-score todos pesam igual.
# μ e σ vêm só do treino para não "espiar" o teste (evita vazamento).
# ===========================================================================
class PadronizadorZScore:
    def ajustar(self, X):
        # Chamado SÓ com o treino: calcula e guarda μ e σ de cada coluna (axis=0 = por coluna)
        self.media = X.mean(axis=0)     # μ de cada atributo
        self.desvio = X.std(axis=0)     # σ de cada atributo (divide por n)
        self.desvio[self.desvio == 0] = 1.0  # coluna constante: evita divisão por zero
        return self

    def transformar(self, X):
        # Chamado no treino E no teste, sempre com o μ e o σ do treino.
        # Devolve uma tabela NOVA, com os valores em z. Ex.: coluna 0/1 do one-hot
        # com μ = 0,25 e σ = 0,433 vira -0,58 / +1,73.
        return (X - self.media) / self.desvio


# ===========================================================================
# 3. K-FOLD
#
# Divide as n amostras em k partes (folds). Em cada rodada i = 1..k:
#   - treino = todos os folds menos o i
#   - teste  = fold i
# Assim toda amostra é testada exatamente uma vez.
#
#  * estratificado=True (classificação): cada fold mantém a proporção de classes.
#  * grupos (regressão): amostras do mesmo grupo (mesmo instance_id) ficam
#    sempre no mesmo fold, para o modelo não ver a "mesma instância" no treino.
# ===========================================================================
def gerar_folds(n, k=5, semente=42, y=None, estratificado=False, grupos=None):
    """Devolve uma lista de k tuplas (indices_treino, indices_teste)."""
    # Ideia: como dar cartas. Embaralha e entrega uma amostra para cada fold, em
    # rodízio (0, 1, 2, 3, 4, 0, 1, ...). O "% k" é o resto da divisão por k e
    # é ele que faz o rodízio.
    rng = np.random.default_rng(semente)   # embaralhador; semente fixa = mesmo resultado sempre
    fold_de = np.empty(n, dtype=int)  # fold_de[i] = em qual fold a amostra i caiu

    if grupos is not None:
        # REGRESSÃO: distribui os GRUPOS (pátios), não as linhas.
        unicos = rng.permutation(np.unique(grupos))                 # embaralha os 527 pátios
        fold_do_grupo = {g: i % k for i, g in enumerate(unicos)}    # pátio -> fold, em rodízio
        fold_de[:] = [fold_do_grupo[g] for g in grupos]             # cada linha herda o fold do seu pátio
    elif estratificado:
        # CLASSIFICAÇÃO: faz o rodízio separado em cada classe, para todo fold
        # ficar com a mesma proporção (65% comum / 35% outro).
        for c in np.unique(y):
            idx = rng.permutation(np.where(y == c)[0])              # embaralha as amostras da classe c
            fold_de[idx] = np.arange(len(idx)) % k                  # rodízio 0, 1, 2, 3, 4, 0, ...
    else:
        # Divisão simples, sem estratificar (não é usada no trabalho).
        idx = rng.permutation(n)
        fold_de[idx] = np.arange(n) % k

    # Para cada fold i: teste = quem caiu no fold i; treino = todo o resto.
    todos = np.arange(n)
    return [(todos[fold_de != i], todos[fold_de == i]) for i in range(k)]


# ===========================================================================
# 4. DISTÂNCIAS
#
# As duas funções recebem A (m pontos de teste) e B (n pontos de treino) e
# devolvem a matriz D (m, n) com D[i, j] = distância entre A[i] e B[j].
# O cálculo é feito em blocos para não estourar a memória.
# ===========================================================================
TAMANHO_BLOCO = 256


# ---------------------------------------------------------------------------
# DISTÂNCIA EUCLIDIANA
#                          _______________________
#                         /   n
#   d_Euclidiana(X, Y) = /    Σ  (x_i - y_i)²
#                       √    i=1
#
#   X, Y : dois pontos (vetores) com n atributos cada
#   x_i  : valor do i-ésimo atributo do ponto X
#   y_i  : valor do i-ésimo atributo do ponto Y
#   n    : número de atributos (dimensões)
#   Σ    : soma sobre todos os atributos
#   √    : raiz quadrada
#
# É a distância "em linha reta" (régua). Por elevar ao quadrado, diferenças
# grandes em um único atributo pesam muito.
# ---------------------------------------------------------------------------
def distancia_euclidiana(A, B):
    D = np.empty((A.shape[0], B.shape[0]))
    for ini in range(0, A.shape[0], TAMANHO_BLOCO):                       # 256 amostras de teste por vez
        # None cria uma dimensão extra: cada amostra de teste é comparada com TODAS as de treino
        diferenca = A[ini:ini + TAMANHO_BLOCO, None, :] - B[None, :, :]   # (x_i - y_i)
        D[ini:ini + TAMANHO_BLOCO] = np.sqrt((diferenca ** 2).sum(axis=2))  # √Σ(x_i - y_i)², somando os atributos
    return D


# ---------------------------------------------------------------------------
# DISTÂNCIA MANHATTAN (ou "City Block")
#                         n
#   d_Manhattan(X, Y) =   Σ  | x_i - y_i |
#                        i=1
#
#   X, Y : dois pontos com n atributos
#   x_i  : i-ésimo atributo de X
#   y_i  : i-ésimo atributo de Y
#   |.|  : valor absoluto (módulo)
#   n    : número de atributos
#
# Soma das diferenças absolutas, como andar em quarteirões de uma cidade.
# É menos sensível a um único atributo muito diferente (outliers).
# ---------------------------------------------------------------------------
def distancia_manhattan(A, B):
    D = np.empty((A.shape[0], B.shape[0]))
    for ini in range(0, A.shape[0], TAMANHO_BLOCO):
        diferenca = A[ini:ini + TAMANHO_BLOCO, None, :] - B[None, :, :]   # (x_i - y_i), igual à Euclidiana
        D[ini:ini + TAMANHO_BLOCO] = np.abs(diferenca).sum(axis=2)          # Σ|x_i - y_i|: sem quadrado e sem raiz
    return D


# O kNN escolhe a distância pelo nome: KNNClassificador(k=5, distancia="manhattan")
DISTANCIAS = {"euclidiana": distancia_euclidiana, "manhattan": distancia_manhattan}


# ===========================================================================
# 5. K-NEAREST NEIGHBORS (kNN)
#
# Ideia: "diga-me com quem andas e te direi quem és".
#   1. Treinar = apenas guardar os dados de treino (não há cálculo).
#   2. Para prever um ponto novo x:
#        a) calcula a distância d(x, x_j) até TODOS os pontos de treino;
#        b) pega os k pontos mais próximos (k vizinhos);
#        c) classificação -> classe mais votada entre os k vizinhos:
#
#               ŷ = argmax_c  Σ_{j ∈ N_k(x)}  1[ y_j = c ]
#
#           regressão     -> média dos alvos dos k vizinhos:
#
#               ŷ = (1/k) Σ_{j ∈ N_k(x)}  y_j
#
#   x        : ponto a ser previsto
#   N_k(x)   : conjunto dos k vizinhos mais próximos de x
#   y_j      : rótulo (ou valor) do vizinho j
#   1[...]   : vale 1 se a condição for verdadeira, 0 caso contrário
#   argmax_c : a classe c que tem o maior número de votos
#   ŷ        : previsão
# ===========================================================================
class _KNNBase:
    def __init__(self, k=5, distancia="euclidiana"):
        self.k = k
        self.funcao_distancia = DISTANCIAS[distancia]

    def treinar(self, X, y):
        # kNN é "preguiçoso" (lazy): o treino só memoriza os dados
        self.X_treino = X
        self.y_treino = y
        return self

    def _vizinhos(self, X):
        # D[i, j] = distância da amostra de teste i até a amostra de treino j
        D = self.funcao_distancia(X, self.X_treino)
        # argpartition acha os k menores sem ordenar a linha inteira
        return np.argpartition(D, self.k - 1, axis=1)[:, :self.k]


class KNNClassificador(_KNNBase):
    def prever(self, X):
        rotulos_vizinhos = self.y_treino[self._vizinhos(X)]   # (m, k): a classe de cada um dos 5 vizinhos
        n_classes = int(self.y_treino.max()) + 1               # 2 (0 = comum, 1 = outro defeito)
        votos = np.zeros((X.shape[0], n_classes), dtype=int)
        for c in range(n_classes):
            votos[:, c] = (rotulos_vizinhos == c).sum(axis=1)  # quantos vizinhos são da classe c
        return votos.argmax(axis=1)                            # a classe mais votada (com k = 5 não há empate)


class KNNRegressor(_KNNBase):
    def prever(self, X):
        # média dos alvos (log do tempo) dos 5 vizinhos
        return self.y_treino[self._vizinhos(X)].mean(axis=1)


# ===========================================================================
# 6. CORRELAÇÃO DE PEARSON
#
#              Σ (x_i - x̄)(y_i - ȳ)
#     r = -------------------------------
#          √Σ (x_i - x̄)²  · √Σ (y_i - ȳ)²
#
#   x_i, y_i : valores do atributo e do alvo na amostra i
#   x̄, ȳ     : médias do atributo e do alvo
#   r        : vai de -1 a +1
#                +1 -> sobem juntos;  -1 -> um sobe e o outro desce;
#                 0 -> sem relação LINEAR
# ===========================================================================
def correlacao_pearson(x, y):
    dx, dy = x - x.mean(), y - y.mean()
    denominador = np.sqrt((dx ** 2).sum()) * np.sqrt((dy ** 2).sum())
    return float((dx * dy).sum() / denominador) if denominador > 0 else 0.0


def top_correlacoes(X, nomes, y, quantidade=6):
    lista = [{"atributo": n, "r": correlacao_pearson(X[:, j], y)} for j, n in enumerate(nomes)]
    return sorted(lista, key=lambda c: -abs(c["r"]))[:quantidade]


# ===========================================================================
# 7. VALIDAÇÃO CRUZADA
# ===========================================================================
def validacao_cruzada(criar_modelo, X, y, folds, funcao_metricas):
    """
    Para cada fold:
      1. separa treino / teste;
      2. padroniza (z-score) usando só o treino;
      3. mede o tempo de treino   -> modelo.treinar(...)
      4. mede o tempo de teste    -> modelo.prever(...)
      5. calcula as métricas.

    criar_modelo    : função sem argumentos que devolve um modelo NOVO
    funcao_metricas : função (y_real, y_prev) -> dict de métricas
    Retorna (lista de resultados por fold, lista de previsões por fold).
    """
    resultados, previsoes = [], []
    for idx_treino, idx_teste in folds:
        # z-score: μ e σ calculados SÓ no treino e aplicados nos dois (evita vazamento)
        padronizador = PadronizadorZScore().ajustar(X[idx_treino])
        X_tr = padronizador.transformar(X[idx_treino])
        X_te = padronizador.transformar(X[idx_teste])

        modelo = criar_modelo()   # modelo novo a cada fold: nada passa de um fold para o outro

        inicio = time.perf_counter()
        modelo.treinar(X_tr, y[idx_treino])
        tempo_treino = time.perf_counter() - inicio

        inicio = time.perf_counter()
        y_prev = modelo.prever(X_te)
        tempo_teste = time.perf_counter() - inicio

        # compara a previsão com a resposta real do teste (o modelo não aprende com isso)
        r = funcao_metricas(y[idx_teste], y_prev)
        r["tempo_treino"] = tempo_treino
        r["tempo_teste"] = tempo_teste
        resultados.append(r)
        previsoes.append(y_prev)
    return resultados, previsoes


# ---------------------------------------------------------------------------
# MÉDIA e DESVIO PADRÃO entre os k folds
#
#            1   k                         ______________________
#     m̄  = ---   Σ  m_i            s  =   /  1   k
#            k  i=1                      / ---   Σ  (m_i - m̄)²
#                                       √    k  i=1
#
#   m_i : valor da métrica no fold i
#   k   : número de folds (5)
#   m̄   : média - desempenho típico do modelo
#   s   : desvio padrão - quanto o resultado oscila entre os folds
#         (desvio pequeno = modelo estável)
#
# O resultado é apresentado como  m̄ ± s  (ex.: 0.85 ± 0.03).
# ---------------------------------------------------------------------------
def resumir(resultados_folds):
    resumo = {}
    for chave in resultados_folds[0]:
        valores = np.array([r[chave] for r in resultados_folds])
        # o ± da tabela: média e desvio dos 5 folds (np.std divide por k, desvio populacional)
        resumo[chave] = {"media": float(valores.mean()), "desvio": float(valores.std())}
    return resumo


def tabela_markdown(colunas, resumos):
    """colunas: lista de (chave, título). Tempos aparecem com 6 casas decimais."""
    titulos = ["Modelo"] + [t for _, t in colunas]
    linhas = ["| " + " | ".join(titulos) + " |", "|" + "---|" * len(titulos)]
    for modelo, r in resumos.items():
        valores = []
        for chave, _ in colunas:
            casas = 6 if chave.startswith("tempo") else 4
            valores.append(f"{r[chave]['media']:.{casas}f} ± {r[chave]['desvio']:.{casas}f}")
        linhas.append("| " + " | ".join([modelo] + valores) + " |")
    return "\n".join(linhas)


def salvar(nome_base, conteudo_json, texto_markdown):
    """Grava resultados/<nome_base>.json e resultados/<nome_base>.md"""
    os.makedirs(PASTA_RESULTADOS, exist_ok=True)
    with open(os.path.join(PASTA_RESULTADOS, nome_base + ".json"), "w", encoding="utf-8") as f:
        json.dump(conteudo_json, f, ensure_ascii=False, indent=2)
    with open(os.path.join(PASTA_RESULTADOS, nome_base + ".md"), "w", encoding="utf-8") as f:
        f.write(texto_markdown + "\n")
