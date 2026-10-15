"""
regressao_41700.py
==================
REGRESSÃO - OpenML 41700 "CPMP-2015-regression"
(tempo de execução de algoritmos no problema de pré-empilhamento de contêineres)

O que este script faz:
  1. Lê o dataset e prepara X (atributos) e y (alvo).
  2. Mostra a análise do dataset (amostras, atributos, distribuição do alvo).
  3. Avalia 3 regressores com validação cruzada de 5 folds AGRUPADA por instância:
       - kNN (Distância Euclidiana), k = 5
       - kNN (Distância Manhattan),  k = 5
       - Regressão Linear Múltipla
  4. Mostra a tabela com média ± desvio de R², R² ajustado, MAE, RMSE,
     Tempo de treino e Tempo de teste.
  5. Salva tudo em resultados/regressao_41700.json e .md

    python src/regressao_41700.py
"""

import numpy as np

from scripts.comum import (KNNRegressor, PadronizadorZScore, gerar_folds, ler_arff, resumir,
                   salvar, tabela_markdown, top_correlacoes, validacao_cruzada)

ARQUIVO = "CPMP-2015-regression.arff"
K_FOLDS = 5
K_VIZINHOS = 5
SEMENTE = 42


# ===========================================================================
# 1. CARREGAR E PREPARAR OS DADOS
#
# Decisões de pré-processamento:
#   * instance_id -> não é atributo; é usado como GRUPO na validação cruzada
#                    (a mesma instância aparece 4 vezes, uma por algoritmo).
#   * repetition  -> constante (sempre 1), não traz informação -> removida.
#   * runstatus   -> ok/timeout/memout só é conhecido DEPOIS de rodar;
#                    usá-lo seria vazamento -> removido.
#   * algorithm   -> categórico com 4 valores; vira 3 colunas 0/1 (one-hot,
#                    a primeira categoria fica como referência).
#   * runtime     -> alvo em segundos, de 0 a 3600 e muito assimétrico.
#                    Os modelos preveem y = log(1 + runtime).
# ===========================================================================
def carregar():
    nomes_arff, linhas = ler_arff(ARQUIVO)
    i_id, i_rep, i_alg, i_rt, i_st = (nomes_arff.index(c) for c in
                                      ("instance_id", "repetition", "algorithm", "runtime", "runstatus"))

    numericas = [j for j in range(len(nomes_arff)) if j not in (i_id, i_rep, i_alg, i_rt, i_st)]
    X_num = np.array([[float(l[j]) for j in numericas] for l in linhas])

    algoritmos = sorted(set(l[i_alg] for l in linhas))
    X_alg = np.array([[1.0 if l[i_alg] == a else 0.0 for a in algoritmos[1:]] for l in linhas])

    X = np.hstack([X_num, X_alg])
    nomes = [nomes_arff[j] for j in numericas] + [f"alg={a}" for a in algoritmos[1:]]

    runtime = np.array([float(l[i_rt]) for l in linhas])
    y = np.log1p(runtime)                                  # log(1 + runtime)

    ids = [l[i_id] for l in linhas]
    mapa = {v: k for k, v in enumerate(sorted(set(ids)))}
    grupos = np.array([mapa[v] for v in ids])

    status = [l[i_st] for l in linhas]
    return X, y, runtime, grupos, nomes, algoritmos, status


# ===========================================================================
# 2. REGRESSÃO LINEAR MÚLTIPLA
#
# Modelo:
#     ŷ = β₀ + β₁·x₁ + β₂·x₂ + ... + β_p·x_p        (forma matricial: ŷ = Xβ)
#
#   ŷ   : valor previsto
#   β₀  : intercepto (valor de ŷ quando todos os x são 0)
#   β_j : coeficiente do atributo j (quanto ŷ muda se x_j aumentar 1 unidade)
#   x_j : valor do atributo j
#   p   : número de atributos
#   X   : matriz n × (p+1) com uma coluna de 1s (para o β₀) + os atributos
#
# Os β são escolhidos para MINIMIZAR a soma dos erros ao quadrado
#     SSE = Σ (y_i - ŷ_i)²
# cuja solução é a EQUAÇÃO NORMAL:
#
#     β = (Xᵗ X)⁻¹ Xᵗ y
#
#   Xᵗ       : transposta de X
#   (Xᵗ X)⁻¹ : inversa de XᵗX
#   y        : vetor com os valores reais do alvo
#
# Usamos a pseudo-inversa (np.linalg.pinv): dá o mesmo resultado quando XᵗX
# é inversível e não quebra se houver atributos colineares.
# ===========================================================================
class RegressaoLinearMultipla:
    def treinar(self, X, y):
        X1 = np.hstack([np.ones((X.shape[0], 1)), X])         # coluna de 1s -> β₀
        self.beta = np.linalg.pinv(X1.T @ X1) @ X1.T @ y      # β = (XᵗX)⁻¹ Xᵗy
        return self

    def prever(self, X):
        X1 = np.hstack([np.ones((X.shape[0], 1)), X])
        return X1 @ self.beta                                 # ŷ = Xβ


# ===========================================================================
# 3. MÉTRICAS DE REGRESSÃO
#
#   y_i : valor real da amostra i          ŷ_i : valor previsto
#   ȳ   : média dos valores reais           n   : número de amostras
#   p   : número de atributos do modelo
#
# MAE - Erro Absoluto Médio
#              1   n
#     MAE  = ---   Σ  | y_i - ŷ_i |        erro médio na unidade do alvo
#              n  i=1
#
# RMSE - Raiz do Erro Quadrático Médio
#                   _______________________
#                  /  1   n
#     RMSE  =     / ---   Σ  (y_i - ŷ_i)²   como o MAE, mas pune mais os
#                √    n  i=1                erros grandes
#
# R² - Coeficiente de Determinação
#                 SS_res          Σ (y_i - ŷ_i)²
#     R² = 1 - ---------- = 1 - -----------------
#                 SS_tot          Σ (y_i - ȳ)²
#
#   SS_res : soma dos quadrados dos resíduos (erro do modelo)
#   SS_tot : soma dos quadrados total (erro de "chutar sempre a média")
#   R² = 1 -> perfeito | R² = 0 -> igual a prever a média | R² < 0 -> pior
#
# R² AJUSTADO
#                       (1 - R²) · (n - 1)
#     R²_ajustado = 1 - ------------------
#                          n - p - 1
#
#   O R² comum sempre sobe ao colocar mais atributos, mesmo inúteis.
#   O ajustado penaliza o excesso de atributos.
# ===========================================================================
def metricas(y_real, y_prev, p):
    n = len(y_real)
    r2 = 1 - np.sum((y_real - y_prev) ** 2) / np.sum((y_real - y_real.mean()) ** 2)
    return {
        "r2": float(r2),
        "r2_ajustado": float(1 - (1 - r2) * (n - 1) / (n - p - 1)),
        "mae": float(np.mean(np.abs(y_real - y_prev))),
        "rmse": float(np.sqrt(np.mean((y_real - y_prev) ** 2))),
        # MAE em segundos: desfaz o log com expm1 (= e^y - 1)
        "mae_segundos": float(np.mean(np.abs(np.expm1(y_real) - np.expm1(y_prev)))),
    }


# ===========================================================================
# 4. EXECUÇÃO
# ===========================================================================
MODELOS = {
    "kNN (Dist. Euclidiana)": lambda: KNNRegressor(k=K_VIZINHOS, distancia="euclidiana"),
    "kNN (Dist. Manhattan)": lambda: KNNRegressor(k=K_VIZINHOS, distancia="manhattan"),
    "Regressão Linear Múltipla": lambda: RegressaoLinearMultipla(),
}

COLUNAS = [
    ("r2", "R²"), ("r2_ajustado", "R² ajustado"), ("mae", "MAE (log)"), ("rmse", "RMSE (log)"),
    ("mae_segundos", "MAE (s)"), ("tempo_treino", "Tempo Treino (s)"), ("tempo_teste", "Tempo Teste (s)"),
]

if __name__ == "__main__":
    X, y, runtime, grupos, nomes, algoritmos, status = carregar()
    p = X.shape[1]
    correlacoes = top_correlacoes(X, nomes, y)
    faixas = np.histogram(runtime, bins=[0, 1, 10, 100, 1000, 3599.99, 3600.01])[0]
    contagem_status = {s: status.count(s) for s in sorted(set(status))}

    print("=" * 70)
    print("REGRESSÃO - OpenML 41700 CPMP-2015-regression")
    print("=" * 70)
    print(f"Amostras: {X.shape[0]} ({len(np.unique(grupos))} instâncias x {len(algoritmos)} algoritmos)")
    print(f"Atributos usados: {p}")
    print("Removidos: instance_id (vira grupo), repetition, runstatus")
    print(f"runstatus: {contagem_status}")
    print(f"runtime (s): mediana {np.median(runtime):.1f} | média {runtime.mean():.1f} | máx {runtime.max():.0f}")
    print(f"Execuções no limite de 3600 s: {faixas[-1]} ({faixas[-1]/len(runtime):.0%})")
    print("Atributos mais correlacionados com log(1 + runtime):")
    for c in correlacoes:
        print(f"   {c['atributo']:<25} r = {c['r']:+.3f}")

    # --- validação cruzada ---
    folds = gerar_folds(len(y), k=K_FOLDS, semente=SEMENTE, grupos=grupos)
    funcao_metricas = lambda yr, yp: metricas(yr, yp, p)
    resumos = {}
    for nome, criar in MODELOS.items():
        resultados, _ = validacao_cruzada(criar, X, y, folds, funcao_metricas)
        resumos[nome] = resumir(resultados)

    # --- curva de k (análise complementar) ---
    curva_k = {}
    for dist in ("euclidiana", "manhattan"):
        curva_k[dist] = []
        for k in (1, 3, 5, 7, 9, 11, 15, 21):
            res, _ = validacao_cruzada(lambda: KNNRegressor(k=k, distancia=dist), X, y, folds, funcao_metricas)
            curva_k[dist].append({"k": k, "r2": resumir(res)["r2"]["media"]})

    # --- coeficientes da regressão linear (todos os dados, só para interpretar) ---
    pad = PadronizadorZScore().ajustar(X)
    beta = RegressaoLinearMultipla().treinar(pad.transformar(X), y).beta
    coeficientes = sorted([{"atributo": n, "beta": float(b)} for n, b in zip(nomes, beta[1:])],
                          key=lambda c: -abs(c["beta"]))

    tabela = tabela_markdown(COLUNAS, resumos)
    print(f"\nResultados ({K_FOLDS}-fold agrupado por instância, k = {K_VIZINHOS}, alvo = log(1 + runtime)):\n")
    print(tabela)
    print("\nMaiores coeficientes da regressão linear (atributos padronizados):")
    for c in coeficientes[:5]:
        print(f"   {c['atributo']:<25} β = {c['beta']:+.3f}")

    rotulos = ["<1", "1-10", "10-100", "100-1000", "1000-3600", "3600"]
    md = (f"# Regressão - OpenML 41700 CPMP-2015-regression\n\n"
          f"- Amostras: {X.shape[0]} | Atributos usados: {p}\n"
          f"- Alvo: log(1 + runtime) | mediana do runtime {np.median(runtime):.1f} s\n"
          f"- runstatus: {contagem_status}\n\n{tabela}\n")
    salvar("regressao_41700", {
        "amostras": int(X.shape[0]), "instancias": int(len(np.unique(grupos))), "atributos_usados": p,
        "runstatus": contagem_status,
        "runtime": {"mediana": float(np.median(runtime)), "media": float(runtime.mean())},
        "faixas_runtime": dict(zip(rotulos, faixas.tolist())),
        "correlacoes": correlacoes, "resumo": resumos, "curva_k": curva_k, "coeficientes": coeficientes,
    }, md)
    print("\nSalvo em resultados/regressao_41700.json e .md")
