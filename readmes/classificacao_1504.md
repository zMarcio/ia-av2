# Classificação - OpenML 1504 steel-plates-fault

- Amostras: 1941 | Atributos usados: 26 de 33
- Removidos: Pastry, Z_Scratch, K_Scatch, Stains, Dirtiness, Bumps, TypeOfSteel_A400
- Classes: defeito comum 1268 | Other_Faults 673

| Modelo | Acurácia | Precisão | Recall | F1-Score | Tempo Treino (s) | Tempo Teste (s) |
|---|---|---|---|---|---|---|
| kNN (Dist. Euclidiana) | 0.7882 ± 0.0124 | 0.7726 ± 0.0164 | 0.7476 ± 0.0147 | 0.7561 ± 0.0143 | 0.000005 ± 0.000002 | 0.062156 ± 0.002933 |
| kNN (Dist. Manhattan) | 0.7903 ± 0.0177 | 0.7767 ± 0.0211 | 0.7461 ± 0.0221 | 0.7561 ± 0.0222 | 0.000003 ± 0.000000 | 0.057895 ± 0.000865 |
| Bayesiano (Univariado) | 0.5111 ± 0.0034 | 0.6506 ± 0.0110 | 0.6101 ± 0.0035 | 0.5017 ± 0.0049 | 0.000248 ± 0.000034 | 0.000086 ± 0.000010 |
| Bayesiano (Multivariado) | 0.6060 ± 0.0499 | 0.6912 ± 0.0116 | 0.6732 ± 0.0134 | 0.6002 ± 0.0445 | 0.000368 ± 0.000100 | 0.000546 ± 0.000020 |

