# -*- coding: utf-8 -*-
"""
fuzzy_core.py
─────────────────────────────────────────────────────────────────────────────
Núcleo do Sistema de Inferência Fuzzy (Mamdani) — TE264 / ITA

Implementa, em Python, exatamente o que o artigo descreve:
  • Variáveis linguísticas PD, LGD e IRC com 4 categorias
    (Baixo, Moderado, Alto, Crítico)
  • Funções de pertinência TRIANGULARES (conforme Seção 3.3 do artigo)
  • As 3 bases de regras elicitadas (ChatGPT, Claude, Especialista)
  • Inferência Mamdani transparente (fuzzificação → ativação → agregação →
    defuzzificação por centroide)
  • Superfície de decisão IRC(PD, LGD)
  • Análise de convergência/divergência entre as 3 fontes

Correspondência R → Python (conforme solicitado no escopo original):
  sets / FuzzyNumbers / FRBS / lfl  →  scikit-fuzzy (skfuzzy)
"""

import numpy as np
import skfuzzy as fuzz

# ─────────────────────────────────────────────────────────────────────────────
# 1. Variáveis linguísticas e universo de discurso
# ─────────────────────────────────────────────────────────────────────────────
CATEGORIAS = ["Baixo", "Moderado", "Alto", "Crítico"]

# Universo normalizado [0, 1] para PD, LGD e IRC.
# (PD, LGD e IRC compartilham a mesma escala linguística — Seção 3.2.4 do artigo)
UNIVERSO = np.linspace(0.0, 1.0, 201)

# Parâmetros das funções de pertinência TRIANGULARES (a, m, b).
# 4 categorias uniformemente distribuídas em [0,1], com picos em 0, 1/3, 2/3, 1.
MF_PARAMS = {
    "Baixo":    [0.00, 0.00, 0.34],
    "Moderado": [0.00, 0.34, 0.67],
    "Alto":     [0.34, 0.67, 1.00],
    "Crítico":  [0.67, 1.00, 1.00],
}

# Cores por categoria (consistente em todo o app)
COR_CATEGORIA = {
    "Baixo":    "#4ade80",   # verde
    "Moderado": "#fbbf24",   # amarelo
    "Alto":     "#fb923c",   # laranja
    "Crítico":  "#ef4444",   # vermelho
}

# Valor crisp representativo de cada categoria (pico da MF, com ajuste nas bordas)
# Usado para avaliar a matriz de regras célula a célula.
VALOR_REPRESENTATIVO = {
    "Baixo":    0.12,
    "Moderado": 0.34,
    "Alto":     0.67,
    "Crítico":  0.88,
}


def funcoes_pertinencia(universo=UNIVERSO):
    """Retorna dict {categoria: array de pertinência} sobre o universo."""
    return {cat: fuzz.trimf(universo, MF_PARAMS[cat]) for cat in CATEGORIAS}


def fuzzificar(valor, universo=UNIVERSO):
    """
    Fuzzificação (Etapa 1 do Mamdani).
    Converte um valor crisp em graus de pertinência a cada categoria.
    """
    mfs = funcoes_pertinencia(universo)
    return {cat: float(fuzz.interp_membership(universo, mfs[cat], valor))
            for cat in CATEGORIAS}


# ─────────────────────────────────────────────────────────────────────────────
# 2. Bases de regras elicitadas (matrizes 4×4 indexadas [PD][LGD])
#    Transcritas EXATAMENTE das Tabelas 3, 4 e 5/8 do artigo.
#    Ordem das linhas/colunas: Baixo, Moderado, Alto, Crítico
# ─────────────────────────────────────────────────────────────────────────────
BASES_REGRAS = {
    # Tabela 3 / Apêndice B.1.1
    "ChatGPT": [
        ["Baixo",    "Baixo",    "Moderado", "Alto"],
        ["Baixo",    "Moderado", "Alto",     "Alto"],
        ["Moderado", "Alto",     "Crítico",  "Crítico"],
        ["Alto",     "Alto",     "Crítico",  "Crítico"],
    ],
    # Tabela 4 / Apêndice B.2.1
    "Claude": [
        ["Baixo",    "Baixo",    "Moderado", "Moderado"],
        ["Baixo",    "Moderado", "Alto",     "Alto"],
        ["Moderado", "Alto",     "Crítico",  "Crítico"],
        ["Alto",     "Crítico",  "Crítico",  "Crítico"],
    ],
    # Tabela 5 / Tabela 8 / Apêndice B.3.1 / B.5  (BASE FINAL ADOTADA)
    "Especialista": [
        ["Baixo",    "Baixo",    "Moderado", "Moderado"],
        ["Moderado", "Moderado", "Alto",     "Crítico"],
        ["Moderado", "Alto",     "Crítico",  "Crítico"],
        ["Alto",     "Crítico",  "Crítico",  "Crítico"],
    ],
}

# Base final do sistema (Tabela 8) = matriz do especialista
BASE_FINAL = "Especialista"


def regra_consequente(base, pd_cat, lgd_cat):
    """Retorna a categoria de saída (IRC) da regra SE PD=pd_cat E LGD=lgd_cat."""
    i = CATEGORIAS.index(pd_cat)
    j = CATEGORIAS.index(lgd_cat)
    return BASES_REGRAS[base][i][j]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Inferência Mamdani transparente
# ─────────────────────────────────────────────────────────────────────────────
def inferir_mamdani(pd_val, lgd_val, base="Especialista",
                    universo=UNIVERSO, detalhar=False):
    """
    Sistema de inferência Mamdani completo (Seção 2.4 / 3.5 do artigo).

    Etapas:
      1. Fuzzificação das entradas PD e LGD
      2. Avaliação das regras: w_ij = min(μ_PD_i, μ_LGD_j)   (operador AND)
      3. Agregação: para cada categoria de saída, máximo das ativações,
         truncando a MF de saída (implicação de Mamdani por mínimo)
      4. Defuzzificação por centroide (centro de gravidade)

    Retorna o IRC defuzzificado em [0,1].
    Se detalhar=True, retorna também o passo a passo.
    """
    mfs = funcoes_pertinencia(universo)
    mu_pd = fuzzificar(pd_val, universo)
    mu_lgd = fuzzificar(lgd_val, universo)

    # Ativação por categoria de saída + registro das regras disparadas
    ativacao = {cat: 0.0 for cat in CATEGORIAS}
    regras_disparadas = []
    for i, pd_cat in enumerate(CATEGORIAS):
        for j, lgd_cat in enumerate(CATEGORIAS):
            w = min(mu_pd[pd_cat], mu_lgd[lgd_cat])
            out_cat = BASES_REGRAS[base][i][j]
            if w > 1e-9:
                ativacao[out_cat] = max(ativacao[out_cat], w)
                regras_disparadas.append({
                    "pd": pd_cat, "lgd": lgd_cat, "irc": out_cat,
                    "mu_pd": mu_pd[pd_cat], "mu_lgd": mu_lgd[lgd_cat], "w": w
                })

    # Agregação (máximo das MFs de saída truncadas)
    agregada = np.zeros_like(universo)
    for cat in CATEGORIAS:
        truncada = np.minimum(ativacao[cat], mfs[cat])
        agregada = np.maximum(agregada, truncada)

    # Defuzzificação por centroide
    if agregada.sum() > 1e-12:
        irc = float(fuzz.defuzz(universo, agregada, "centroid"))
    else:
        irc = 0.0

    if detalhar:
        return irc, {
            "mu_pd": mu_pd,
            "mu_lgd": mu_lgd,
            "ativacao": ativacao,
            "agregada": agregada,
            "regras_disparadas": regras_disparadas,
            "categoria_irc": irc_para_categoria(irc),
        }
    return irc


def irc_para_categoria(irc):
    """Categoria linguística dominante para um valor de IRC."""
    mu = fuzzificar(irc)
    return max(mu, key=mu.get)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Superfície de decisão IRC(PD, LGD)
# ─────────────────────────────────────────────────────────────────────────────
def superficie_decisao(base="Especialista", n=41):
    """
    Calcula a superfície de decisão do sistema: IRC para cada par (PD, LGD)
    em uma grade n×n sobre [0,1]². Retorna (pd_grid, lgd_grid, Z).
    """
    pd_grid = np.linspace(0, 1, n)
    lgd_grid = np.linspace(0, 1, n)
    Z = np.zeros((n, n))
    for a, pv in enumerate(pd_grid):
        for b, lv in enumerate(lgd_grid):
            Z[a, b] = inferir_mamdani(pv, lv, base=base)
    return pd_grid, lgd_grid, Z


# ─────────────────────────────────────────────────────────────────────────────
# 5. Análise de convergência / divergência entre as 3 fontes
# ─────────────────────────────────────────────────────────────────────────────
def matriz_categoria_para_indice(base):
    """Converte matriz de categorias em matriz de índices (0..3)."""
    return np.array([[CATEGORIAS.index(c) for c in linha]
                     for linha in BASES_REGRAS[base]])


def analise_convergencia():
    """
    Compara as 3 matrizes célula a célula.

    Retorna dict com:
      • concordancia_total : nº de células onde as 3 fontes concordam
      • concordancia_parcial : nº de células onde 2 concordam
      • divergencia_total : nº de células com 3 classificações distintas
      • celulas_divergentes : lista detalhada das células não-unânimes
      • mapa : matriz 4×4 com rótulo 'total'|'parcial'|'divergente'
    """
    fontes = ["ChatGPT", "Claude", "Especialista"]
    total = parcial = divergente = 0
    celulas = []
    mapa = [["" for _ in CATEGORIAS] for _ in CATEGORIAS]

    for i, pd_cat in enumerate(CATEGORIAS):
        for j, lgd_cat in enumerate(CATEGORIAS):
            vals = [BASES_REGRAS[f][i][j] for f in fontes]
            distintos = set(vals)
            if len(distintos) == 1:
                total += 1
                mapa[i][j] = "total"
            elif len(distintos) == 2:
                parcial += 1
                mapa[i][j] = "parcial"
                celulas.append({
                    "pd": pd_cat, "lgd": lgd_cat,
                    "ChatGPT": vals[0], "Claude": vals[1], "Especialista": vals[2],
                })
            else:
                divergente += 1
                mapa[i][j] = "divergente"
                celulas.append({
                    "pd": pd_cat, "lgd": lgd_cat,
                    "ChatGPT": vals[0], "Claude": vals[1], "Especialista": vals[2],
                })

    return {
        "concordancia_total": total,
        "concordancia_parcial": parcial,
        "divergencia_total": divergente,
        "total_combinacoes": len(CATEGORIAS) ** 2,
        "celulas_divergentes": celulas,
        "mapa": mapa,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. Sensibilidade: peso relativo de PD vs LGD
# ─────────────────────────────────────────────────────────────────────────────
def sensibilidade_pd_lgd(base="Especialista", n=21):
    """
    Mede quanto o IRC varia ao mover PD (mantendo LGD fixo no centro) versus
    mover LGD (mantendo PD fixo no centro). Quantifica a 'predominância da PD'
    discutida na Seção 4.4.1 do artigo.

    Retorna dict com grids e a razão de sensibilidade média (PD/LGD).
    """
    grid = np.linspace(0, 1, n)
    centro = 0.5

    irc_var_pd = np.array([inferir_mamdani(g, centro, base=base) for g in grid])
    irc_var_lgd = np.array([inferir_mamdani(centro, g, base=base) for g in grid])

    # amplitude de variação (range) de cada eixo
    amp_pd = irc_var_pd.max() - irc_var_pd.min()
    amp_lgd = irc_var_lgd.max() - irc_var_lgd.min()
    razao = amp_pd / amp_lgd if amp_lgd > 1e-9 else np.inf

    return {
        "grid": grid,
        "irc_var_pd": irc_var_pd,
        "irc_var_lgd": irc_var_lgd,
        "amplitude_pd": amp_pd,
        "amplitude_lgd": amp_lgd,
        "razao_pd_lgd": razao,
    }


if __name__ == "__main__":
    # Teste rápido
    print("=== Teste do núcleo fuzzy ===")
    for base in BASES_REGRAS:
        irc = inferir_mamdani(0.7, 0.7, base=base)
        print(f"{base:13s}  PD=0.70 LGD=0.70 → IRC={irc:.3f} ({irc_para_categoria(irc)})")

    print("\n=== Convergência ===")
    conv = analise_convergencia()
    print(f"Total: {conv['concordancia_total']}  "
          f"Parcial: {conv['concordancia_parcial']}  "
          f"Divergência: {conv['divergencia_total']}  "
          f"de {conv['total_combinacoes']}")
    print("Células não-unânimes:")
    for c in conv["celulas_divergentes"]:
        print(f"  PD {c['pd']:9s} + LGD {c['lgd']:9s}: "
              f"GPT={c['ChatGPT']:9s} Claude={c['Claude']:9s} Esp={c['Especialista']}")

    print("\n=== Sensibilidade ===")
    s = sensibilidade_pd_lgd()
    print(f"Amplitude IRC ao variar PD : {s['amplitude_pd']:.3f}")
    print(f"Amplitude IRC ao variar LGD: {s['amplitude_lgd']:.3f}")
    print(f"Razão PD/LGD               : {s['razao_pd_lgd']:.2f}")
