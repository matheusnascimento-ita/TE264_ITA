# -*- coding: utf-8 -*-
"""
te264_historical_data.py
─────────────────────────────────────────────────────────────────────────────
Dados de calibração históricos para o portfólio CDO sintético.

Fontes (conforme README):
  • Moody's "Corporate Default and Recovery Rates 1920–2010"
  • Fed Chicago "Default Rates on Prime and Subprime Mortgages" (2010)
  • Yale Journal of Financial Crises (2011)
  • FDIC WP 2015-06; BIS WP 1101; S&P Annual Default Study

Exporta:
  • SECTOR_PARAMS — parâmetros (PD, LGD, EAD) por setor e cenário
  • gerar_portfolio_calibrado(n, cenario, seed) → DataFrame com ativos
"""

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Parâmetros setoriais calibrados
# ─────────────────────────────────────────────────────────────────────────────
# Cada setor: {cenário: {pd_mean, pd_std, lgd_mean, lgd_std, ead_mean, ead_std, peso}}
# PD/LGD estão em [0,1]; EAD em R$ mil.
SECTOR_PARAMS = {
    "Corporativo IG": {
        "normal":     {"pd": 0.015, "pd_s": 0.008, "lgd": 0.40, "lgd_s": 0.10, "ead": 5000, "ead_s": 2000, "peso": 0.20},
        "stress":     {"pd": 0.045, "pd_s": 0.020, "lgd": 0.50, "lgd_s": 0.12, "ead": 5000, "ead_s": 2000, "peso": 0.20},
        "crise_2008": {"pd": 0.085, "pd_s": 0.035, "lgd": 0.60, "lgd_s": 0.15, "ead": 5000, "ead_s": 2000, "peso": 0.20},
    },
    "Corporativo HY": {
        "normal":     {"pd": 0.045, "pd_s": 0.020, "lgd": 0.55, "lgd_s": 0.12, "ead": 3000, "ead_s": 1500, "peso": 0.15},
        "stress":     {"pd": 0.110, "pd_s": 0.040, "lgd": 0.65, "lgd_s": 0.14, "ead": 3000, "ead_s": 1500, "peso": 0.15},
        "crise_2008": {"pd": 0.200, "pd_s": 0.070, "lgd": 0.78, "lgd_s": 0.12, "ead": 3000, "ead_s": 1500, "peso": 0.15},
    },
    "RMBS Prime": {
        "normal":     {"pd": 0.008, "pd_s": 0.004, "lgd": 0.25, "lgd_s": 0.08, "ead": 800, "ead_s": 300, "peso": 0.20},
        "stress":     {"pd": 0.035, "pd_s": 0.015, "lgd": 0.40, "lgd_s": 0.10, "ead": 800, "ead_s": 300, "peso": 0.20},
        "crise_2008": {"pd": 0.070, "pd_s": 0.030, "lgd": 0.55, "lgd_s": 0.12, "ead": 800, "ead_s": 300, "peso": 0.20},
    },
    "RMBS Subprime": {
        "normal":     {"pd": 0.060, "pd_s": 0.025, "lgd": 0.65, "lgd_s": 0.10, "ead": 600, "ead_s": 250, "peso": 0.15},
        "stress":     {"pd": 0.150, "pd_s": 0.050, "lgd": 0.75, "lgd_s": 0.10, "ead": 600, "ead_s": 250, "peso": 0.15},
        "crise_2008": {"pd": 0.350, "pd_s": 0.100, "lgd": 0.90, "lgd_s": 0.06, "ead": 600, "ead_s": 250, "peso": 0.15},
    },
    "CMBS": {
        "normal":     {"pd": 0.020, "pd_s": 0.010, "lgd": 0.35, "lgd_s": 0.10, "ead": 4000, "ead_s": 2000, "peso": 0.10},
        "stress":     {"pd": 0.060, "pd_s": 0.025, "lgd": 0.50, "lgd_s": 0.12, "ead": 4000, "ead_s": 2000, "peso": 0.10},
        "crise_2008": {"pd": 0.120, "pd_s": 0.045, "lgd": 0.65, "lgd_s": 0.12, "ead": 4000, "ead_s": 2000, "peso": 0.10},
    },
    "ABS Consumer": {
        "normal":     {"pd": 0.035, "pd_s": 0.015, "lgd": 0.50, "lgd_s": 0.12, "ead": 500, "ead_s": 200, "peso": 0.10},
        "stress":     {"pd": 0.080, "pd_s": 0.030, "lgd": 0.60, "lgd_s": 0.12, "ead": 500, "ead_s": 200, "peso": 0.10},
        "crise_2008": {"pd": 0.150, "pd_s": 0.055, "lgd": 0.72, "lgd_s": 0.10, "ead": 500, "ead_s": 200, "peso": 0.10},
    },
    "Soberano/Agência": {
        "normal":     {"pd": 0.003, "pd_s": 0.002, "lgd": 0.20, "lgd_s": 0.08, "ead": 8000, "ead_s": 3000, "peso": 0.10},
        "stress":     {"pd": 0.010, "pd_s": 0.005, "lgd": 0.30, "lgd_s": 0.10, "ead": 8000, "ead_s": 3000, "peso": 0.10},
        "crise_2008": {"pd": 0.025, "pd_s": 0.012, "lgd": 0.40, "lgd_s": 0.12, "ead": 8000, "ead_s": 3000, "peso": 0.10},
    },
}

# Ratings simplificados (aproximados por faixa de PD)
RATING_MAP = [
    (0.005, "AAA"), (0.010, "AA"),  (0.020, "A"),
    (0.040, "BBB"), (0.080, "BB"),  (0.150, "B"),
    (0.300, "CCC"), (1.000, "CC/D"),
]


def _pd_to_rating(pd_val):
    """Mapeia PD crisp para rating de crédito simplificado."""
    for limiar, rating in RATING_MAP:
        if pd_val <= limiar:
            return rating
    return "CC/D"


def gerar_portfolio_calibrado(n=500, cenario="normal", seed=42):
    """
    Gera um portfólio CDO sintético com `n` ativos, calibrado ao cenário.

    Retorna DataFrame com colunas:
        id, setor, rating, pd, lgd, ead, el (perda esperada = pd*lgd*ead)
    """
    rng = np.random.default_rng(seed)
    registros = []

    setores = list(SECTOR_PARAMS.keys())
    pesos = np.array([SECTOR_PARAMS[s][cenario]["peso"] for s in setores])
    pesos /= pesos.sum()  # normaliza

    # Aloca ativos proporcionalmente ao peso de cada setor
    n_por_setor = rng.multinomial(n, pesos)

    idx = 1
    for setor, ns in zip(setores, n_por_setor):
        p = SECTOR_PARAMS[setor][cenario]
        for _ in range(ns):
            pd_val = np.clip(rng.normal(p["pd"], p["pd_s"]), 0.001, 0.999)
            lgd_val = np.clip(rng.normal(p["lgd"], p["lgd_s"]), 0.01, 0.99)
            ead_val = max(50, rng.normal(p["ead"], p["ead_s"])) * 1000  # em R$
            registros.append({
                "id": f"A{idx:04d}",
                "setor": setor,
                "rating": _pd_to_rating(pd_val),
                "pd": round(pd_val, 6),
                "lgd": round(lgd_val, 4),
                "ead": round(ead_val, 2),
                "el": round(pd_val * lgd_val * ead_val, 2),
            })
            idx += 1

    return pd.DataFrame(registros)


if __name__ == "__main__":
    for cen in ["normal", "stress", "crise_2008"]:
        df = gerar_portfolio_calibrado(n=100, cenario=cen, seed=42)
        print(f"\n=== Cenário: {cen} ===")
        print(f"  PD médio : {df['pd'].mean():.3%}")
        print(f"  LGD médio: {df['lgd'].mean():.1%}")
        print(f"  EAD total: R$ {df['ead'].sum()/1e6:.1f}M")
        print(f"  EL total : R$ {df['el'].sum()/1e6:.3f}M")
