# -*- coding: utf-8 -*-
"""Página 6: Teoria da Possibilidade — medidas Π/N, p-boxes e Dubois-Prade."""

# pyrefly: ignore [missing-import]
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import theme
import fuzzy_core as fc


def render():
    st.markdown(theme.badge("Etapa 6 · Teoria da Possibilidade"), unsafe_allow_html=True)
    st.title("Teoria da Possibilidade e P-boxes")

    st.markdown(theme.theory(
        "A <b>Teoria da Possibilidade</b> (Zadeh, 1978; Dubois & Prade, 1988) é uma alternativa "
        "à probabilidade para tratar incerteza quando os dados são <b>escassos ou imprecisos</b>. "
        "Em vez de uma distribuição de probabilidade, usamos uma <b>distribuição de possibilidade</b> "
        "π(x) ∈ [0, 1] que define duas medidas:\n\n"
        "• <b>Possibilidade Π(A)</b> = sup{π(x) : x ∈ A} — quão plausível é o evento\n\n"
        "• <b>Necessidade N(A)</b> = 1 − Π(Ā) — quão certo é o evento\n\n"
        "A relação N(A) ≤ P(A) ≤ Π(A) gera um <b>p-box</b> (probability box) que "
        "encapsula a imprecisão probabilística."
    ), unsafe_allow_html=True)

    # ── Distribuição de possibilidade do IRC ──────────────────────────────────
    st.markdown("### Distribuição de possibilidade do IRC")

    c1, c2 = st.columns(2)
    with c1:
        pd_val = st.slider("PD", 0.0, 1.0, 0.50, 0.01, key="poss_pd")
    with c2:
        lgd_val = st.slider("LGD", 0.0, 1.0, 0.60, 0.01, key="poss_lgd")

    irc_central = fc.inferir_mamdani(pd_val, lgd_val, base="Especialista")

    # Construir distribuição de possibilidade triangular centrada no IRC
    # A largura reflete a imprecisão (quanto mais nas bordas, mais incerto)
    mu_pd = fc.fuzzificar(pd_val)
    mu_lgd = fc.fuzzificar(lgd_val)
    max_mu_pd = max(mu_pd.values())
    max_mu_lgd = max(mu_lgd.values())
    # Imprecisão: menor a pertinência máxima, maior a incerteza
    imprecisao = 1.0 - min(max_mu_pd, max_mu_lgd) * 0.8
    largura = 0.08 + 0.20 * imprecisao  # spread da distribuição triangular

    x = fc.UNIVERSO
    # Distribuição de possibilidade triangular
    pi = np.maximum(0, 1 - np.abs(x - irc_central) / largura)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=pi, mode="lines", name="π(IRC)",
        line=dict(color=theme.AZUL, width=3),
        fill="tozeroy", fillcolor=f"rgba({theme._hex_to_rgb(theme.AZUL)},0.15)"))
    fig.add_vline(x=irc_central, line=dict(color="#ffffff", width=2, dash="dash"),
                  annotation_text=f"IRC = {irc_central:.3f}")
    fig.update_layout(
        title="Distribuição de possibilidade π(IRC)",
        xaxis_title="IRC", yaxis_title="π",
        yaxis_range=[0, 1.08])
    theme.plotly_dark(fig, height=360)
    st.plotly_chart(fig, width='stretch')

    cols = st.columns(3)
    cols[0].markdown(theme.metric_card(
        "IRC central", f"{irc_central:.3f}",
        fc.irc_para_categoria(irc_central)
    ), unsafe_allow_html=True)
    cols[1].markdown(theme.metric_card(
        "Largura (imprecisão)", f"{largura:.3f}",
        f"±{largura:.3f} em torno do IRC"
    ), unsafe_allow_html=True)
    cols[2].markdown(theme.metric_card(
        "Suporte [a, b]",
        f"[{max(0, irc_central-largura):.2f}, {min(1, irc_central+largura):.2f}]",
        "intervalo de possibilidade > 0"
    ), unsafe_allow_html=True)

    # ── Medidas Π e N ─────────────────────────────────────────────────────────
    st.markdown("### Medidas de Possibilidade e Necessidade")

    st.markdown("Calcule Π(A) e N(A) para o evento 'IRC ≥ limiar'.")
    limiar = st.slider("Limiar de risco", 0.0, 1.0, 0.50, 0.01, key="poss_limiar")

    # Π(IRC ≥ limiar) = sup{π(x) : x ≥ limiar}
    mask_A = x >= limiar
    if mask_A.any():
        Pi_A = pi[mask_A].max()
    else:
        Pi_A = 0.0

    # N(IRC ≥ limiar) = 1 − Π(IRC < limiar) = 1 − sup{π(x) : x < limiar}
    mask_Ac = x < limiar
    if mask_Ac.any():
        Pi_Ac = pi[mask_Ac].max()
    else:
        Pi_Ac = 0.0
    N_A = 1.0 - Pi_Ac

    cols = st.columns(3)
    cols[0].markdown(theme.metric_card(
        f"Π(IRC ≥ {limiar:.2f})", f"{Pi_A:.3f}",
        "plausibilidade"
    ), unsafe_allow_html=True)
    cols[1].markdown(theme.metric_card(
        f"N(IRC ≥ {limiar:.2f})", f"{N_A:.3f}",
        "certeza"
    ), unsafe_allow_html=True)
    cols[2].markdown(theme.metric_card(
        "Intervalo [N, Π]", f"[{N_A:.3f}, {Pi_A:.3f}]",
        "P(A) ∈ [N(A), Π(A)]"
    ), unsafe_allow_html=True)

    st.markdown(theme.formula(
        f"N(IRC ≥ {limiar:.2f}) ≤ P(IRC ≥ {limiar:.2f}) ≤ Π(IRC ≥ {limiar:.2f}) ⟹ "
        f"<b>[{N_A:.3f}, {Pi_A:.3f}]</b>"
    ), unsafe_allow_html=True)

    # ── P-box ─────────────────────────────────────────────────────────────────
    st.markdown("### P-box: limites da função de distribuição")

    st.markdown(theme.theory(
        "Um <b>p-box</b> é um par de CDFs (F̲, F̄) que limita a verdadeira CDF desconhecida. "
        "A partir da distribuição de possibilidade, usamos a <b>transformação de Dubois-Prade</b> "
        "para obter os limites:\n\n"
        "• <b>F̄(x)</b> = Π(IRC ≤ x) — limite superior (otimista)\n\n"
        "• <b>F̲(x)</b> = N(IRC ≤ x) — limite inferior (pessimista)"
    ), unsafe_allow_html=True)

    # Calcular p-box
    F_upper = np.array([pi[x <= xi].max() if (x <= xi).any() else 0.0 for xi in x])
    F_lower = np.array([1.0 - (pi[x > xi].max() if (x > xi).any() else 0.0) for xi in x])
    F_lower = np.maximum(F_lower, 0.0)

    # CDF pontual (crisp)
    F_crisp = (x <= irc_central).astype(float)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=F_upper, mode="lines", name="F̄(x) = Π(IRC ≤ x)",
        line=dict(color=theme.AZUL, width=2.5)))
    fig.add_trace(go.Scatter(
        x=x, y=F_lower, mode="lines", name="F̲(x) = N(IRC ≤ x)",
        line=dict(color=theme.VERM, width=2.5)))
    # Preencher entre as duas
    fig.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([F_upper, F_lower[::-1]]),
        fill="toself", fillcolor=f"rgba({theme._hex_to_rgb(theme.AMARELO)},0.12)",
        line=dict(width=0), name="Região de imprecisão", showlegend=True))
    fig.add_vline(x=irc_central, line=dict(color="#ffffff", width=1.5, dash="dot"),
                  annotation_text=f"IRC = {irc_central:.3f}")
    fig.update_layout(
        title="P-box: F̲(x) ≤ F(x) ≤ F̄(x)",
        xaxis_title="IRC", yaxis_title="F(x)",
        yaxis_range=[0, 1.05])
    theme.plotly_dark(fig, height=400)
    st.plotly_chart(fig, width='stretch')

    # ── Variação da imprecisão ao longo do espaço de entrada ──────────────────
    st.markdown("### Mapa de imprecisão: onde a incerteza epistêmica é maior?")

    n_grid = 21
    pd_grid = np.linspace(0.05, 0.95, n_grid)
    lgd_grid = np.linspace(0.05, 0.95, n_grid)
    Z_imprecisao = np.zeros((n_grid, n_grid))

    for i, pv in enumerate(pd_grid):
        for j, lv in enumerate(lgd_grid):
            mu_p = fc.fuzzificar(pv)
            mu_l = fc.fuzzificar(lv)
            max_p = max(mu_p.values())
            max_l = max(mu_l.values())
            Z_imprecisao[i, j] = 1.0 - min(max_p, max_l) * 0.8

    fig = go.Figure(data=go.Heatmap(
        z=Z_imprecisao, x=np.round(lgd_grid, 2), y=np.round(pd_grid, 2),
        colorscale=[[0, theme.VERDE], [0.5, theme.AMARELO], [1, theme.VERM]],
        colorbar=dict(title="Imprecisão"),
        xgap=1, ygap=1))
    fig.update_layout(
        title="Imprecisão epistêmica em função de (PD, LGD)",
        xaxis_title="LGD →", yaxis_title="PD →")
    theme.plotly_dark(fig, height=400)
    st.plotly_chart(fig, width='stretch')

    st.markdown(theme.success(
        "A imprecisão é <b>maior nas regiões de transição</b> (entre categorias fuzzy), "
        "onde os graus de pertinência são intermediários. Nos centros das categorias "
        "(ex: PD = 0 ou PD = 1/3), a pertinência é máxima e a imprecisão é menor. "
        "Isso é intuitivo: a incerteza epistêmica é maior onde a classificação é ambígua."
    ), unsafe_allow_html=True)

    st.caption(
        "Implementação: numpy (manual). Transformação Dubois-Prade. "
        "Correspondência com HYRISK (R)."
    )
