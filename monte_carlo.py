# -*- coding: utf-8 -*-
"""Página 5: Monte Carlo de 2ª ordem — decomposição de incerteza epistêmica vs. aleatória."""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy import stats
import theme
import fuzzy_core as fc


def render():
    st.markdown(theme.badge("Etapa 5 · Propagação de Incerteza"), unsafe_allow_html=True)
    st.title("Monte Carlo de 2ª Ordem")

    st.markdown(theme.theory(
        "O Monte Carlo de 2ª ordem propaga <b>duas camadas de incerteza</b> simultaneamente:\n\n"
        "• <b>Loop externo (epistêmico)</b>: amostras dos parâmetros incertos (PD, LGD) "
        "de suas distribuições de 2ª ordem (a 'distribuição da distribuição').\n\n"
        "• <b>Loop interno (aleatório)</b>: para cada amostra de parâmetros, simula os "
        "defaults estocásticos e calcula a perda.\n\n"
        "O resultado é uma <b>família de distribuições de perda</b>, cuja largura quantifica "
        "a incerteza epistêmica."
    ), unsafe_allow_html=True)

    # ── Parâmetros ────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        pd_center = st.slider("PD central", 0.01, 0.50, 0.10, 0.01)
    with c2:
        lgd_center = st.slider("LGD central", 0.10, 0.90, 0.55, 0.05)
    with c3:
        n_ativos = st.select_slider("Nº de ativos", [50, 100, 200], value=100)

    c1, c2 = st.columns(2)
    with c1:
        n_outer = st.select_slider("Loop externo (epistêmico)", [50, 100, 200], value=100)
    with c2:
        n_inner = st.select_slider("Loop interno (aleatório)", [100, 500, 1000], value=500)

    # ── Simulação 2ª ordem ────────────────────────────────────────────────────
    with st.spinner("Executando Monte Carlo de 2ª ordem..."):
        rng = np.random.default_rng(42)

        # EAD uniforme por ativo
        ead_per_asset = 1_000_000 / n_ativos  # portfolio de R$1M

        # Distribuições de 2ª ordem para PD e LGD (Beta parametrizada)
        pd_alpha = pd_center * 30
        pd_beta_p = (1 - pd_center) * 30
        lgd_alpha = lgd_center * 15
        lgd_beta_p = (1 - lgd_center) * 15

        # Loop externo: amostras dos parâmetros
        pd_samples = rng.beta(pd_alpha, pd_beta_p, size=n_outer)
        lgd_samples = rng.beta(lgd_alpha, lgd_beta_p, size=n_outer)

        # IRC fuzzy para cada amostra
        irc_samples = np.array([
            fc.inferir_mamdani(pd_s, lgd_s, base="Especialista")
            for pd_s, lgd_s in zip(pd_samples, lgd_samples)
        ])

        # Perda por cenário (loop duplo)
        all_losses = []
        quantis_05 = []
        quantis_50 = []
        quantis_95 = []

        for i in range(n_outer):
            pd_i = pd_samples[i]
            lgd_i = lgd_samples[i]
            # Loop interno: defaults estocásticos
            u = rng.random((n_inner, n_ativos))
            defaults = (u < pd_i).astype(float)
            losses_inner = defaults.sum(axis=1) * lgd_i * ead_per_asset
            all_losses.append(losses_inner)
            quantis_05.append(np.percentile(losses_inner, 5))
            quantis_50.append(np.percentile(losses_inner, 50))
            quantis_95.append(np.percentile(losses_inner, 95))

        all_losses_flat = np.concatenate(all_losses)

    # ── Métricas ──────────────────────────────────────────────────────────────
    cols = st.columns(4)
    cols[0].markdown(theme.metric_card(
        "IRC médio", f"{irc_samples.mean():.3f}",
        fc.irc_para_categoria(irc_samples.mean())
    ), unsafe_allow_html=True)
    cols[1].markdown(theme.metric_card(
        "Perda média", f"R$ {all_losses_flat.mean()/1e3:.1f}k",
        f"de R$ 1M"
    ), unsafe_allow_html=True)
    cols[2].markdown(theme.metric_card(
        "Largura epistêmica",
        f"R$ {(np.mean(quantis_95) - np.mean(quantis_05))/1e3:.1f}k",
        "P95 − P5 médio"
    ), unsafe_allow_html=True)
    cols[3].markdown(theme.metric_card(
        "VaR 99%", f"R$ {np.percentile(all_losses_flat, 99)/1e3:.1f}k",
        "global (2ª ordem)"
    ), unsafe_allow_html=True)

    # ── Família de CDFs ───────────────────────────────────────────────────────
    st.markdown("### Família de distribuições de perda (incerteza epistêmica)")
    fig = go.Figure()

    # Plotar subconjunto de CDFs internas
    n_show = min(30, n_outer)
    for i in range(n_show):
        x_sorted = np.sort(all_losses[i]) / 1e3
        cdf = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
        fig.add_trace(go.Scatter(
            x=x_sorted, y=cdf, mode="lines",
            line=dict(color=theme.AZUL, width=0.8),
            opacity=0.15, showlegend=False))

    # CDF global
    x_global = np.sort(all_losses_flat) / 1e3
    cdf_global = np.arange(1, len(x_global) + 1) / len(x_global)
    fig.add_trace(go.Scatter(
        x=x_global, y=cdf_global, mode="lines",
        line=dict(color="#ffffff", width=2.5),
        name="CDF global (2ª ordem)"))

    fig.update_layout(
        title="CDFs individuais (cinza) vs. global (branco)",
        xaxis_title="Perda (R$ mil)", yaxis_title="F(x)",
        yaxis_range=[0, 1.05])
    theme.plotly_dark(fig, height=420)
    st.plotly_chart(fig, width='stretch')

    st.markdown(theme.theory(
        "Cada linha cinza é uma CDF de perda condicional a um cenário de parâmetros "
        "(PD, LGD). A <b>dispersão horizontal</b> dessas linhas é a <b>incerteza epistêmica</b>: "
        "não sabemos qual é a 'verdadeira' distribuição de perda. A CDF branca global "
        "integra sobre todos os cenários — é a distribuição não-condicional de 2ª ordem."
    ), unsafe_allow_html=True)

    # ── Histograma de IRC ─────────────────────────────────────────────────────
    st.markdown("### Distribuição do IRC fuzzy (loop externo)")
    fig = go.Figure()
    for cat in fc.CATEGORIAS:
        mask = np.array([fc.irc_para_categoria(v) == cat for v in irc_samples])
        if mask.any():
            fig.add_trace(go.Histogram(
                x=irc_samples[mask], name=cat,
                marker_color=theme.COR_CAT[cat], opacity=0.85,
                xbins=dict(size=0.03)))
    fig.update_layout(
        barmode="stack",
        title="IRC fuzzy amostrado a cada iteração do loop externo",
        xaxis_title="IRC", yaxis_title="Frequência")
    theme.plotly_dark(fig, height=340)
    st.plotly_chart(fig, width='stretch')

    # ── Decomposição da variância ─────────────────────────────────────────────
    st.markdown("### Decomposição da incerteza")

    # Variância total = Var_epistêmica(E_aleatória[L]) + E_epistêmica(Var_aleatória[L])
    medias_internas = np.array([l.mean() for l in all_losses])
    vars_internas = np.array([l.var() for l in all_losses])

    var_total = all_losses_flat.var()
    var_epistemica = medias_internas.var()
    var_aleatoria = vars_internas.mean()
    pct_ep = 100 * var_epistemica / (var_epistemica + var_aleatoria) if (var_epistemica + var_aleatoria) > 0 else 0
    pct_al = 100 - pct_ep

    cols = st.columns(3)
    cols[0].markdown(theme.metric_card(
        "Variância total", f"{var_total/1e6:.1f}M²",
        "Var(L)"
    ), unsafe_allow_html=True)
    cols[1].markdown(theme.metric_card(
        "% Epistêmica", f"{pct_ep:.1f}%",
        f"Var(E[L|θ]) = {var_epistemica/1e6:.1f}M²"
    ), unsafe_allow_html=True)
    cols[2].markdown(theme.metric_card(
        "% Aleatória", f"{pct_al:.1f}%",
        f"E[Var(L|θ)] = {var_aleatoria/1e6:.1f}M²"
    ), unsafe_allow_html=True)

    st.markdown(theme.formula(
        "Var(L) = Var_θ(E[L|θ]) + E_θ(Var[L|θ]) — Lei da Variância Total"
    ), unsafe_allow_html=True)

    # Pie chart
    fig = go.Figure(data=[go.Pie(
        labels=["Epistêmica", "Aleatória"],
        values=[pct_ep, pct_al],
        marker=dict(colors=[theme.AMARELO, theme.AZUL]),
        hole=0.45, textinfo="label+percent")])
    fig.update_layout(
        title="Decomposição da incerteza total",
        paper_bgcolor=theme.BG, font=dict(color=theme.TXT),
        height=320, showlegend=False)
    st.plotly_chart(fig, width='stretch')

    st.markdown(theme.success(
        f"A incerteza <b>epistêmica</b> responde por <b>{pct_ep:.1f}%</b> da variância total "
        "e a <b>aleatória</b> por <b>" + f"{pct_al:.1f}%" + "</b>. "
        "Essa decomposição é crucial: se a epistêmica domina, coletar mais dados reduz "
        "a incerteza; se a aleatória domina, precisamos de mais diversificação."
    ), unsafe_allow_html=True)

    st.caption(
        "Implementação: numpy.random (Beta, Bernoulli). Lei da Variância Total. "
        "Correspondência com smc_basics.R (disciplina)."
    )
