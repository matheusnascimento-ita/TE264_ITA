# -*- coding: utf-8 -*-
"""Página 1: Variáveis linguísticas fuzzy — funções de pertinência triangulares."""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import theme
import fuzzy_core as fc


def _plot_mf(titulo, universo, mfs, valor=None):
    """Gráfico das 4 funções de pertinência + valor crisp opcional."""
    fig = go.Figure()
    for cat in fc.CATEGORIAS:
        cor = theme.COR_CAT[cat]
        fig.add_trace(go.Scatter(
            x=universo, y=mfs[cat], name=cat, mode="lines",
            line=dict(color=cor, width=2.5),
            fill="tozeroy", fillcolor=f"rgba({theme._hex_to_rgb(cor)},0.12)"))

    if valor is not None:
        fig.add_vline(
            x=valor, line=dict(color="#ffffff", width=2, dash="dash"),
            annotation_text=f"{titulo} = {valor:.2f}",
            annotation_position="top")

    fig.update_layout(
        title=f"Funções de pertinência — {titulo}",
        xaxis_title=f"{titulo} [0, 1]",
        yaxis_title="Grau de pertinência μ",
        yaxis_range=[0, 1.08],
    )
    theme.plotly_dark(fig, height=340)
    return fig


def render():
    st.markdown(theme.badge("Etapa 1 · Fuzzificação"), unsafe_allow_html=True)
    st.title("Variáveis Linguísticas Fuzzy")

    st.markdown(theme.theory(
        "Em lógica fuzzy, cada variável (PD, LGD, IRC) é descrita por <b>categorias "
        "linguísticas</b> — Baixo, Moderado, Alto, Crítico. Cada categoria tem uma "
        "<b>função de pertinência</b> que mapeia valores numéricos em graus de pertinência "
        "μ ∈ [0, 1]. Usamos <b>funções triangulares</b> (trimf), como no artigo (Seção 3.3). "
        "As 4 funções cobrem uniformemente o intervalo [0, 1], com sobreposição — um valor "
        "pode pertencer a duas categorias simultaneamente (com graus diferentes)."
    ), unsafe_allow_html=True)

    mfs = fc.funcoes_pertinencia()

    # ── MFs das três variáveis (mesmos parâmetros, mesma escala) ──────────────
    st.markdown("### Funções de pertinência triangulares")
    st.caption("PD, LGD e IRC compartilham a mesma escala linguística [0, 1] e os mesmos parâmetros.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.plotly_chart(_plot_mf("PD", fc.UNIVERSO, mfs), width='stretch')
    with c2:
        st.plotly_chart(_plot_mf("LGD", fc.UNIVERSO, mfs), width='stretch')
    with c3:
        st.plotly_chart(_plot_mf("IRC", fc.UNIVERSO, mfs), width='stretch')

    # ── Parâmetros ────────────────────────────────────────────────────────────
    st.markdown("### Parâmetros das funções triangulares (a, m, b)")
    for cat in fc.CATEGORIAS:
        a, m, b = fc.MF_PARAMS[cat]
        cor = theme.COR_CAT[cat]
        st.markdown(
            f"<div class='rule-fire'>"
            f"{theme.cat_pill(cat)} "
            f"<span style='color:{cor}'>trimf({a:.2f}, {m:.2f}, {b:.2f})</span>"
            f"</div>",
            unsafe_allow_html=True)

    st.markdown(theme.formula(
        "μ_cat(x) = max(0, min((x − a)/(m − a), (b − x)/(b − m)))"
    ), unsafe_allow_html=True)

    # ── Fuzzificação interativa ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Fuzzificação interativa")
    st.markdown("Mova os sliders para ver como um valor crisp é fuzzificado em graus de pertinência.")

    c1, c2 = st.columns(2)
    with c1:
        pd_val = st.slider("Valor de PD", 0.0, 1.0, 0.35, 0.01, key="fv_pd")
    with c2:
        lgd_val = st.slider("Valor de LGD", 0.0, 1.0, 0.60, 0.01, key="fv_lgd")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(_plot_mf("PD", fc.UNIVERSO, mfs, pd_val), width='stretch')
        mu_pd = fc.fuzzificar(pd_val)
        for cat in fc.CATEGORIAS:
            if mu_pd[cat] > 1e-3:
                st.markdown(
                    f"<div class='rule-fire'>{theme.cat_pill(cat)} "
                    f"<span style='color:{theme.COR_CAT[cat]}'>μ = {mu_pd[cat]:.4f}</span></div>",
                    unsafe_allow_html=True)

    with c2:
        st.plotly_chart(_plot_mf("LGD", fc.UNIVERSO, mfs, lgd_val), width='stretch')
        mu_lgd = fc.fuzzificar(lgd_val)
        for cat in fc.CATEGORIAS:
            if mu_lgd[cat] > 1e-3:
                st.markdown(
                    f"<div class='rule-fire'>{theme.cat_pill(cat)} "
                    f"<span style='color:{theme.COR_CAT[cat]}'>μ = {mu_lgd[cat]:.4f}</span></div>",
                    unsafe_allow_html=True)

    st.markdown(theme.success(
        "A sobreposição entre funções de pertinência adjacentes é a chave da lógica fuzzy: "
        "um valor PD = 0.35 pertence tanto a <b>Baixo</b> quanto a <b>Moderado</b>, com graus "
        "diferentes. Isso permite uma transição <b>suave</b> entre classificações, ao contrário "
        "de um sistema crisp binário."
    ), unsafe_allow_html=True)

    st.caption(
        "Implementação: fuzzy_core.funcoes_pertinencia() + fuzzificar(). "
        "Correspondência com sets::trimf (R)."
    )
