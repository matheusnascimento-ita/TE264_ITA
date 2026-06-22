# -*- coding: utf-8 -*-
"""Página 4: Cópulas — dependência Gaussiana vs. t-Student e tail dependence."""

# pyrefly: ignore [missing-import]
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy import stats
import theme


def render():
    st.markdown(theme.badge("Etapa 4 · Modelagem de Dependência"), unsafe_allow_html=True)
    st.title("Cópulas: Dependência entre Defaults")

    st.markdown(theme.theory(
        "Uma <b>cópula</b> separa a estrutura de dependência das distribuições marginais. "
        "No contexto de CDOs, a cópula define como os defaults de diferentes ativos se "
        "relacionam. A <b>cópula Gaussiana</b> (David Li, 2000) foi o modelo padrão pré-2008 "
        "— mas ela tem <b>tail dependence zero</b>: não captura o fato de que, em crises, "
        "defaults tendem a ocorrer juntos. A <b>cópula t-Student</b> corrige isso com um "
        "parâmetro adicional (graus de liberdade ν) que controla o peso da cauda."
    ), unsafe_allow_html=True)

    # ── Parâmetros ────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        rho = st.slider("Correlação ρ", 0.0, 0.95, 0.50, 0.05)
    with c2:
        nu = st.slider("ν (graus de liberdade, cópula t)", 2, 30, 4, 1)
    with c3:
        n_samples = st.select_slider("Nº de amostras", [1000, 5000, 10000], value=5000)

    # ── Simulação ─────────────────────────────────────────────────────────────
    rng = np.random.default_rng(42)

    # Cópula Gaussiana
    z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=n_samples)
    u_gauss = stats.norm.cdf(z)

    # Cópula t-Student
    g = rng.chisquare(nu, size=n_samples) / nu
    z_t = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=n_samples)
    t_samples = z_t / np.sqrt(g[:, None])
    u_t = stats.t.cdf(t_samples, nu)

    # ── Scatter plots ─────────────────────────────────────────────────────────
    st.markdown("### Amostras das cópulas no espaço [0,1]²")
    c1, c2 = st.columns(2)

    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scattergl(
            x=u_gauss[:, 0], y=u_gauss[:, 1], mode="markers",
            marker=dict(color=theme.AZUL, size=2, opacity=0.3),
            name="Gaussiana"))
        fig.update_layout(
            title=f"Cópula Gaussiana (ρ = {rho:.2f})",
            xaxis_title="U₁", yaxis_title="U₂",
            xaxis_range=[0, 1], yaxis_range=[0, 1])
        theme.plotly_dark(fig, height=400)
        st.plotly_chart(fig, width='stretch')

    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scattergl(
            x=u_t[:, 0], y=u_t[:, 1], mode="markers",
            marker=dict(color=theme.VERM, size=2, opacity=0.3),
            name="t-Student"))
        fig.update_layout(
            title=f"Cópula t-Student (ρ = {rho:.2f}, ν = {nu})",
            xaxis_title="U₁", yaxis_title="U₂",
            xaxis_range=[0, 1], yaxis_range=[0, 1])
        theme.plotly_dark(fig, height=400)
        st.plotly_chart(fig, width='stretch')

    # ── Tail dependence ───────────────────────────────────────────────────────
    st.markdown("### Tail dependence (dependência de cauda)")

    # Fórmula da tail dependence da cópula t
    if nu > 2:
        lambda_t = 2 * stats.t.cdf(
            -np.sqrt((nu + 1) * (1 - rho) / (1 + rho)), nu + 1)
    else:
        lambda_t = 2 * stats.t.cdf(
            -np.sqrt((nu + 1) * (1 - rho) / (1 + rho)), nu + 1)

    cols = st.columns(3)
    cols[0].markdown(theme.metric_card(
        "λ (Gaussiana)", "0.000", "tail dependence = 0 sempre"
    ), unsafe_allow_html=True)
    cols[1].markdown(theme.metric_card(
        "λ (t-Student)", f"{lambda_t:.4f}",
        f"ν = {nu}, ρ = {rho:.2f}"
    ), unsafe_allow_html=True)
    cols[2].markdown(theme.metric_card(
        "Diferença", f"{lambda_t:.4f}",
        "risco subestimado pela Gaussiana"
    ), unsafe_allow_html=True)

    st.markdown(theme.formula(
        f"λ_t = 2 · t_{{ν+1}}(−√((ν+1)(1−ρ)/(1+ρ))) = <b>{lambda_t:.4f}</b>"
    ), unsafe_allow_html=True)

    # ── Curva de tail dependence vs ν ─────────────────────────────────────────
    st.markdown("### λ vs. graus de liberdade ν")
    nus = np.arange(2, 31)
    lambdas = []
    for v in nus:
        l = 2 * stats.t.cdf(-np.sqrt((v + 1) * (1 - rho) / (1 + rho)), v + 1)
        lambdas.append(l)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=nus, y=lambdas, mode="lines+markers",
        line=dict(color=theme.VERM, width=2.5),
        marker=dict(color=theme.VERM, size=6),
        name="λ_t"))
    fig.add_hline(y=0, line=dict(color=theme.AZUL, width=2, dash="dash"),
                  annotation_text="λ Gaussiana = 0")
    fig.add_vline(x=nu, line=dict(color="#ffffff", width=1.5, dash="dot"),
                  annotation_text=f"ν = {nu}")
    fig.update_layout(
        title=f"Tail dependence λ(ν) para ρ = {rho:.2f}",
        xaxis_title="ν (graus de liberdade)",
        yaxis_title="λ (tail dependence)",
        yaxis_range=[0, max(lambdas) * 1.2 + 0.01])
    theme.plotly_dark(fig, height=360)
    st.plotly_chart(fig, width='stretch')

    # ── Contagem empírica nos cantos ──────────────────────────────────────────
    st.markdown("### Concentração nos cantos (evidência empírica)")
    limiar = 0.05
    corner_gauss = np.sum((u_gauss[:, 0] < limiar) & (u_gauss[:, 1] < limiar))
    corner_t = np.sum((u_t[:, 0] < limiar) & (u_t[:, 1] < limiar))
    pct_gauss = 100 * corner_gauss / n_samples
    pct_t = 100 * corner_t / n_samples

    cols = st.columns(2)
    cols[0].markdown(theme.metric_card(
        f"Canto inferior ({limiar:.0%}²)",
        f"{corner_gauss} ({pct_gauss:.2f}%)",
        "cópula Gaussiana"
    ), unsafe_allow_html=True)
    cols[1].markdown(theme.metric_card(
        f"Canto inferior ({limiar:.0%}²)",
        f"{corner_t} ({pct_t:.2f}%)",
        "cópula t-Student"
    ), unsafe_allow_html=True)

    st.markdown(theme.warning(
        f"A cópula t concentra <b>{pct_t:.2f}%</b> das amostras no canto inferior "
        f"(ambas as variáveis abaixo de {limiar:.0%}), contra <b>{pct_gauss:.2f}%</b> "
        "da Gaussiana. Em termos de CDOs, isso significa mais cenários de <b>defaults "
        "simultâneos</b> — exatamente o que destruiu as tranches Senior em 2008."
    ), unsafe_allow_html=True)

    st.markdown(theme.success(
        "A cópula t-Student é uma generalização da Gaussiana: quando ν → ∞, ela converge "
        "para a Gaussiana e λ → 0. Com ν finito (ν ≤ 10 é típico para crédito), ela "
        "captura a <b>dependência de cauda</b> que modelos tradicionais ignoram."
    ), unsafe_allow_html=True)

    st.caption(
        "Implementação: scipy.stats (norm, t, multivariate_normal) + numpy. "
        "Correspondência com copula (R)."
    )
