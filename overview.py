# -*- coding: utf-8 -*-
"""Página 0: Visão geral, tese central e contexto da disciplina TE264."""

import streamlit as st
import theme
import fuzzy_core as fc


def render():
    st.markdown(theme.badge("Visão Geral"), unsafe_allow_html=True)
    st.title("Análise Fuzzy de Risco de Crédito Estruturado (CDOs)")

    st.markdown(theme.theory(
        "A perda de crédito depende de dois parâmetros incertos — <b>PD</b> (probabilidade de "
        "default) e <b>LGD</b> (perda dado o default). Em vez de tratá-los como números exatos, "
        "este app os modela como <b>variáveis linguísticas fuzzy</b> e os combina por um "
        "<b>Sistema de Inferência Mamdani</b> que produz um <b>Índice de Risco Composto (IRC)</b>. "
        "As <b>cópulas</b> entram como camada complementar, modelando a dependência entre defaults — "
        "o elo que os modelos pré-2008 subestimaram."
    ), unsafe_allow_html=True)

    st.markdown("### Duas dimensões de incerteza")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(theme.metric_card(
            "Incerteza Epistêmica",
            "Lógica Fuzzy",
            "imprecisão nos parâmetros PD e LGD"
        ), unsafe_allow_html=True)
        st.markdown(
            "A incerteza epistêmica vem da <b>falta de conhecimento preciso</b> sobre os "
            "parâmetros. Não sabemos com exatidão qual é a PD de um ativo — ela pode ser "
            "'Moderada' ou 'Alta', com graus de pertinência diferentes. A lógica fuzzy "
            "captura essa imprecisão de forma natural.",
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(theme.metric_card(
            "Incerteza Aleatória",
            "Cópulas + MC",
            "variabilidade intrínseca dos eventos"
        ), unsafe_allow_html=True)
        st.markdown(
            "A incerteza aleatória vem da <b>variabilidade inerente</b> dos eventos — "
            "mesmo com PD perfeitamente conhecida, defaults ainda são estocásticos. "
            "Cópulas modelam a dependência entre defaults, e o Monte Carlo de 2ª ordem "
            "propaga ambas as incertezas simultaneamente.",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Pipeline visual
    st.markdown("### Pipeline do sistema")
    st.markdown(
        "```\n"
        "  PD ──┐                                    ┌── IRC por ativo\n"
        "       ├── Fuzzificação → Mamdani → IRC ───┤\n"
        "  LGD ─┘                                    └── IRC do portfólio\n"
        "                                                    │\n"
        "  Cópula (t-Student) → Simulação MC 2ª ordem ──────┘\n"
        "          │                                          │\n"
        "          └── Teoria da Possibilidade ──── p-boxes ──┘\n"
        "```"
    )

    st.markdown("---")

    # Navegação
    st.markdown("### Estrutura do app")

    etapas = [
        ("1 · Variáveis Fuzzy", "Funções de pertinência triangulares para PD, LGD e IRC"),
        ("2 · Bases de Regras", "Três bases elicitadas (ChatGPT, Claude, Especialista) + convergência"),
        ("3 · Inferência Mamdani", "Motor de inferência passo a passo + superfície de decisão 3D"),
        ("4 · Cópulas", "Dependência Gaussiana vs. t-Student e tail dependence"),
        ("5 · Monte Carlo 2ª Ordem", "Decomposição da incerteza: epistêmica vs. aleatória"),
        ("6 · Teoria da Possibilidade", "Medidas Π/N, p-boxes e transformação de Dubois-Prade"),
        ("7 · Integração no Portfólio", "IRC fuzzy + cópula aplicados a um portfólio CDO calibrado"),
    ]

    for titulo, desc in etapas:
        st.markdown(
            f"<div style='background:{theme.BG2}; border:1px solid {theme.BORDER}; "
            f"border-radius:8px; padding:0.7rem 1rem; margin:0.3rem 0;'>"
            f"<b style='color:{theme.AZUL};'>{titulo}</b> — {desc}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    st.markdown(theme.success(
        "O objetivo é demonstrar que as ferramentas da disciplina TE264 — conjuntos fuzzy, "
        "cópulas, Monte Carlo, teoria da possibilidade — são <b>complementares</b>, não "
        "concorrentes. Cada uma trata um aspecto distinto da incerteza. Juntas, elas formam "
        "uma visão completa do risco que um único número de VaR não consegue capturar."
    ), unsafe_allow_html=True)

    st.caption(
        "TE264 — Métodos Quantitativos Aplicados ao Risco · ITA Mestrado\n\n"
        "Artigo: \"Integração de Lógica Fuzzy e Modelagem de Dependência em Risco de Crédito "
        "Estruturado: Abordagem Mamdani para PD e LGD\"\n\n"
        "Correspondência R → Python: sets/FuzzyNumbers/FRBS/lfl → scikit-fuzzy"
    )
