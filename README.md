# Sistema de Inferência Fuzzy para Risco de Crédito Estruturado

**Abordagem Mamdani para Avaliação Conjunta de PD e LGD**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B)
![scikit-fuzzy](https://img.shields.io/badge/scikit--fuzzy-0.4.2-green)
![License](https://img.shields.io/badge/Licen%C3%A7a-MIT-lightgrey)

Implementação computacional do sistema de inferência fuzzy descrito no artigo
*"Integração de Lógica Fuzzy e Modelagem de Dependência em Risco de Crédito
Estruturado: Uma Abordagem Baseada em Inferência Mamdani para Avaliação de PD e
LGD"* (TE264 — Métodos Quantitativos Aplicados ao Risco, ITA, 2026).

O projeto materializa, em uma aplicação interativa, a tese central do trabalho: a
perda de crédito depende de dois parâmetros sujeitos a **incerteza epistêmica** — a
*Probability of Default* (PD) e a *Loss Given Default* (LGD). Em vez de tratá-los como
valores pontuais, o sistema os representa como **variáveis linguísticas fuzzy** e os
combina por um **sistema de inferência Mamdani**, gerando um **Índice de Risco Composto
(IRC)**. As **cópulas** entram como camada complementar, modelando a dependência
estatística entre eventos de inadimplência.

---

## Tese: fuzzy e cópulas como abordagens complementares

| Natureza da incerteza | Pergunta que responde | Ferramenta |
|---|---|---|
| **Epistêmica** (sobre o parâmetro) | *Em que grau este ativo pertence à categoria "alto risco"?* | Lógica fuzzy (Mamdani) |
| **Aleatória** (sobre o evento) | *Qual a chance de os defaults ocorrerem juntos?* | Cópulas + Monte Carlo |

A lógica fuzzy torna **explícita** a imprecisão dos parâmetros, enquanto as cópulas
capturam a sincronização de defaults — o elo subestimado pelos modelos que
antecederam a crise de 2008. As duas abordagens não competem: complementam-se.

---

## O sistema de inferência

- **Variáveis:** PD, LGD e IRC, todas sobre o universo normalizado `[0, 1]`.
- **Categorias linguísticas:** Baixo, Moderado, Alto e Crítico.
- **Funções de pertinência:** triangulares, permitindo pertencimento simultâneo a
  categorias adjacentes.
- **Base de regras:** matriz 4×4 (16 regras `SE PD é X E LGD é Y, ENTÃO IRC é Z`),
  obtida por elicitação de conhecimento junto a **três fontes independentes** —
  ChatGPT, Claude e um especialista humano (20+ anos de experiência). A matriz do
  especialista é a base final adotada.
- **Inferência:** operador E (AND) = mínimo; agregação por máximo; defuzzificação
  pelo **centroide**.
- **Achado central:** predominância da PD sobre a LGD na composição do risco.

---

## Funcionalidades da aplicação

A aplicação Streamlit é organizada em módulos, cada um cobrindo um aspecto da análise:

| Módulo | Conteúdo |
|---|---|
| **Visão Geral** | Contexto da crise de 2008 e a tese fuzzy + cópula |
| **Variáveis Fuzzy** | Funções de pertinência triangulares e fuzzificação interativa |
| **Bases de Regras** | As três matrizes elicitadas e a análise de convergência/divergência |
| **Inferência Mamdani** | Motor PD × LGD → IRC, passo a passo, e superfície de decisão 3D |
| **Cópulas** | Dependência entre defaults: Gaussiana vs. t-Student e dependência de cauda |
| **Monte Carlo 2ª Ordem** | Propagação conjunta de incerteza epistêmica e aleatória |
| **Teoria da Possibilidade** | Medidas de possibilidade/necessidade e *p-boxes* |
| **Integração no Portfólio** | IRC fuzzy + cópula aplicados a um portfólio de crédito calibrado |

---

## Instalação

Requer Python 3.10 ou superior.

```bash
git clone https://github.com/<usuario>/<repositorio>.git
cd <repositorio>
pip install -r requirements.txt
```

## Execução

```bash
streamlit run app.py
```

A aplicação abre no navegador (por padrão em `http://localhost:8501`). Use a barra
lateral para navegar entre os módulos.

### Uso do núcleo de inferência (sem a interface)

O motor de inferência pode ser usado diretamente em Python:

```python
import fuzzy_core as fc

# Inferência para um ativo com PD = 0,55 e LGD = 0,60
irc, detalhes = fc.inferir_mamdani(0.55, 0.60, base="Especialista", detalhar=True)

print(irc)                          # 0.622
print(detalhes["categoria_irc"])    # 'Alto'
```

---

## Estrutura do projeto

Os módulos correspondem diretamente aos códigos apresentados no **Apêndice C** do artigo.

```
.
├── app.py                     # navegação principal (Streamlit)
├── fuzzy_core.py              # motor de inferência Mamdani (Apêndice C.1–C.7)
├── theme.py                   # estilo visual e componentes de interface
├── te264_historical_data.py   # parâmetros calibrados (PD, LGD e EAD por setor)
├── requirements.txt
└── pages/
    ├── overview.py            # visão geral e tese
    ├── fuzzy_variables.py     # funções de pertinência (Apêndice C.3)
    ├── rule_bases.py          # bases de regras e convergência (Apêndice C.6)
    ├── mamdani.py             # inferência e superfície de decisão
    ├── copulas.py             # dependência — cópulas (Apêndice C.8.1)
    ├── monte_carlo.py         # Monte Carlo de 2ª ordem (Apêndice C.8.2)
    ├── possibility.py         # teoria da possibilidade / p-boxes
    └── integration.py         # IRC fuzzy + cópula no portfólio
```

---

## Correspondência entre pacotes R e Python

A literatura de referência da disciplina utiliza pacotes em R; a implementação foi
desenvolvida em Python, preservando a equivalência funcional.

| Funcionalidade | Pacote R | Biblioteca Python |
|---|---|---|
| Funções de pertinência | `sets`, `FuzzyNumbers` | `skfuzzy.trimf` |
| Fuzzificação (α-cut) | `FuzzyNumbers` | `skfuzzy.interp_membership` |
| Inferência Mamdani | `FRBS` | `skfuzzy` + `numpy` (mín–máx) |
| Defuzzificação (centroide) | `FRBS`, `lfl` | `skfuzzy.defuzz` |
| Cópulas (Gaussiana / t) | `copula` | `scipy.stats` + Cholesky |
| Monte Carlo | base R | `numpy.random` |
| Aplicação interativa | `shiny` | `streamlit` |

---

## Calibração

Os parâmetros de PD, LGD e correlação utilizados nos módulos de simulação foram
calibrados a partir de fontes históricas: Moody's *Corporate Default and Recovery
Rates* (1920–2010), Federal Reserve Bank of Chicago (2010), *Yale Journal of Financial
Crises* (2011), FDIC Working Paper 2015-06 e BIS Working Paper 1101.

---

## Como citar

> MOLON, B. F. P.; ROSA, M.; NASCIMENTO, M. A. **Integração de Lógica Fuzzy e
> Modelagem de Dependência em Risco de Crédito Estruturado: Uma Abordagem Baseada em
> Inferência Mamdani para Avaliação de PD e LGD**. Trabalho da disciplina TE264 —
> Métodos Quantitativos Aplicados ao Risco. Instituto Tecnológico de Aeronáutica
> (ITA), São José dos Campos, 2026.

```bibtex
@techreport{molon2026fuzzy,
  title        = {Integração de Lógica Fuzzy e Modelagem de Dependência em Risco de
                  Crédito Estruturado: Uma Abordagem Baseada em Inferência Mamdani
                  para Avaliação de PD e LGD},
  author       = {Molon, Breno Fernando Pereira and Rosa, Marina and
                  Nascimento, Matheus de Azevedo},
  institution  = {Instituto Tecnológico de Aeronáutica (ITA)},
  type         = {Trabalho de disciplina (TE264 --- Métodos Quantitativos Aplicados ao Risco)},
  address      = {São José dos Campos, Brasil},
  year         = {2026}
}
```

---

## Autores

- **Breno Fernando Pereira Molon**
- **Marina Rosa**
- **Matheus de Azevedo Nascimento**

Orientador: **Prof. Moacyr Machado Cardoso Junior**
Instituto Tecnológico de Aeronáutica (ITA) — TE264, 2026.

---

## Licença

Distribuído sob a licença MIT. Consulte o arquivo `LICENSE` para mais informações.
