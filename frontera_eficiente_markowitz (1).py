"""
=============================================================
  FRONTERA EFICIENTE DE MARKOWITZ — APP STREAMLIT
  Portafolio: AMZN | BTC | HSBCSGE | SP500 | GC=F
=============================================================
  Datos extraídos del archivo:
  RD4_1_2_Modelo_BASE_-_FRONTERA_EFICIENTE_-_EJERCICIO.xlsx
-------------------------------------------------------------
  Dependencias (ver requirements.txt):
    pip install -r requirements.txt

  Ejecución local:
    streamlit run frontera_eficiente_markowitz.py

  Librerías utilizadas:
    streamlit==1.41.0  → interfaz web interactiva
    numpy==2.2.0       → álgebra lineal, matrices de covarianza
    pandas==2.2.3      → manipulación de datos
    plotly==5.24.1     → gráficos interactivos
    scipy==1.15.0      → optimización cuadrática (frontera)
=============================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.optimize import minimize

# ─────────────────────────────────────────────────────────────
#  CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Frontera Eficiente · Markowitz",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS personalizado
st.markdown("""
<style>
    .main { background-color: #0d1117; }
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-value { font-size: 1.6rem; font-weight: 700; color: #58a6ff; }
    .metric-label { font-size: 0.78rem; color: #8b949e; margin-top: 2px; }
    h1, h2, h3 { color: #e6edf3 !important; }
    .stTabs [data-baseweb="tab"] { color: #8b949e; }
    .stTabs [aria-selected="true"] { color: #58a6ff !important; border-bottom-color: #58a6ff !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  1. DATOS DEL MODELO (extraídos del Excel)
# ─────────────────────────────────────────────────────────────

ACTIVOS = ["AMZN", "BTC", "HSBC-SGE", "SP500", "GC=F"]
N = len(ACTIVOS)

RENT_ANUAL = np.array([0.146510, 0.232469, 0.069322, 0.138868, 0.241717])
VOL_ANUAL  = np.array([0.350689, 0.511291, 0.143709, 0.168599, 0.171409])
I_SHARPE   = np.array([0.417779, 0.454671, 0.482381, 0.823660, 1.410171])

COV_DIARIA = np.array([
    [ 4.8997e-04,  1.2105e-05,  8.3339e-06,  1.6894e-04, -3.9888e-06],
    [ 1.2105e-05,  1.0415e-03,  1.5533e-05, -1.6981e-06,  7.2993e-06],
    [ 8.3339e-06,  1.5533e-05,  8.2280e-05,  7.9888e-06,  5.0032e-07],
    [ 1.6894e-04, -1.6981e-06,  7.9888e-06,  1.1325e-04, -2.7637e-06],
    [-3.9888e-06,  7.2993e-06,  5.0032e-07, -2.7637e-06,  1.1834e-04],
])
COV_ANUAL = COV_DIARIA * 252

MEZCLA_OPTIMA   = np.array([0.233684, 0.245244, 0.118336, 0.148318, 0.254418])
PORT_VOL_DIARIA = 0.010751
PORT_SHARPE     = 1.056816

# ─────────────────────────────────────────────────────────────
#  2. FUNCIONES DE OPTIMIZACIÓN
# ─────────────────────────────────────────────────────────────

def port_ret(w):
    return float(np.dot(w, RENT_ANUAL))

def port_vol(w):
    return float(np.sqrt(w @ COV_ANUAL @ w))

def neg_sharpe(w, rf=0.0):
    return -(port_ret(w) - rf) / port_vol(w)

@st.cache_data
def calcular_minima_varianza():
    res = minimize(port_vol, np.ones(N)/N, method="SLSQP",
                   bounds=[(0,1)]*N,
                   constraints={"type":"eq","fun":lambda w: np.sum(w)-1})
    return res.x

@st.cache_data
def calcular_maximo_sharpe(rf=0.0):
    res = minimize(neg_sharpe, np.ones(N)/N, args=(rf,), method="SLSQP",
                   bounds=[(0,1)]*N,
                   constraints={"type":"eq","fun":lambda w: np.sum(w)-1})
    return res.x

@st.cache_data
def calcular_frontera(n_puntos=400):
    vols, rets = [], []
    for obj in np.linspace(np.min(RENT_ANUAL), np.max(RENT_ANUAL), n_puntos):
        res = minimize(port_vol, np.ones(N)/N, method="SLSQP",
                       bounds=[(0,1)]*N,
                       constraints=[
                           {"type":"eq","fun":lambda w: np.sum(w)-1},
                           {"type":"eq","fun":lambda w, t=obj: port_ret(w)-t},
                       ])
        if res.success:
            vols.append(port_vol(res.x))
            rets.append(obj)
    return np.array(vols), np.array(rets)

@st.cache_data
def montecarlo(n_sim=6000, seed=42):
    rng = np.random.default_rng(seed)
    W = rng.dirichlet(np.ones(N), size=n_sim)
    vols    = np.array([port_vol(w) for w in W])
    rets    = np.array([port_ret(w) for w in W])
    sharpes = rets / vols
    return vols, rets, sharpes

# ─────────────────────────────────────────────────────────────
#  3. SIDEBAR — PARÁMETROS
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Parámetros")
    st.markdown("---")

    rf_pct = st.slider("Tasa libre de riesgo (%)", 0.0, 8.0, 0.0, 0.25,
                       help="Tasa anualizada. Afecta CML y Sharpe óptimo.")
    rf = rf_pct / 100

    n_sim = st.select_slider("Simulaciones Monte Carlo",
                              options=[2000, 4000, 6000, 8000, 10000], value=6000)

    n_fe = st.select_slider("Puntos en la frontera",
                             options=[100, 200, 400, 600], value=400)

    mostrar_cml   = st.checkbox("Mostrar CML",              value=True)
    mostrar_activos = st.checkbox("Mostrar activos individuales", value=True)
    mostrar_modelo  = st.checkbox("Mostrar portafolio del modelo Excel", value=True)

    st.markdown("---")
    st.markdown("**Activos del portafolio**")
    for nombre, r, v, s in zip(ACTIVOS, RENT_ANUAL, VOL_ANUAL, I_SHARPE):
        st.markdown(f"· **{nombre}** — Rent: `{r:.1%}` | Vol: `{v:.1%}`")

    st.markdown("---")
    st.caption("Datos: RD4_1_2_Modelo_BASE_FRONTERA_EFICIENTE.xlsx")

# ─────────────────────────────────────────────────────────────
#  4. CÁLCULOS
# ─────────────────────────────────────────────────────────────

w_minvar    = calcular_minima_varianza()
w_maxsharpe = calcular_maximo_sharpe(rf)

ret_minvar  = port_ret(w_minvar);    vol_minvar  = port_vol(w_minvar)
ret_maxsh   = port_ret(w_maxsharpe); vol_maxsh   = port_vol(w_maxsharpe)
ret_modelo  = port_ret(MEZCLA_OPTIMA); vol_modelo = port_vol(MEZCLA_OPTIMA)

fe_vols, fe_rets = calcular_frontera(n_fe)
mc_vols, mc_rets, mc_sharpes = montecarlo(n_sim)

# ─────────────────────────────────────────────────────────────
#  5. ENCABEZADO
# ─────────────────────────────────────────────────────────────

st.markdown("# 📈 Frontera Eficiente de Markowitz")
st.markdown(
    f"**Portafolio:** {' · '.join(ACTIVOS)}  &nbsp;|&nbsp; "
    f"**Rf:** {rf_pct:.2f}%  &nbsp;|&nbsp;  "
    f"**Monte Carlo:** {n_sim:,} portafolios"
)
st.markdown("---")

# ─────────────────────────────────────────────────────────────
#  6. KPI CARDS
# ─────────────────────────────────────────────────────────────

c1, c2, c3, c4, c5 = st.columns(5)

def kpi_card(col, valor, label, color="#58a6ff"):
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color:{color}">{valor}</div>
        <div class="metric-label">{label}</div>
    </div>""", unsafe_allow_html=True)

kpi_card(c1, f"{ret_modelo:.2%}",         "Rent. Portafolio Óptimo", "#58a6ff")
kpi_card(c2, f"{vol_modelo:.2%}",          "Vol. Portafolio Óptimo",  "#f78166")
kpi_card(c3, f"{ret_modelo/vol_modelo:.4f}", "Sharpe Portafolio",       "#3fb950")
kpi_card(c4, f"{vol_minvar:.2%}",          "Vol. Mínima Varianza",    "#d2a8ff")
kpi_card(c5, f"{ret_maxsh/vol_maxsh:.4f}", f"Sharpe Máximo (Rf={rf_pct:.1f}%)", "#e3b341")

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  7. TABS
# ─────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Frontera Eficiente",
    "🥧 Composición",
    "📉 Métricas por Activo",
    "🔢 Datos & Correlaciones",
])

# ── TAB 1: FRONTERA EFICIENTE ─────────────────────────────────
with tab1:
    fig = go.Figure()

    # Monte Carlo
    fig.add_trace(go.Scatter(
        x=mc_vols*100, y=mc_rets*100,
        mode="markers",
        marker=dict(size=4, color=mc_sharpes, colorscale="Plasma",
                    opacity=0.4, showscale=True,
                    colorbar=dict(title="Sharpe", x=1.01, thickness=14,
                                  tickfont=dict(color="#8b949e"),
                                  titlefont=dict(color="#8b949e"))),
        name="Monte Carlo", hovertemplate="Vol: %{x:.2f}%<br>Ret: %{y:.2f}%<extra></extra>",
    ))

    # Frontera eficiente
    fig.add_trace(go.Scatter(
        x=fe_vols*100, y=fe_rets*100,
        mode="lines", line=dict(color="#58a6ff", width=3),
        name="Frontera Eficiente",
        hovertemplate="Vol: %{x:.2f}%<br>Ret: %{y:.2f}%<extra>Frontera</extra>",
    ))

    # CML
    if mostrar_cml:
        slope = (ret_maxsh - rf) / vol_maxsh
        x_cml = np.linspace(0, vol_maxsh * 1.1, 80)
        y_cml = rf + slope * x_cml
        fig.add_trace(go.Scatter(
            x=x_cml*100, y=y_cml*100,
            mode="lines", line=dict(color="#3fb950", width=1.8, dash="dash"),
            name=f"CML (Rf={rf_pct:.1f}%)",
        ))

    # Activos individuales
    if mostrar_activos:
        fig.add_trace(go.Scatter(
            x=VOL_ANUAL*100, y=RENT_ANUAL*100,
            mode="markers+text",
            marker=dict(size=12, color="#e3b341", symbol="circle",
                        line=dict(color="white", width=1.5)),
            text=ACTIVOS, textposition="top right",
            textfont=dict(color="#e3b341", size=11),
            name="Activos individuales",
            hovertemplate="%{text}<br>Vol: %{x:.2f}%<br>Ret: %{y:.2f}%<extra></extra>",
        ))

    # Mínima varianza
    fig.add_trace(go.Scatter(
        x=[vol_minvar*100], y=[ret_minvar*100],
        mode="markers+text",
        marker=dict(size=16, color="#f78166", symbol="diamond",
                    line=dict(color="white", width=1.5)),
        text=["Mín. Varianza"], textposition="bottom right",
        textfont=dict(color="#f78166", size=10),
        name=f"Mín. Varianza ({vol_minvar:.1%})",
        hovertemplate=f"Vol: {vol_minvar:.2%}<br>Ret: {ret_minvar:.2%}<extra>Mín. Varianza</extra>",
    ))

    # Máximo Sharpe
    fig.add_trace(go.Scatter(
        x=[vol_maxsh*100], y=[ret_maxsh*100],
        mode="markers+text",
        marker=dict(size=20, color="#3fb950", symbol="star",
                    line=dict(color="white", width=1)),
        text=["Máx. Sharpe"], textposition="top left",
        textfont=dict(color="#3fb950", size=10),
        name=f"Máx. Sharpe ({vol_maxsh:.1%})",
        hovertemplate=f"Vol: {vol_maxsh:.2%}<br>Ret: {ret_maxsh:.2%}<extra>Máx. Sharpe</extra>",
    ))

    # Portafolio modelo
    if mostrar_modelo:
        fig.add_trace(go.Scatter(
            x=[vol_modelo*100], y=[ret_modelo*100],
            mode="markers+text",
            marker=dict(size=16, color="#d2a8ff", symbol="cross",
                        line=dict(color="white", width=1.5)),
            text=["Portafolio Excel"], textposition="top right",
            textfont=dict(color="#d2a8ff", size=10),
            name=f"Portafolio Excel ({vol_modelo:.1%})",
            hovertemplate=f"Vol: {vol_modelo:.2%}<br>Ret: {ret_modelo:.2%}<extra>Portafolio Excel</extra>",
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        title=dict(text="Espacio Riesgo–Rendimiento  ·  Frontera Eficiente de Markowitz",
                   font=dict(size=16, color="#e6edf3")),
        xaxis=dict(title="Volatilidad Anualizada (%)", gridcolor="#21262d",
                   zeroline=False),
        yaxis=dict(title="Rentabilidad Anualizada (%)", gridcolor="#21262d",
                   zeroline=False),
        legend=dict(bgcolor="#161b22", bordercolor="#21262d", borderwidth=1,
                    font=dict(color="#e6edf3", size=11)),
        height=580,
        hovermode="closest",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Tabla portafolios clave
    st.markdown("#### Portafolios Clave")
    tabla_clave = pd.DataFrame({
        "Portafolio":     ["Mínima Varianza", "Máximo Sharpe", "Portafolio Excel"],
        "Rentabilidad":   [f"{ret_minvar:.2%}", f"{ret_maxsh:.2%}", f"{ret_modelo:.2%}"],
        "Volatilidad":    [f"{vol_minvar:.2%}", f"{vol_maxsh:.2%}", f"{vol_modelo:.2%}"],
        "Ratio de Sharpe":[f"{ret_minvar/vol_minvar:.4f}", f"{ret_maxsh/vol_maxsh:.4f}", f"{ret_modelo/vol_modelo:.4f}"],
    })
    st.dataframe(tabla_clave, hide_index=True, use_container_width=True)

# ── TAB 2: COMPOSICIÓN ────────────────────────────────────────
with tab2:
    col_a, col_b = st.columns([1, 1])

    with col_a:
        colores = ["#58a6ff","#3fb950","#f78166","#d2a8ff","#e3b341"]
        fig_pie = go.Figure(go.Pie(
            labels=ACTIVOS,
            values=(MEZCLA_OPTIMA*100).round(2),
            marker=dict(colors=colores,
                        line=dict(color="#0d1117", width=2.5)),
            textinfo="label+percent",
            textfont=dict(size=13),
            hole=0.35,
            pull=[0.04]*N,
        ))
        fig_pie.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            title=dict(text="Portafolio Óptimo — Mezcla Excel",
                       font=dict(color="#e6edf3", size=14)),
            legend=dict(font=dict(color="#e6edf3")),
            height=420,
            showlegend=True,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        fig_ms = go.Figure(go.Pie(
            labels=ACTIVOS,
            values=(w_maxsharpe*100).round(2),
            marker=dict(colors=colores,
                        line=dict(color="#0d1117", width=2.5)),
            textinfo="label+percent",
            textfont=dict(size=13),
            hole=0.35,
        ))
        fig_ms.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            title=dict(text=f"Portafolio Máx. Sharpe (Rf={rf_pct:.1f}%)",
                       font=dict(color="#e6edf3", size=14)),
            legend=dict(font=dict(color="#e6edf3")),
            height=420,
        )
        st.plotly_chart(fig_ms, use_container_width=True)

    # Tabla comparativa de pesos
    st.markdown("#### Pesos por Portafolio")
    df_pesos = pd.DataFrame({
        "Activo":             ACTIVOS,
        "Excel (%)":          (MEZCLA_OPTIMA*100).round(2),
        "Máx. Sharpe (%)":    (w_maxsharpe*100).round(2),
        "Mín. Varianza (%)":  (w_minvar*100).round(2),
    })
    st.dataframe(df_pesos, hide_index=True, use_container_width=True)

# ── TAB 3: MÉTRICAS POR ACTIVO ────────────────────────────────
with tab3:
    fig_met = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Rentabilidad Anualizada (%)",
                        "Volatilidad Anualizada (%)",
                        "Ratio de Sharpe"),
    )
    colores = ["#58a6ff","#3fb950","#f78166","#d2a8ff","#e3b341"]

    # Rentabilidades
    fig_met.add_trace(go.Bar(
        x=ACTIVOS, y=(RENT_ANUAL*100).round(2),
        marker_color=colores, text=[f"{v:.1f}%" for v in RENT_ANUAL*100],
        textposition="outside", showlegend=False,
    ), row=1, col=1)
    fig_met.add_hline(y=ret_modelo*100, line_dash="dash",
                      line_color="#d2a8ff", row=1, col=1,
                      annotation_text=f"Portafolio: {ret_modelo:.1%}",
                      annotation_font_color="#d2a8ff")

    # Volatilidades
    fig_met.add_trace(go.Bar(
        x=ACTIVOS, y=(VOL_ANUAL*100).round(2),
        marker_color=colores, text=[f"{v:.1f}%" for v in VOL_ANUAL*100],
        textposition="outside", showlegend=False,
    ), row=1, col=2)
    fig_met.add_hline(y=vol_modelo*100, line_dash="dash",
                      line_color="#d2a8ff", row=1, col=2,
                      annotation_text=f"Portafolio: {vol_modelo:.1%}",
                      annotation_font_color="#d2a8ff")

    # Sharpes
    todos_s = list(I_SHARPE) + [PORT_SHARPE]
    todos_n = ACTIVOS + ["Portafolio"]
    todos_c = colores + ["#d2a8ff"]
    fig_met.add_trace(go.Bar(
        x=todos_n, y=[round(v,4) for v in todos_s],
        marker_color=todos_c,
        text=[f"{v:.3f}" for v in todos_s],
        textposition="outside", showlegend=False,
    ), row=1, col=3)
    fig_met.add_hline(y=1.0, line_dash="dot", line_color="#8b949e",
                      row=1, col=3)

    fig_met.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        height=440,
        font=dict(color="#e6edf3"),
        title=dict(text="Métricas Comparativas por Activo",
                   font=dict(size=15, color="#e6edf3")),
    )
    fig_met.update_yaxes(gridcolor="#21262d")
    st.plotly_chart(fig_met, use_container_width=True)

# ── TAB 4: DATOS & CORRELACIONES ─────────────────────────────
with tab4:
    col_x, col_y = st.columns([1, 1])

    with col_x:
        st.markdown("#### Indicadores por Activo")
        df_ind = pd.DataFrame({
            "Activo":            ACTIVOS,
            "Rent. Diaria":      [f"{v:.4%}" for v in RENT_ANUAL/252],
            "Rent. Anual":       [f"{v:.2%}" for v in RENT_ANUAL],
            "Vol. Diaria":       [f"{v:.4%}" for v in VOL_ANUAL/np.sqrt(252)],
            "Vol. Anual":        [f"{v:.2%}" for v in VOL_ANUAL],
            "Sharpe":            [f"{v:.4f}" for v in I_SHARPE],
            "VaR 95% ($10k)":    [f"${-1.6449*v/np.sqrt(252)*10000:.2f}" for v in VOL_ANUAL],
        })
        st.dataframe(df_ind, hide_index=True, use_container_width=True)

    with col_y:
        st.markdown("#### Matriz de Correlaciones")
        stds = np.sqrt(np.diag(COV_DIARIA))
        corr = COV_DIARIA / np.outer(stds, stds)
        df_corr = pd.DataFrame(corr.round(4), index=ACTIVOS, columns=ACTIVOS)

        fig_corr = go.Figure(go.Heatmap(
            z=corr, x=ACTIVOS, y=ACTIVOS,
            colorscale="RdBu_r", zmid=0,
            text=corr.round(2), texttemplate="%{text}",
            textfont=dict(size=12),
            colorbar=dict(thickness=14, tickfont=dict(color="#8b949e"),
                          titlefont=dict(color="#8b949e")),
        ))
        fig_corr.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            height=320,
            font=dict(color="#e6edf3"),
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("#### Matriz de Covarianzas (anualizada)")
    df_cov = pd.DataFrame((COV_ANUAL*1e4).round(4), index=ACTIVOS, columns=ACTIVOS)
    st.dataframe(df_cov.style.background_gradient(cmap="Blues", axis=None),
                 use_container_width=True)
    st.caption("Valores × 10⁻⁴ para legibilidad")

# ─────────────────────────────────────────────────────────────
#  8. PIE DE PÁGINA
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Modelo de Markowitz (1952)  ·  Datos: RD4_1_2_Modelo_BASE_FRONTERA_EFICIENTE.xlsx  ·  "
    "Restricciones: wᵢ ≥ 0,  Σwᵢ = 1  ·  Covarianzas anualizadas × 252 días"
)
