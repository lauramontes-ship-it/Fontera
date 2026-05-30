"""
Frontera Eficiente de Markowitz — App interactiva en Streamlit
================================================================
Calcula y grafica de forma dinámica la frontera eficiente de Markowitz
a partir de un conjunto de activos.

Fuentes de datos admitidas:
  1. Archivo subido (Excel/CSV) con PRECIOS o RENDIMIENTOS de varios activos.
  2. Captura manual de rendimientos esperados, volatilidades y correlaciones.
  3. Dataset DEMO (para ver la frontera de inmediato).

Además, si subes el archivo del caso Nike (WACC/CAPM), la app extrae
automáticamente la tasa libre de riesgo, el CAPM y el WACC para usarlos
como parámetros por defecto.

Ejecutar con:
    streamlit run markowitz_app.py
"""

from __future__ import annotations

import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.optimize import minimize


# =============================================================================
#  MOTOR DE CÁLCULO (funciones puras, sin dependencias de Streamlit)
# =============================================================================

def annualize_from_prices(prices: pd.DataFrame, periods_per_year: int):
    """A partir de una tabla de PRECIOS devuelve (mu, cov) anualizados."""
    rets = prices.sort_index().pct_change().dropna(how="all")
    mu = rets.mean() * periods_per_year
    cov = rets.cov() * periods_per_year
    return mu, cov, rets


def annualize_from_returns(returns: pd.DataFrame, periods_per_year: int):
    """A partir de una tabla de RENDIMIENTOS periódicos devuelve (mu, cov)."""
    rets = returns.dropna(how="all")
    mu = rets.mean() * periods_per_year
    cov = rets.cov() * periods_per_year
    return mu, cov, rets


def build_cov_from_corr(vols: np.ndarray, corr: np.ndarray) -> np.ndarray:
    """Construye la matriz de covarianzas a partir de volatilidades y correlación."""
    vols = np.asarray(vols, dtype=float)
    D = np.diag(vols)
    return D @ np.asarray(corr, dtype=float) @ D


def portfolio_performance(weights, mu, cov):
    """Devuelve (retorno_esperado, volatilidad) de un portafolio."""
    w = np.asarray(weights, dtype=float)
    ret = float(w @ mu)
    vol = float(np.sqrt(w @ cov @ w))
    return ret, vol


def _bounds(n: int, allow_short: bool):
    return tuple((-1.0, 1.0) if allow_short else (0.0, 1.0) for _ in range(n))


def min_variance_portfolio(mu, cov, allow_short=False):
    n = len(mu)
    x0 = np.repeat(1.0 / n, n)
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    res = minimize(
        lambda w: w @ cov @ w,
        x0, method="SLSQP",
        bounds=_bounds(n, allow_short), constraints=cons,
    )
    return res.x


def max_sharpe_portfolio(mu, cov, rf, allow_short=False):
    n = len(mu)
    x0 = np.repeat(1.0 / n, n)
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)

    def neg_sharpe(w):
        ret, vol = portfolio_performance(w, mu, cov)
        return -(ret - rf) / vol if vol > 0 else 1e9

    res = minimize(
        neg_sharpe, x0, method="SLSQP",
        bounds=_bounds(n, allow_short), constraints=cons,
    )
    return res.x


def efficient_frontier(mu, cov, n_points=50, allow_short=False):
    """Frontera eficiente: para cada retorno objetivo minimiza la varianza."""
    mu = np.asarray(mu, dtype=float)
    n = len(mu)
    # Rango de retornos objetivo entre el activo de menor y mayor mu
    targets = np.linspace(mu.min(), mu.max(), n_points)
    vols, rets, weights = [], [], []
    x0 = np.repeat(1.0 / n, n)
    for t in targets:
        cons = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, t=t: w @ mu - t},
        )
        res = minimize(
            lambda w: w @ cov @ w,
            x0, method="SLSQP",
            bounds=_bounds(n, allow_short), constraints=cons,
        )
        if res.success:
            r, v = portfolio_performance(res.x, mu, cov)
            rets.append(r)
            vols.append(v)
            weights.append(res.x)
    return np.array(rets), np.array(vols), np.array(weights)


def random_portfolios(mu, cov, rf, n_portfolios=8000, allow_short=False, seed=42):
    """Genera portafolios aleatorios (nube de Monte Carlo)."""
    rng = np.random.default_rng(seed)
    n = len(mu)
    rets = np.empty(n_portfolios)
    vols = np.empty(n_portfolios)
    sharpes = np.empty(n_portfolios)
    for i in range(n_portfolios):
        if allow_short:
            w = rng.normal(size=n)
        else:
            w = rng.random(n)
        s = w.sum()
        w = w / s if s != 0 else np.repeat(1.0 / n, n)
        r, v = portfolio_performance(w, mu, cov)
        rets[i], vols[i] = r, v
        sharpes[i] = (r - rf) / v if v > 0 else 0.0
    return rets, vols, sharpes


# =============================================================================
#  LECTURA / EXTRACCIÓN DE DATOS
# =============================================================================

def extract_nike_case_params(file_bytes: bytes):
    """Intenta extraer Rf, CAPM y WACC del archivo del caso Nike (WACC/CAPM)."""
    params = {}
    try:
        xls = pd.ExcelFile(io.BytesIO(file_bytes))
        if "exhibit 4" in [s.lower() for s in xls.sheet_names]:
            sheet = [s for s in xls.sheet_names if s.lower() == "exhibit 4"][0]
            raw = pd.read_excel(xls, sheet_name=sheet, header=None)
            flat = raw.astype(str)
            for r in range(raw.shape[0]):
                for c in range(raw.shape[1]):
                    label = str(raw.iat[r, c]).strip().upper()
                    # Buscar valores numéricos a la derecha de las etiquetas
                    def val_right(rr, cc):
                        for k in range(cc + 1, raw.shape[1]):
                            v = raw.iat[rr, k]
                            if isinstance(v, (int, float)) and not pd.isna(v):
                                return float(v)
                        return None
                    if label == "RFR" and "rf" not in params:
                        v = val_right(r, c)
                        if v is not None:
                            params["rf"] = v
                    if label == "CAPM" and "capm" not in params:
                        v = val_right(r, c)
                        if v is not None:
                            params["capm"] = v
                    if label == "WACC" and "wacc" not in params:
                        v = val_right(r, c)
                        if v is not None:
                            params["wacc"] = v
    except Exception:
        pass
    return params


def looks_like_returns(df: pd.DataFrame) -> bool:
    """Heurística: si la mayoría de los valores están en [-1, 1] son rendimientos."""
    num = df.select_dtypes(include=[np.number])
    if num.size == 0:
        return False
    share = ((num.abs() <= 1.0).sum().sum()) / num.size
    return share > 0.8


def demo_assets():
    """Dataset demo: 6 activos con mu y matriz de correlación realistas."""
    names = ["Acciones EE.UU.", "Acciones Intl.", "Bonos", "Bienes raíces",
             "Materias primas", "Mercados emerg."]
    mu = np.array([0.10, 0.085, 0.040, 0.075, 0.055, 0.115])
    vols = np.array([0.16, 0.18, 0.05, 0.14, 0.20, 0.24])
    corr = np.array([
        [1.00, 0.75, 0.10, 0.55, 0.30, 0.65],
        [0.75, 1.00, 0.15, 0.50, 0.35, 0.70],
        [0.10, 0.15, 1.00, 0.20, 0.05, 0.10],
        [0.55, 0.50, 0.20, 1.00, 0.40, 0.45],
        [0.30, 0.35, 0.05, 0.40, 1.00, 0.40],
        [0.65, 0.70, 0.10, 0.45, 0.40, 1.00],
    ])
    cov = build_cov_from_corr(vols, corr)
    return names, mu, vols, corr, cov


# =============================================================================
#  APP DE STREAMLIT
# =============================================================================

def main():
    st.set_page_config(page_title="Frontera de Markowitz", page_icon="📈", layout="wide")

    st.title("📈 Frontera Eficiente de Markowitz")
    st.caption(
        "Optimización de portafolios — nube de Monte Carlo, frontera eficiente, "
        "mínima varianza, máximo Sharpe (tangencia) y Línea del Mercado de Capitales."
    )

    # --------------------------- Sidebar -------------------------------------
    st.sidebar.header("⚙️ Configuración")

    source = st.sidebar.radio(
        "Fuente de datos de los activos",
        ["Dataset demo", "Subir archivo (precios/rendimientos)", "Captura manual"],
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Parámetros del modelo")

    rf_default = 0.0539  # Caso Nike: tasa libre de riesgo = 5.39%
    case_params = {}

    # Archivo opcional para extraer parámetros del caso (Nike WACC/CAPM)
    case_file = st.sidebar.file_uploader(
        "Archivo del caso (opcional, p. ej. Nike WACC/CAPM) para extraer Rf",
        type=["xlsx", "xls"], key="case",
    )
    if case_file is not None:
        case_params = extract_nike_case_params(case_file.getvalue())
        if case_params.get("rf"):
            rf_default = float(case_params["rf"])
            st.sidebar.success(
                f"Parámetros extraídos del caso → Rf = {case_params['rf']:.2%}"
                + (f", CAPM = {case_params['capm']:.2%}" if case_params.get("capm") else "")
                + (f", WACC = {case_params['wacc']:.2%}" if case_params.get("wacc") else "")
            )

    rf = st.sidebar.number_input(
        "Tasa libre de riesgo (anual)", value=float(rf_default),
        min_value=-0.05, max_value=0.30, step=0.001, format="%.4f",
    )
    allow_short = st.sidebar.checkbox("Permitir ventas en corto (pesos negativos)", value=False)
    n_random = st.sidebar.slider("Portafolios aleatorios (Monte Carlo)", 1000, 30000, 8000, 1000)
    n_frontier = st.sidebar.slider("Puntos de la frontera", 20, 120, 60, 10)

    # --------------------------- Obtener mu y cov ----------------------------
    names, mu, cov = None, None, None
    rets_df = None

    if source == "Dataset demo":
        names, mu, vols, corr, cov = demo_assets()
        mu = pd.Series(mu, index=names)
        cov = pd.DataFrame(cov, index=names, columns=names)
        st.info(
            "Usando **dataset demo** de 6 clases de activos. Cambia la fuente en la "
            "barra lateral para subir tus propios datos o capturarlos manualmente."
        )

    elif source == "Subir archivo (precios/rendimientos)":
        up = st.file_uploader(
            "Sube un Excel/CSV con una columna por activo (precios históricos o rendimientos). "
            "La primera columna puede ser la fecha.",
            type=["xlsx", "xls", "csv"],
        )
        freq = st.selectbox(
            "Frecuencia de los datos (para anualizar)",
            ["Diaria (252)", "Semanal (52)", "Mensual (12)", "Anual (1)"],
            index=2,
        )
        ppy = {"Diaria (252)": 252, "Semanal (52)": 52, "Mensual (12)": 12, "Anual (1)": 1}[freq]

        if up is None:
            st.warning("Sube un archivo para continuar, o usa el dataset demo.")
            st.stop()

        if up.name.lower().endswith(".csv"):
            df = pd.read_csv(up)
        else:
            df = pd.read_excel(up)

        # Si la primera columna no es numérica, úsala como índice (fechas/etiquetas)
        if not np.issubdtype(df.iloc[:, 0].dtype, np.number):
            df = df.set_index(df.columns[0])
        df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")

        if df.shape[1] < 2:
            st.error("Se necesitan al menos 2 activos (columnas numéricas).")
            st.stop()

        data_kind = st.radio(
            "¿El archivo contiene precios o rendimientos?",
            ["Detectar automáticamente", "Precios", "Rendimientos"], horizontal=True,
        )
        if data_kind == "Detectar automáticamente":
            is_rets = looks_like_returns(df)
        else:
            is_rets = data_kind == "Rendimientos"

        if is_rets:
            mu, cov, rets_df = annualize_from_returns(df, ppy)
        else:
            mu, cov, rets_df = annualize_from_prices(df, ppy)

        names = list(mu.index)
        st.success(f"Datos cargados: {len(names)} activos · interpretados como "
                   f"{'rendimientos' if is_rets else 'precios'}.")
        with st.expander("Ver datos cargados"):
            st.dataframe(df.head(20), use_container_width=True)

    else:  # Captura manual
        st.subheader("✍️ Captura manual de activos")
        n_assets = st.number_input("Número de activos", 2, 12, 4, 1)
        default_names = [f"Activo {i+1}" for i in range(int(n_assets))]
        default_mu = [0.08, 0.10, 0.06, 0.12, 0.05, 0.09, 0.07, 0.11, 0.04, 0.13, 0.06, 0.10][:int(n_assets)]
        default_vol = [0.15, 0.20, 0.10, 0.25, 0.08, 0.18, 0.12, 0.22, 0.06, 0.27, 0.11, 0.19][:int(n_assets)]

        base = pd.DataFrame({
            "Activo": default_names,
            "Retorno esperado": default_mu,
            "Volatilidad": default_vol,
        })
        st.markdown("**Rendimientos esperados y volatilidades (anuales)**")
        edited = st.data_editor(base, use_container_width=True, hide_index=True, key="assets")

        names = edited["Activo"].tolist()
        mu_vals = edited["Retorno esperado"].to_numpy(dtype=float)
        vols = edited["Volatilidad"].to_numpy(dtype=float)

        st.markdown("**Matriz de correlación** (simétrica, diagonal = 1)")
        corr0 = np.eye(int(n_assets))
        corr_df = pd.DataFrame(corr0, index=names, columns=names)
        corr_edit = st.data_editor(corr_df, use_container_width=True, key="corr")
        corr = corr_edit.to_numpy(dtype=float)
        corr = (corr + corr.T) / 2.0  # forzar simetría
        np.fill_diagonal(corr, 1.0)

        cov = pd.DataFrame(build_cov_from_corr(vols, corr), index=names, columns=names)
        mu = pd.Series(mu_vals, index=names)

    # --------------------------- Cálculos ------------------------------------
    mu_arr = mu.to_numpy(dtype=float)
    cov_arr = cov.to_numpy(dtype=float)
    asset_vols = np.sqrt(np.diag(cov_arr))

    f_rets, f_vols, f_w = efficient_frontier(mu_arr, cov_arr, n_frontier, allow_short)
    r_rets, r_vols, r_sharpe = random_portfolios(mu_arr, cov_arr, rf, n_random, allow_short)

    w_minvar = min_variance_portfolio(mu_arr, cov_arr, allow_short)
    w_maxsharpe = max_sharpe_portfolio(mu_arr, cov_arr, rf, allow_short)
    mv_ret, mv_vol = portfolio_performance(w_minvar, mu_arr, cov_arr)
    ms_ret, ms_vol = portfolio_performance(w_maxsharpe, mu_arr, cov_arr)
    ms_sharpe = (ms_ret - rf) / ms_vol if ms_vol > 0 else 0.0

    # --------------------------- Gráfica -------------------------------------
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=r_vols, y=r_rets, mode="markers",
        marker=dict(size=4, color=r_sharpe, colorscale="Viridis", showscale=True,
                    colorbar=dict(title="Sharpe"), opacity=0.55),
        name="Portafolios aleatorios", hovertemplate="σ=%{x:.2%}<br>μ=%{y:.2%}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=f_vols, y=f_rets, mode="lines",
        line=dict(color="#111", width=3), name="Frontera eficiente",
    ))
    # Línea del Mercado de Capitales (CML)
    if ms_vol > 0:
        xmax = max(r_vols.max(), f_vols.max()) * 1.05
        fig.add_trace(go.Scatter(
            x=[0, xmax], y=[rf, rf + ms_sharpe * xmax], mode="lines",
            line=dict(color="#e63946", width=2, dash="dash"),
            name="Línea del Mercado de Capitales",
        ))
    # Activos individuales
    fig.add_trace(go.Scatter(
        x=asset_vols, y=mu_arr, mode="markers+text",
        marker=dict(size=11, color="#457b9d", symbol="diamond",
                    line=dict(color="white", width=1)),
        text=names, textposition="top center", textfont=dict(size=10),
        name="Activos individuales",
    ))
    # Portafolios óptimos
    fig.add_trace(go.Scatter(
        x=[mv_vol], y=[mv_ret], mode="markers",
        marker=dict(size=18, color="#2a9d8f", symbol="star",
                    line=dict(color="white", width=1.5)),
        name="Mínima varianza",
        hovertemplate=f"Mín. varianza<br>σ={mv_vol:.2%}<br>μ={mv_ret:.2%}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[ms_vol], y=[ms_ret], mode="markers",
        marker=dict(size=18, color="#e9c46a", symbol="star",
                    line=dict(color="black", width=1.5)),
        name="Máximo Sharpe (tangencia)",
        hovertemplate=f"Máx. Sharpe<br>σ={ms_vol:.2%}<br>μ={ms_ret:.2%}<extra></extra>",
    ))

    fig.update_layout(
        xaxis=dict(title="Riesgo (volatilidad anual σ)", tickformat=".0%"),
        yaxis=dict(title="Retorno esperado anual (μ)", tickformat=".0%"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=620, margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="white",
    )
    fig.update_xaxes(gridcolor="#eee")
    fig.update_yaxes(gridcolor="#eee")

    st.plotly_chart(fig, use_container_width=True)

    # --------------------------- Métricas ------------------------------------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tasa libre de riesgo", f"{rf:.2%}")
    c2.metric("Sharpe máximo (tangencia)", f"{ms_sharpe:.3f}")
    c3.metric("μ tangencia", f"{ms_ret:.2%}", f"σ {ms_vol:.2%}")
    c4.metric("μ mín. varianza", f"{mv_ret:.2%}", f"σ {mv_vol:.2%}")

    # --------------------------- Pesos ---------------------------------------
    st.subheader("📊 Composición de los portafolios óptimos")
    weights_df = pd.DataFrame({
        "Activo": names,
        "Máx. Sharpe (tangencia)": w_maxsharpe,
        "Mínima varianza": w_minvar,
    })
    fmt = {"Máx. Sharpe (tangencia)": "{:.2%}", "Mínima varianza": "{:.2%}"}
    st.dataframe(
        weights_df.style.format(fmt).bar(
            subset=["Máx. Sharpe (tangencia)", "Mínima varianza"], color="#a8dadc"
        ),
        use_container_width=True, hide_index=True,
    )

    # Descargar pesos
    csv = weights_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar pesos (CSV)", csv, "pesos_portafolios.csv", "text/csv")

    with st.expander("ℹ️ Notas metodológicas"):
        st.markdown(
            """
- **Frontera eficiente**: para cada retorno objetivo se minimiza la varianza del
  portafolio sujeto a que los pesos sumen 1 (optimización cuadrática con SLSQP).
- **Portafolio de mínima varianza**: el de menor riesgo posible.
- **Portafolio de máximo Sharpe (tangencia)**: maximiza la razón
  (μ − Rf) / σ; es el punto donde la **Línea del Mercado de Capitales**
  (que parte de la tasa libre de riesgo) toca la frontera.
- **Ventas en corto**: al activarlas, los pesos pueden ser negativos
  (rango −1 a 1); de lo contrario, se restringen a [0, 1].
- Los rendimientos y covarianzas se anualizan según la frecuencia indicada.
            """
        )


if __name__ == "__main__":
    main()
