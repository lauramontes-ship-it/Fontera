"""
=============================================================
  FRONTERA EFICIENTE DE MARKOWITZ
  Portafolio: AMZN | BTC | HSBCSGE | SP500 | GC=F
=============================================================
  Datos extraídos del archivo:
  RD4_1_2_Modelo_BASE_-_FRONTERA_EFICIENTE_-_EJERCICIO.xlsx
-------------------------------------------------------------
  Dependencias (ver requirements.txt):
    pip install -r requirements.txt

  Librerías utilizadas:
    numpy==2.4.4      → álgebra lineal, matrices de covarianza
    pandas==3.0.2     → manipulación de datos y lectura Excel
    matplotlib==3.10.8 → visualización y gráficos
    scipy==1.17.1     → optimización cuadrática (frontera)
    openpyxl>=3.1.0   → motor de lectura .xlsx (vía pandas)
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.optimize import minimize, linprog
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
#  1. DATOS DEL MODELO (extraídos del Excel)
# ─────────────────────────────────────────────────────────────

ACTIVOS = ["AMZN", "BTC", "HSBC-SGE", "SP500", "GC=F"]
N = len(ACTIVOS)

# Rentabilidades diarias
RENT_DIARIA = np.array([0.000545, 0.000833, 0.000267, 0.000518, 0.000863])

# Volatilidades diarias
VOL_DIARIA = np.array([0.022135, 0.032272, 0.009071, 0.010642, 0.010819])

# Rentabilidades anualizadas
RENT_ANUAL = np.array([0.146510, 0.232469, 0.069322, 0.138868, 0.241717])

# Volatilidades anualizadas
VOL_ANUAL = np.array([0.350689, 0.511291, 0.143709, 0.168599, 0.171409])

# Ratios de Sharpe individuales (tasa libre de riesgo ≈ 0)
I_SHARPE = np.array([0.417779, 0.454671, 0.482381, 0.823660, 1.410171])

# Matriz de covarianzas (diaria)
COV_DIARIA = np.array([
    [ 4.8997e-04,  1.2105e-05,  8.3339e-06,  1.6894e-04, -3.9888e-06],
    [ 1.2105e-05,  1.0415e-03,  1.5533e-05, -1.6981e-06,  7.2993e-06],
    [ 8.3339e-06,  1.5533e-05,  8.2280e-05,  7.9888e-06,  5.0032e-07],
    [ 1.6894e-04, -1.6981e-06,  7.9888e-06,  1.1325e-04, -2.7637e-06],
    [-3.9888e-06,  7.2993e-06,  5.0032e-07, -2.7637e-06,  1.1834e-04],
])

# Matriz de covarianzas anualizada (× 252 días)
COV_ANUAL = COV_DIARIA * 252

# Portafolio óptimo (máx. Sharpe) del modelo
MEZCLA_OPTIMA = np.array([0.233684, 0.245244, 0.118336, 0.148318, 0.254418])

# KPIs del portafolio óptimo
PORT_VOL_DIARIA  = 0.010751   # Desviación estándar diaria
PORT_RENT_DIARIA = 0.000660   # Rendimiento efectivo diario
PORT_VOL_ANUAL   = 0.170323   # Desviación estándar anualizada
PORT_RENT_ANUAL  = 0.180000   # Rendimiento anualizado
PORT_SHARPE      = 1.056816   # Índice de Sharpe del portafolio

# ─────────────────────────────────────────────────────────────
#  2. FUNCIONES DE OPTIMIZACIÓN
# ─────────────────────────────────────────────────────────────

def portafolio_rendimiento(pesos, rentabilidades=RENT_ANUAL):
    """Rentabilidad esperada del portafolio."""
    return np.dot(pesos, rentabilidades)


def portafolio_volatilidad(pesos, cov=COV_ANUAL):
    """Volatilidad (desviación estándar) anualizada del portafolio."""
    return np.sqrt(pesos @ cov @ pesos)


def neg_sharpe(pesos, rf=0.0, cov=COV_ANUAL, rentabilidades=RENT_ANUAL):
    """Negativo del ratio de Sharpe (para minimización)."""
    ret = portafolio_rendimiento(pesos, rentabilidades)
    vol = portafolio_volatilidad(pesos, cov)
    return -(ret - rf) / vol


def portafolio_minima_varianza(cov=COV_ANUAL):
    """Portafolio de mínima varianza global."""
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = [(0, 1)] * N
    w0 = np.ones(N) / N
    result = minimize(
        lambda w: portafolio_volatilidad(w, cov),
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    return result.x


def portafolio_maximo_sharpe(rf=0.0, cov=COV_ANUAL, rentabilidades=RENT_ANUAL):
    """Portafolio de máximo Sharpe ratio."""
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = [(0, 1)] * N
    w0 = np.ones(N) / N
    result = minimize(
        neg_sharpe,
        w0,
        args=(rf, cov, rentabilidades),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    return result.x


def frontera_eficiente(n_puntos=500, cov=COV_ANUAL, rentabilidades=RENT_ANUAL):
    """
    Calcula la frontera eficiente de Markowitz.
    Devuelve arrays de volatilidades y rendimientos.
    """
    ret_min = np.min(rentabilidades)
    ret_max = np.max(rentabilidades)
    objetivos = np.linspace(ret_min, ret_max, n_puntos)

    vols, rets, pesos_lista = [], [], []

    w_min = portafolio_minima_varianza(cov)
    ret_minvar = portafolio_rendimiento(w_min, rentabilidades)

    for objetivo in objetivos:
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, t=objetivo: portafolio_rendimiento(w, rentabilidades) - t},
        ]
        bounds = [(0, 1)] * N
        w0 = np.ones(N) / N
        result = minimize(
            lambda w: portafolio_volatilidad(w, cov),
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        if result.success:
            vol = portafolio_volatilidad(result.x, cov)
            vols.append(vol)
            rets.append(objetivo)
            pesos_lista.append(result.x)

    return np.array(vols), np.array(rets), pesos_lista


def simulacion_montecarlo(n_sim=8000, cov=COV_ANUAL, rentabilidades=RENT_ANUAL, seed=42):
    """Simulación Monte Carlo de portafolios aleatorios."""
    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(np.ones(N), size=n_sim)
    vols = np.array([portafolio_volatilidad(w, cov) for w in weights])
    rets = np.array([portafolio_rendimiento(w, rentabilidades) for w in weights])
    sharpes = rets / vols
    return vols, rets, sharpes, weights


# ─────────────────────────────────────────────────────────────
#  3. CÁLCULOS PRINCIPALES
# ─────────────────────────────────────────────────────────────

print("=" * 65)
print("  ANÁLISIS DE PORTAFOLIO — FRONTERA EFICIENTE DE MARKOWITZ")
print("=" * 65)

# Portafolios clave
w_minvar   = portafolio_minima_varianza()
w_maxsharpe = portafolio_maximo_sharpe()

ret_minvar  = portafolio_rendimiento(w_minvar)
vol_minvar  = portafolio_volatilidad(w_minvar)

ret_maxsharpe = portafolio_rendimiento(w_maxsharpe)
vol_maxsharpe = portafolio_volatilidad(w_maxsharpe)

# Portafolio del modelo (datos del Excel)
ret_modelo = portafolio_rendimiento(MEZCLA_OPTIMA)
vol_modelo = portafolio_volatilidad(MEZCLA_OPTIMA)

# Frontera eficiente y Monte Carlo
print("\n[1/3] Calculando frontera eficiente...")
fe_vols, fe_rets, fe_pesos = frontera_eficiente(n_puntos=500)

print("[2/3] Ejecutando simulación Monte Carlo (8 000 portafolios)...")
mc_vols, mc_rets, mc_sharpes, mc_weights = simulacion_montecarlo(n_sim=8000)

print("[3/3] Generando visualización...\n")

# ─────────────────────────────────────────────────────────────
#  4. REPORTE EN CONSOLA
# ─────────────────────────────────────────────────────────────

print("\n── ACTIVOS INDIVIDUALES ─────────────────────────────────────")
header = f"{'Activo':>12} {'Rent. Anual':>12} {'Vol. Anual':>12} {'Sharpe':>10}"
print(header)
print("─" * len(header))
for i, nombre in enumerate(ACTIVOS):
    print(f"{nombre:>12} {RENT_ANUAL[i]:>11.2%} {VOL_ANUAL[i]:>11.2%} {I_SHARPE[i]:>10.4f}")

print("\n── PORTAFOLIOS ÓPTIMOS ──────────────────────────────────────")
print(f"\n  Portafolio de MÍNIMA VARIANZA")
print(f"    Rentabilidad anual : {ret_minvar:.2%}")
print(f"    Volatilidad anual  : {vol_minvar:.2%}")
print(f"    Ratio de Sharpe    : {ret_minvar/vol_minvar:.4f}")
print(f"    Pesos              : ", end="")
for nombre, w in zip(ACTIVOS, w_minvar):
    print(f"{nombre}={w:.1%} ", end="")

print(f"\n\n  Portafolio MÁXIMO SHARPE (calculado)")
print(f"    Rentabilidad anual : {ret_maxsharpe:.2%}")
print(f"    Volatilidad anual  : {vol_maxsharpe:.2%}")
print(f"    Ratio de Sharpe    : {ret_maxsharpe/vol_maxsharpe:.4f}")
print(f"    Pesos              : ", end="")
for nombre, w in zip(ACTIVOS, w_maxsharpe):
    print(f"{nombre}={w:.1%} ", end="")

print(f"\n\n  Portafolio ÓPTIMO del modelo (Excel)")
print(f"    Rentabilidad anual : {ret_modelo:.2%}  (modelo: {PORT_RENT_ANUAL:.2%})")
print(f"    Volatilidad anual  : {vol_modelo:.2%}  (modelo: {PORT_VOL_ANUAL:.2%})")
print(f"    Ratio de Sharpe    : {ret_modelo/vol_modelo:.4f}  (modelo: {PORT_SHARPE:.4f})")
print(f"    Pesos              : ", end="")
for nombre, w in zip(ACTIVOS, MEZCLA_OPTIMA):
    print(f"{nombre}={w:.1%} ", end="")
print()

print("\n── MATRIZ DE CORRELACIONES ──────────────────────────────────")
# Correlación desde covarianza diaria
stds = np.sqrt(np.diag(COV_DIARIA))
corr = COV_DIARIA / np.outer(stds, stds)
corr_df = pd.DataFrame(corr, index=ACTIVOS, columns=ACTIVOS)
print(corr_df.round(4).to_string())

# ─────────────────────────────────────────────────────────────
#  5. VISUALIZACIÓN
# ─────────────────────────────────────────────────────────────

# Estética: oscura, financiera, sofisticada
BG       = "#0d1117"
BG2      = "#161b22"
GRID     = "#21262d"
TEXT     = "#e6edf3"
TEXT2    = "#8b949e"
ACCENT1  = "#58a6ff"   # azul — frontera eficiente
ACCENT2  = "#3fb950"   # verde — máx. Sharpe
ACCENT3  = "#f78166"   # rojo-coral — mín. varianza
ACCENT4  = "#d2a8ff"   # lila — portafolio modelo
GOLD     = "#e3b341"   # dorado — activos individuales

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    BG2,
    "axes.edgecolor":    GRID,
    "axes.labelcolor":   TEXT2,
    "xtick.color":       TEXT2,
    "ytick.color":       TEXT2,
    "text.color":        TEXT,
    "grid.color":        GRID,
    "grid.linewidth":    0.6,
    "font.family":       "DejaVu Sans",
    "font.size":         10,
})

fig = plt.figure(figsize=(18, 12), facecolor=BG)
fig.suptitle(
    "FRONTERA EFICIENTE DE MARKOWITZ",
    fontsize=20, fontweight="bold", color=TEXT, y=0.98,
    fontfamily="DejaVu Sans"
)

# ── Subtítulo con activos
sub = "  ·  ".join(ACTIVOS)
fig.text(0.5, 0.945, sub, ha="center", fontsize=11, color=TEXT2)

# Grid de subplots
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.38,
                      left=0.07, right=0.97, top=0.92, bottom=0.06)

# ── 5.1  GRÁFICO PRINCIPAL: Frontera + Monte Carlo ────────────
ax1 = fig.add_subplot(gs[0, :2])

# Nube Monte Carlo coloreada por Sharpe
sc = ax1.scatter(
    mc_vols * 100, mc_rets * 100,
    c=mc_sharpes, cmap="plasma", s=6, alpha=0.35, zorder=2,
    vmin=0, vmax=mc_sharpes.max()
)
cbar = fig.colorbar(sc, ax=ax1, pad=0.01, fraction=0.025)
cbar.set_label("Ratio de Sharpe", color=TEXT2, fontsize=9)
cbar.ax.yaxis.set_tick_params(color=TEXT2)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT2)

# Frontera eficiente
ax1.plot(
    fe_vols * 100, fe_rets * 100,
    color=ACCENT1, linewidth=2.8, zorder=5, label="Frontera Eficiente"
)

# Activos individuales
for i, nombre in enumerate(ACTIVOS):
    ax1.scatter(VOL_ANUAL[i] * 100, RENT_ANUAL[i] * 100,
                s=90, color=GOLD, zorder=8, edgecolors="white", linewidths=0.8)
    ax1.annotate(nombre,
                 xy=(VOL_ANUAL[i] * 100, RENT_ANUAL[i] * 100),
                 xytext=(6, 4), textcoords="offset points",
                 fontsize=8.5, color=GOLD, fontweight="bold")

# Portafolio mínima varianza
ax1.scatter(vol_minvar * 100, ret_minvar * 100,
            s=150, color=ACCENT3, marker="D", zorder=10,
            edgecolors="white", linewidths=1.2,
            label=f"Mín. Varianza  ({vol_minvar:.1%}, {ret_minvar:.1%})")

# Portafolio máximo Sharpe
ax1.scatter(vol_maxsharpe * 100, ret_maxsharpe * 100,
            s=200, color=ACCENT2, marker="*", zorder=10,
            edgecolors="white", linewidths=0.8,
            label=f"Máx. Sharpe  ({vol_maxsharpe:.1%}, {ret_maxsharpe:.1%})")

# Portafolio modelo Excel
ax1.scatter(vol_modelo * 100, ret_modelo * 100,
            s=160, color=ACCENT4, marker="P", zorder=10,
            edgecolors="white", linewidths=1.0,
            label=f"Portafolio Óptimo  ({vol_modelo:.1%}, {ret_modelo:.1%})")

# Capital Market Line (desde tasa libre de riesgo = 0)
rf = 0.0
slope_cml = (ret_maxsharpe - rf) / vol_maxsharpe
x_cml = np.linspace(0, vol_maxsharpe * 1.05, 100)
y_cml = rf + slope_cml * x_cml
ax1.plot(x_cml * 100, y_cml * 100, "--", color=ACCENT2,
         alpha=0.55, linewidth=1.4, zorder=4, label="CML")

ax1.set_xlabel("Volatilidad Anualizada (%)", fontsize=10)
ax1.set_ylabel("Rentabilidad Anualizada (%)", fontsize=10)
ax1.set_title("Espacio Riesgo–Rendimiento", fontsize=12, color=TEXT, pad=10)
ax1.legend(loc="upper left", fontsize=8.5, framealpha=0.25,
           edgecolor=GRID, facecolor=BG2)
ax1.grid(True, alpha=0.5)

# ── 5.2  COMPOSICIÓN DEL PORTAFOLIO ÓPTIMO ────────────────────
ax2 = fig.add_subplot(gs[0, 2])

colores_pie = [ACCENT1, ACCENT2, ACCENT3, ACCENT4, GOLD]
wedges, texts, autotexts = ax2.pie(
    MEZCLA_OPTIMA * 100,
    labels=ACTIVOS,
    autopct="%1.1f%%",
    startangle=140,
    colors=colores_pie,
    pctdistance=0.75,
    wedgeprops={"edgecolor": BG2, "linewidth": 2},
    textprops={"color": TEXT, "fontsize": 9},
)
for at in autotexts:
    at.set_fontsize(8)
    at.set_color(BG)
    at.set_fontweight("bold")

ax2.set_title("Composición del\nPortafolio Óptimo", fontsize=11, color=TEXT, pad=10)

# ── 5.3  RENTABILIDADES INDIVIDUALES ─────────────────────────
ax3 = fig.add_subplot(gs[1, 0])

y_pos = np.arange(N)
bars = ax3.barh(y_pos, RENT_ANUAL * 100, color=colores_pie,
                edgecolor=BG2, linewidth=0.8, height=0.6)
ax3.axvline(ret_modelo * 100, color=ACCENT4, linestyle="--", linewidth=1.5,
            label=f"Portafolio: {ret_modelo:.1%}")
ax3.set_yticks(y_pos)
ax3.set_yticklabels(ACTIVOS, fontsize=9)
ax3.set_xlabel("Rentabilidad Anualizada (%)")
ax3.set_title("Rentabilidades Individuales", fontsize=11, color=TEXT, pad=8)
ax3.legend(fontsize=8, framealpha=0.25, edgecolor=GRID, facecolor=BG2)
ax3.grid(True, axis="x", alpha=0.5)
for bar, v in zip(bars, RENT_ANUAL * 100):
    ax3.text(v + 0.3, bar.get_y() + bar.get_height() / 2,
             f"{v:.1f}%", va="center", fontsize=8, color=TEXT)

# ── 5.4  VOLATILIDADES INDIVIDUALES ──────────────────────────
ax4 = fig.add_subplot(gs[1, 1])

bars2 = ax4.barh(y_pos, VOL_ANUAL * 100, color=colores_pie,
                 edgecolor=BG2, linewidth=0.8, height=0.6)
ax4.axvline(vol_modelo * 100, color=ACCENT4, linestyle="--", linewidth=1.5,
            label=f"Portafolio: {vol_modelo:.1%}")
ax4.set_yticks(y_pos)
ax4.set_yticklabels(ACTIVOS, fontsize=9)
ax4.set_xlabel("Volatilidad Anualizada (%)")
ax4.set_title("Volatilidades Individuales", fontsize=11, color=TEXT, pad=8)
ax4.legend(fontsize=8, framealpha=0.25, edgecolor=GRID, facecolor=BG2)
ax4.grid(True, axis="x", alpha=0.5)
for bar, v in zip(bars2, VOL_ANUAL * 100):
    ax4.text(v + 0.5, bar.get_y() + bar.get_height() / 2,
             f"{v:.1f}%", va="center", fontsize=8, color=TEXT)

# ── 5.5  RATIOS DE SHARPE ────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])

# Sharpes individuales + portafolio
todos_nombres  = ACTIVOS + ["Portafolio\nÓptimo"]
todos_sharpes  = list(I_SHARPE) + [PORT_SHARPE]
todos_colores  = colores_pie + [ACCENT4]
y_pos2 = np.arange(len(todos_nombres))

bars3 = ax5.barh(y_pos2, todos_sharpes, color=todos_colores,
                 edgecolor=BG2, linewidth=0.8, height=0.6)
ax5.set_yticks(y_pos2)
ax5.set_yticklabels(todos_nombres, fontsize=9)
ax5.set_xlabel("Índice de Sharpe")
ax5.set_title("Ratios de Sharpe", fontsize=11, color=TEXT, pad=8)
ax5.grid(True, axis="x", alpha=0.5)
ax5.axvline(1.0, color=TEXT2, linestyle=":", linewidth=1, alpha=0.6)
ax5.text(1.0, -0.6, "Sharpe = 1", fontsize=7, color=TEXT2, ha="center")
for bar, v in zip(bars3, todos_sharpes):
    ax5.text(v + 0.02, bar.get_y() + bar.get_height() / 2,
             f"{v:.3f}", va="center", fontsize=8, color=TEXT)

# ── Pie de página ──────────────────────────────────────────
fig.text(0.5, 0.015,
         "Modelo de Markowitz  ·  Datos: RD4_1_2_Modelo_BASE_FRONTERA_EFICIENTE  ·  "
         "Monte Carlo: 8 000 portafolios  ·  Restricciones: w ≥ 0, Σw = 1",
         ha="center", fontsize=8, color=TEXT2, style="italic")

plt.savefig("frontera_eficiente_markowitz.png", dpi=160, bbox_inches="tight",
            facecolor=BG)
print("Gráfico guardado: frontera_eficiente_markowitz.png")
plt.show()

# ─────────────────────────────────────────────────────────────
#  6. TABLA RESUMEN FINAL
# ─────────────────────────────────────────────────────────────

print("\n\n── TABLA RESUMEN DEL PORTAFOLIO ÓPTIMO ─────────────────────")
resumen = pd.DataFrame({
    "Activo":          ACTIVOS,
    "Peso (%)":        (MEZCLA_OPTIMA * 100).round(2),
    "Rent. Anual (%)": (RENT_ANUAL * 100).round(2),
    "Vol. Anual (%)":  (VOL_ANUAL * 100).round(2),
    "Sharpe":          I_SHARPE.round(4),
})
resumen.set_index("Activo", inplace=True)
print(resumen.to_string())

print(f"""
── KPIs DEL PORTAFOLIO ÓPTIMO ───────────────────────────────
  Rendimiento anualizado  : {ret_modelo:.4%}
  Volatilidad anualizada  : {vol_modelo:.4%}
  Ratio de Sharpe         : {ret_modelo/vol_modelo:.4f}
  Valor en Riesgo (VaR 95%): ${-1.644854 * PORT_VOL_DIARIA * 10_000:.2f}  por $10,000 invertidos
─────────────────────────────────────────────────────────────
""")
