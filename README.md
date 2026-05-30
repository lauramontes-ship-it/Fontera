# Frontera Eficiente de Markowitz — App en Streamlit

App interactiva para calcular y graficar **de forma dinámica** la frontera
eficiente de Markowitz: nube de portafolios aleatorios (Monte Carlo), frontera
eficiente, portafolio de mínima varianza, portafolio de máximo Sharpe
(tangencia) y la Línea del Mercado de Capitales (CML).

## Instalación y ejecución

```bash
# 1. (opcional) crear entorno virtual
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. instalar dependencias
pip install -r requirements.txt

# 3. ejecutar la app
streamlit run markowitz_app.py
```

Se abrirá en el navegador (por defecto en http://localhost:8501).

## Fuentes de datos

En la barra lateral puedes elegir:

1. **Dataset demo** — 6 clases de activos, para ver la frontera de inmediato.
2. **Subir archivo (precios/rendimientos)** — Excel/CSV con una columna por
   activo. La primera columna puede ser la fecha. La app detecta si son precios
   o rendimientos y los anualiza según la frecuencia que indiques.
3. **Captura manual** — escribes los rendimientos esperados, volatilidades y la
   matriz de correlación de cada activo.

## Sobre el archivo del caso Nike (WACC/CAPM)

El archivo `E11_RD1_WACC__CAPM__1_.xlsx` es un caso de **costo de capital de una
sola empresa** (DCF + WACC + CAPM), no un conjunto de varios activos, que es lo
que la frontera de Markowitz necesita.

Por eso la app lo aprovecha de la forma correcta: en la barra lateral hay un
cargador **"Archivo del caso"** que **extrae automáticamente** la tasa libre de
riesgo (Rf = 5.39%), el CAPM (10.12%) y el WACC (10.03%) y los usa como
parámetros por defecto del modelo. Los activos para construir la frontera los
defines tú mediante cualquiera de las tres fuentes de arriba.

## Parámetros ajustables

- Tasa libre de riesgo (Rf) — define la pendiente de la CML y el Sharpe.
- Ventas en corto (pesos negativos) on/off.
- Número de portafolios aleatorios (Monte Carlo).
- Número de puntos de la frontera.
