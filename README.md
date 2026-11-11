# Desempeño de las ventas — Proyecto Final ML for Business

[![CI](https://github.com/Cynthiss/ML-Proyecto-Final/actions/workflows/ci.yml/badge.svg)](https://github.com/Cynthiss/ML-Proyecto-Final/actions/workflows/ci.yml)
[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

**Reto 1:** analizar el desempeño de las ventas de una cadena de supermercados (enero–mayo 2018) para identificar tendencias y patrones, incluyendo el desempeño del personal de ventas (generación de ventas, tendencia, contribución y perfil para contratación), y proponer accionables de negocio.

**Integrantes:** Cynthia Santos, Antonio Valdez y Juan Diego
**Curso:** Machine Learning for Business — Facultad de Ciencias Económicas, 2026

---

## Problema de negocio

| Elemento | Definición |
|---|---|
| **Contexto** | Cadena de supermercados con 23 vendedores y ~6.7 millones de ventas en 129 días. La gerencia necesita saber cómo evolucionan las ventas y qué diferencia a sus vendedores para fijar metas, incentivos y criterios de contratación. |
| **Stakeholder** | Gerencia comercial y Recursos Humanos: fijan metas e incentivos por vendedor, la política de descuentos y el perfil de contratación. |
| **Pregunta de negocio** | ¿Cuánto venderá cada vendedor por día y qué factores, incluido su perfil (edad y antigüedad), explican su desempeño? |
| **Objetivo analítico** | Regresión: pronosticar la venta diaria de cada vendedor, comparar modelos de ML contra líneas base simples y explicar las variables con SHAP. Se complementa con un análisis estadístico de tendencia, contribución, descuentos y perfil. |
| **Métrica de éxito** | *Técnica:* MAE (error absoluto medio diario por vendedor) y MAPE en los últimos 28 días, nunca vistos por el modelo. *De negocio:* un modelo de ML solo se adopta si reduce el MAE de la mejor línea base en al menos 2%; si no, se prefiere la alternativa más simple. |

## Estructura del repositorio

```
├── .github/workflows/ci.yml   # GitHub Actions: Ruff + pytest + kedro run
├── conf/
│   ├── base/                  # catálogo, parámetros y rutas (globals.yml)
│   └── ci/                    # entorno de CI: usa la muestra de data/00_sample
├── data/
│   ├── 00_sample/             # muestra versionada (0.5% de las ventas) para CI
│   ├── 01_raw/                # tablas completas del curso (NO versionadas)
│   └── 02_intermediate … 08_reporting/   # salidas generadas por kedro run
├── docs/erd.png               # diagrama entidad-relación
├── notebooks/
│   ├── 00_intro.ipynb         # introducción al dataset (material del curso)
│   └── 01_exploracion.ipynb   # exploración: calidad, outliers, patrones y decisiones
├── scripts/make_sample.py     # genera data/00_sample a partir de data/01_raw
├── src/sales_performance/pipelines/   # un paquete por pipeline (nodes.py + pipeline.py)
└── tests/                     # pruebas unitarias por pipeline + integración
```

## Instalación

Se requiere Python 3.10 o superior; el proyecto se probó con 3.12.

```bash
git clone https://github.com/Cynthiss/ML-Proyecto-Final.git
cd ML-Proyecto-Final
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e ".[dev]"
```

## Datos

Copiar las siete tablas del curso en `data/01_raw/`:

```
data/01_raw/
├── sales.csv  products.csv  categories.csv  employees.csv
└── customers.csv  cities.csv  countries.csv
```

`sales.csv` pesa ~500 MB y GitHub no acepta archivos de más de 100 MB, por eso `data/01_raw/` no se versiona. El diagrama entidad-relación está en [docs/erd.png](docs/erd.png).



## Fuentes y herramientas

- Dataset de ventas compartido por el curso.
- [Kedro](https://kedro.org), [Optuna](https://optuna.org), [LightGBM](https://lightgbm.readthedocs.io), [scikit-learn](https://scikit-learn.org), [SHAP](https://shap.readthedocs.io), [SciPy](https://scipy.org), [Ruff](https://docs.astral.sh/ruff/).
