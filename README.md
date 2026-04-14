Link data: https://grouplens.org/datasets/movielens/32m/


# Proyecto Final — Pipeline de Datos con App de IA
**Curso:** Administración de Datos  
**Universidad:** LEAD University  
**Profesor:** Alejandro Zamora
**Estudiante:** Isaac Castillo Vega

Sistema de recomendación de películas construido con un pipeline completo de datos y un modelo de inteligencia artificial (SVD - Collaborative Filtering) sobre el dataset MovieLens.

---

## Requisitos previos

- Python 3.10+
- Una cuenta en [MongoDB Atlas](https://www.mongodb.com/atlas) (gratuita)
- Una API key de [TMDB](https://www.themoviedb.org/settings/api) (gratuita)
- El dataset MovieLens (`ml-32m`) descargado de [grouplens.org](https://grouplens.org/datasets/movielens/)

---

## Instalación

**1. Clonar el repositorio:**
```bash
git clone https://github.com/Isaac0021/proyecto_final.git
cd proyecto_final
```

**2. Crear y activar el entorno virtual:**
```bash
python virtualenv venv
source venv/bin/activate        # Mac
```

**3. Instalar dependencias:**
```bash
pip install -r requirements.txt
```

**4. Configurar variables de entorno:**

Crear un archivo `.env` en la raíz del proyecto con el siguiente contenido:
```
MONGO_URI=mongodb+srv://<usuario>:<password>@cluster.mongodb.net
DB_NAME=movielens
COLLECTION_NAME=ratings
TMDB_API_KEY=<tu_api_key_de_tmdb>
```

**5. Colocar los archivos CSV del dataset** en `data/raw/`:
```
data/raw/
├── ratings.csv
├── movies.csv
├── tags.csv
└── links.csv
```

---

## Ejecución

Todos los comandos se corren desde la raíz del proyecto con el entorno virtual activo.

### Paso 1 — Cargar datos en MongoDB
```bash
python movies.py
```
Esto toma una muestra del dataset y la carga en las colecciones `ratings`, `movies`, `tags` y `links` en MongoDB.

### Paso 2 — Correr el pipeline ETL
```bash
python pipeline.py
```
Ejecuta los pasos de extracción, transformación y carga en orden. Los logs quedan guardados en `orchestration/logs/`.

### Paso 3 — Entrenar el modelo
```bash
python model/train.py
```
Entrena el modelo SVD y guarda los artefactos en `model/artifacts/`. Imprime las métricas RMSE y MAE al finalizar.

### Paso 4 — Correr la aplicación
```bash
python app/app.py
```
Abre el navegador en [http://localhost:5000](http://localhost:5000)

---

## Estructura del proyecto

```
proyecto_final/
├── .env                        # Variables de entorno (no incluido en el repo)
├── seed_loader.py              # Seed loader — carga datos crudos a MongoDB
├── pipeline.py                 # Orquestador — corre el ETL completo
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/                    # CSVs originales de MovieLens (no incluido en repo por tamaño)
│   └── processed/              # Archivos intermedios (parquet) (no incluido en repo por tamaño)
│
├── etl/
│   ├── extract.py              # Extracción desde MongoDB
│   ├── transform.py            # Limpieza y feature engineering
│   └── load.py                 # Carga a colección 'processed'
│
├── model/
│   ├── train.py                # Entrenamiento del modelo SVD
│   ├── predict.py              # Lógica de inferencia
│   └── artifacts/              # Modelo entrenado (.joblib)
│
├── orchestration/
│   └── logs/                   # Logs de cada paso del pipeline
│
└── app/
    ├── app.py                  # Aplicación Flask
    └── templates/
        └── index.html          # Interfaz de usuario
```

---

## Arquitectura del pipeline

```
CSVs (MovieLens)
      ↓
  movies.py          → MongoDB (ratings, movies, tags, links)
      ↓
  extract.py         → Lee colecciones crudas
      ↓
  transform.py       → Limpieza, normalización, feature engineering
      ↓
  load.py            → MongoDB (colección: processed)
      ↓
  train.py           → Modelo SVD entrenado
      ↓
  app.py             → Aplicación web con recomendaciones
```

---
