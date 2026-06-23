# Python Music Eval Site

This is a simpler Python/Flask version of the music recommendation evaluation site.
Its pages are styled to match the React version as closely as possible.

It keeps the main idea:

- search for a seed song with Spotify
- generate recommendations from Python model logic
- rate recommendations
- view statistics in `/admin`

## How Recommendations Work

The app first searches Spotify so the user can choose a seed track. For the
local Python models, `notebook_model_bridge.py` loads the exported files from
`model-data/`.

If the selected seed track already exists in the local catalogue, the app uses
the local catalogue audio features. If it is not in the local catalogue, the app
uses RapidAPI audio features as a fallback.

The collaborative SVD model contains only tracks that appeared in the synthetic
interaction data. If a catalogue track is not present in SVD, the app uses a
similar local catalogue track that does exist in SVD for the collaborative part.
The rating page notes whether features came from the local catalogue or RapidAPI
and whether SVD used the selected seed track or a similar SVD-covered track.

## Run Locally

1. Copy `.env.local.example` to `.env.local`.
2. Fill in Spotify keys.
3. Install `requirements.txt` into the Python environment you will use, if the
   packages are not already installed.
4. Optional: add `RAPIDAPI_KEY` for out-of-catalogue seed features and the API
   recommendation baseline.
5. Optional: add `ADMIN_PASSWORD` to unlock `/admin`.
6. Double-click `RUN_LOCAL.bat`.
7. Open `http://localhost:5001/`.

## Deploy Later

This folder is deployable on services like Railway or Render. Railway can use
`railway.json` and `Procfile` to start the app with Gunicorn on the correct port.

Set these environment variables in the deployment dashboard:

- `SPOTIFY_CLIENT_ID`
- `SPOTIFY_CLIENT_SECRET`
- `RAPIDAPI_KEY`
- `ADMIN_PASSWORD`
- `DATABASE_URL`
- `FLASK_SECRET_KEY`
- `MODEL_DATA_DIR`

`ADMIN_PASSWORD` is required to access `/admin`; without it, the admin page is
locked. `DATABASE_URL` should be a Postgres URL from the hosting service. If it
is not set locally, the app uses `music_eval.sqlite3`.

## Main Files

- `app.py` - the Flask site, API calls, database, pages, ratings, admin stats.
- `notebook_model_bridge.py` - loads the exported notebook model data and generates the connected model recommendations.
- `music-recommendation-system-training.ipynb` - offline notebook used to prepare data, train models, evaluate them, and export artifacts.
- `templates/` - HTML pages.
- `static/styles.css` - page styling.
- `model-data/` - exported model files and the local track catalog used by the Python models.
- `requirements.txt` - Python libraries to install.
- `Procfile` - tells hosting services how to start the site.
- `railway.json` - Railway deploy settings and healthcheck.
- `runtime.txt` - Python version for deployment.
