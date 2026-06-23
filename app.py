from __future__ import annotations

import csv
import io
import json
import os
import random
import re
import time
import uuid
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from flask import (
    Flask,
    Response,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session as browser_session,
    url_for,
)
from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    insert,
    select,
    update,
)

from notebook_model_bridge import recommend as notebook_recommend


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env.local", encoding="utf-8-sig")

MODEL_KEYS = ["modelA", "modelB", "modelC", "modelD", "spotify"]
LOCAL_MODEL_KEYS = ["modelA", "modelB", "modelC"]
MODEL_LABELS = {
    "modelA": "Content-Based Model",
    "modelB": "Collaborative SVD Model",
    "modelC": "Hybrid Model (SVD + Audio + Popularity)",
    "modelD": "Popularity Baseline",
    "spotify": "API Recommendation Baseline",
}
RAPIDAPI_RECOMMENDATIONS_HOST = "spotify-extended-audio-features-api.p.rapidapi.com"
market = os.getenv("SPOTIFY_MARKET", "US")

def show_model_names() -> bool:
    return os.getenv("SHOW_MODEL_NAMES", "1") == "1"

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "local-flask-secret-key")


def database_url() -> str:
    raw = os.getenv("DATABASE_URL")
    if not raw:
        return f"sqlite:///{(BASE_DIR / 'music_eval.sqlite3').as_posix()}"  #local: SQLite file
    if raw.startswith("postgres://"):
        return raw.replace("postgres://", "postgresql+psycopg://", 1)  #deployment: convert Railway/Postgres URL for SQLAlchemy
    if raw.startswith("postgresql://"):
        return raw.replace("postgresql://", "postgresql+psycopg://", 1)  #deployment: convert PostgreSQL URL for SQLAlchemy
    return raw  


db_url = database_url()
engine = create_engine(
    db_url,
    future=True,
    pool_pre_ping=True,
    connect_args={"check_same_thread": False} if db_url.startswith("sqlite") else {},
)
metadata = MetaData()

sessions_table = Table(
    "sessions",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("session_id", String, unique=True, nullable=False),
    Column("seed_track_id", String, nullable=False),
    Column("seed_track_name", String, nullable=False),
    Column("seed_artist_name", String, nullable=False),
    Column("seed_image_url", Text, nullable=False, default=""),
    Column("model_a_track_id", String, nullable=False),
    Column("model_a_track_name", String, nullable=False),
    Column("model_a_track_artist", String, nullable=False),
    Column("model_a_image_url", Text, nullable=False, default=""),
    Column("model_a_preview_url", Text),
    Column("model_b_track_id", String, nullable=False),
    Column("model_b_track_name", String, nullable=False),
    Column("model_b_track_artist", String, nullable=False),
    Column("model_b_image_url", Text, nullable=False, default=""),
    Column("model_b_preview_url", Text),
    Column("model_c_track_id", String, nullable=False),
    Column("model_c_track_name", String, nullable=False),
    Column("model_c_track_artist", String, nullable=False),
    Column("model_c_image_url", Text, nullable=False, default=""),
    Column("model_c_preview_url", Text),
    Column("model_d_track_id", String, nullable=False),
    Column("model_d_track_name", String, nullable=False),
    Column("model_d_track_artist", String, nullable=False),
    Column("model_d_image_url", Text, nullable=False, default=""),
    Column("model_d_preview_url", Text),
    Column("spotify_track_id", String, nullable=False),
    Column("spotify_track_name", String, nullable=False),
    Column("spotify_track_artist", String, nullable=False),
    Column("spotify_image_url", Text, nullable=False, default=""),
    Column("spotify_preview_url", Text),
    Column("recommendation_notes", Text),
    Column("rating_model_a", Integer),
    Column("rating_model_b", Integer),
    Column("rating_model_c", Integer),
    Column("rating_model_d", Integer),
    Column("rating_spotify", Integer),
    Column("preferred_model", String),
    Column("submitted_at", DateTime(timezone=True)),
    Column("created_at", DateTime(timezone=True), nullable=False),
)
metadata.create_all(engine)


spotify_token = {"value": None, "expires_at": 0.0}

def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def request_with_retries(method: str, url: str, max_attempts: int = 3, **kwargs: Any) -> requests.Response:
    for attempt in range(max_attempts):
        try:
            response = requests.request(method, url, timeout=(8, 45), **kwargs)
            if response.status_code in {429, 500, 502, 503, 504} and attempt < max_attempts - 1:
                time.sleep(2)
                continue
            return response
        
        except (requests.Timeout, requests.ConnectionError):
            if attempt < max_attempts - 1:
                time.sleep(2)
                continue
            raise


def spotify_token_value() -> str:
    if spotify_token["value"] and spotify_token["expires_at"] > time.time() + 30: #if the token is valid for at least 30 seconds
        return str(spotify_token["value"])

    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError("SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET are required")

    response = request_with_retries( "POST","https://accounts.spotify.com/api/token",data={"grant_type": "client_credentials"},auth=(client_id, client_secret))
    response.raise_for_status()
    data = response.json()
    spotify_token["value"] = data["access_token"]
    spotify_token["expires_at"] = time.time() + int(data.get("expires_in", 3600))
    return str(spotify_token["value"])


def spotify_get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    response = request_with_retries(
        "GET",
        f"https://api.spotify.com/v1{path}",
        headers={"Authorization": f"Bearer {spotify_token_value()}"},
        params=params,
    )
    response.raise_for_status()
    return response.json()


def map_track(raw: dict[str, Any]) -> dict[str, Any]: #from raw response extracts only the details that are needed
    artists = raw.get("artists") or []
    album = raw.get("album") or {}
    images = album.get("images") or []
    return {
        "id": raw.get("id", ""),
        "name": raw.get("name", ""),
        "artist": ", ".join(a.get("name", "") for a in artists if a.get("name")),
        "artist_ids": [a.get("id") for a in artists if a.get("id")],
        "album": album.get("name", ""),
        "image_url": images[0]["url"] if images else "",
        "preview_url": raw.get("preview_url"),
        "spotify_url": (raw.get("external_urls") or {}).get("spotify", f"https://open.spotify.com/track/{raw.get('id', '')}")}


def search_spotify(query: str) -> list[dict[str, Any]]:
    if not query.strip(): #user typed nothing
        return []
    data = spotify_get("/search",{"q": query.strip(), "type": "track", "limit": 8, "market": market}) 
    return [map_track(item) for item in data.get("tracks", {}).get("items", [])]


def get_track_details(track_id: str) -> dict[str, Any]:
    return (map_track(spotify_get(f"/tracks/{track_id}")))


def get_artist_top_tracks(artist_id: str) -> list[dict[str, Any]]:
    try:
        data = spotify_get(f"/artists/{artist_id}/top-tracks", {"market": market})
        return data.get("tracks", [])
    except Exception as error:
        app.logger.warning("Could not load artist top tracks: %s", error)
        return []


def collect_recommendation_ids(data: dict[str, Any]) -> list[str]: #from rapidAPI
    track_ids = []

    for track in data.get("tracks", []):
        track_id = track.get("id")

        if track_id and track_id not in track_ids: #not empty and is not already in list
            track_ids.append(track_id)

    return track_ids


def notebook_recommendations(seed_track_id: str):
    model_folder = Path(os.getenv("MODEL_DATA_DIR", "model-data"))
    if not model_folder.is_absolute():
        model_folder = BASE_DIR / model_folder
    top_k = int(os.getenv("MODEL_BRIDGE_TOP_K", "10"))
    rapidapi_key = os.getenv("RAPIDAPI_KEY")

    try:
        result = notebook_recommend(seed_track_id, model_folder, top_k, rapidapi_key)
        notes = result.get("meta", {}).get("notes", {}) #notes about how recs were produced (actual or fallback)
        recommendations = {}
        for model_key in LOCAL_MODEL_KEYS:
            tracks = []
            for track_id in result.get(model_key, []):
                try:
                    tracks.append(get_track_details(track_id))
                except Exception as error:
                    app.logger.warning("Could not get details for model track %s: %s", track_id, error)
            recommendations[model_key] = tracks
        return recommendations, notes

    except Exception as error:
        message = str(error)
        if "429" in message or "quota" in message.lower():
            note = "RapidAPI audio-features daily quota or rate limit was reached; showing fallback recommendation."
        elif "403" in message:
            note = "RapidAPI audio-features endpoint refused the request; showing fallback recommendation."
        else:
            note = "Exported model recommendation failed; showing fallback recommendation."
        app.logger.warning("Exported model recommendation failed: %s", error)
        return {}, {model_key: note for model_key in LOCAL_MODEL_KEYS}




def rapidapi_endpoint_recommendation(seed_track_id: str, used_ids: set[str]):
    api_key = os.getenv("RAPIDAPI_KEY")

    if not api_key:
        return None, "API recommendation endpoint is not configured; showing fallback recommendation."

    host = os.getenv("RAPIDAPI_RECOMMENDATIONS_HOST", RAPIDAPI_RECOMMENDATIONS_HOST)
    params = {"seed_tracks": seed_track_id,"limit": "10","market": market}

    try:
        response = request_with_retries(
            "GET",
            f"https://{host}/v1/recommendations",
            params=params,
            headers={
                "Content-Type": "application/json",
                "x-rapidapi-host": host,
                "x-rapidapi-key": api_key,
            })

        if not response.ok:
            return None, f"API recommendation endpoint failed ({response.status_code}); showing fallback recommendation."

        ids = collect_recommendation_ids(response.json())
        if not ids:
            return None, "API recommendation endpoint returned no tracks; showing fallback recommendation."

        for track_id in ids:
            if track_id in used_ids:
                continue

            track = get_track_details(track_id)
            used_ids.add(track["id"])
            return track, "API recommendation baseline returned this track from the selected seed track."

    except Exception as error:
        app.logger.warning("API recommendation endpoint failed: %s", error)
        return None, "API recommendation endpoint errored; showing fallback recommendation."
    

def build_fallback_pool(seed_track_details: dict[str, Any])-> list[dict[str, Any]]: #for fallback recs, get top tracks of seed song's artist(s)
    pool = []
    for artist_id in seed_track_details.get("artist_ids", [])[:3]:
        pool.extend(get_artist_top_tracks(artist_id))
    return pool


def catalog_popularity_track_ids() -> list[str]:
    catalog_path = BASE_DIR / "model-data" / "top_popular_tracks.csv"
    df = pd.read_csv(catalog_path)
    return df["track_id"].astype(str).tolist()


def popularity_baseline_recommendation(used_ids: set[str]):
    track_ids = catalog_popularity_track_ids()
    candidates = [track_id for track_id in track_ids if track_id not in used_ids]

    if not candidates:
        return None, "Popularity baseline could not find an unused popular track; showing fallback recommendation."

    random.shuffle(candidates)
    for track_id in candidates:
        try:
            track = get_track_details(track_id)
            used_ids.add(track["id"])
            return track, "Popularity baseline randomly selected one track from the top popular tracks in the local dataset."

        except Exception as error:
            app.logger.warning("Could not get details for popularity baseline track %s: %s", track_id, error)

    return None, "Popularity baseline could not load a usable popular track; showing fallback recommendation."


def pick_fallback(pool: list[dict[str, Any]], used_ids: set[str], seed_track_details: dict[str, Any]) -> dict[str, Any]:
    candidates = [track for track in pool if track.get("id") and track["id"] not in used_ids]

    if candidates:
        raw_track = random.choice(candidates)
        used_ids.add(raw_track["id"])
        return map_track(raw_track)
    
    return seed_track_details


def generate_recommendations(seed_track_id: str) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    seed = get_track_details(seed_track_id)
    used_ids = {seed_track_id}
    selected = {}
    notes = {}

    model_recs, model_notes = notebook_recommendations(seed_track_id)
    notes.update(model_notes)

    for model_key in LOCAL_MODEL_KEYS:
        for track in model_recs.get(model_key, []):
            if track["id"] not in used_ids:
                selected[model_key] = track
                used_ids.add(track["id"])
                break

    spotify_track, spotify_note = rapidapi_endpoint_recommendation(seed_track_id,used_ids)
    if spotify_track:
        selected["spotify"] = spotify_track
    notes["spotify"] = spotify_note

    popularity_track, popularity_note = popularity_baseline_recommendation(used_ids)
    if popularity_track:
        selected["modelD"] = popularity_track
    notes["modelD"] = popularity_note

    fallback_pool = build_fallback_pool(seed)
    for model_key in MODEL_KEYS:
        if model_key not in selected:
            selected[model_key] = pick_fallback(fallback_pool, used_ids, seed)
            notes.setdefault(model_key, "Model did not return a usable recommendation; showing fallback recommendation.") #sets this note for keys that don't exist (models which failed)

    return {key: selected[key] for key in MODEL_KEYS}, notes


def track_columns(model_key: str) -> dict[str, str]: #returns column names (in database) for specific model
    prefix = {
        "modelA": "model_a",
        "modelB": "model_b",
        "modelC": "model_c",
        "modelD": "model_d",
        "spotify": "spotify",
    }[model_key]
    return {
        "id": f"{prefix}_track_id",
        "name": f"{prefix}_track_name",
        "artist": f"{prefix}_track_artist",
        "image_url": f"{prefix}_image_url",
        "preview_url": f"{prefix}_preview_url",
    }


def save_session(seed: dict[str, Any], recommendations: dict[str, dict[str, Any]], notes: dict[str, str]) -> str:
    session_id = str(uuid.uuid4()) #random unique id
    values = {
        "session_id": session_id,
        "seed_track_id": seed["id"],
        "seed_track_name": seed["name"],
        "seed_artist_name": seed["artist"],
        "seed_image_url": seed["image_url"],
        "recommendation_notes": json.dumps(notes),
        "created_at": now_utc(),
    }

    for model_key, track in recommendations.items():
        columns = track_columns(model_key)
        values[columns["id"]] = track["id"]
        values[columns["name"]] = track["name"]
        values[columns["artist"]] = track["artist"]
        values[columns["image_url"]] = track["image_url"]
        values[columns["preview_url"]] = track.get("preview_url") #preview url may not exist

    with engine.begin() as connection:
        connection.execute(insert(sessions_table).values(**values))

    return session_id


def row_to_dict(row: Any) -> dict[str, Any] | None: #a row of database to dictionary
    return dict(row._mapping) if row else None


def get_session_record(session_id: str) -> dict[str, Any] | None: #get session from a database
    with engine.begin() as connection:
        row = connection.execute(
            select(sessions_table).where(sessions_table.c.session_id == session_id)
        ).first()
    return row_to_dict(row)


def list_session_records() -> list[dict[str, Any]]: #get all saved sessions from a database
    with engine.begin() as connection:
        rows = connection.execute(select(sessions_table).order_by(sessions_table.c.created_at)).all()
    return [dict(row._mapping) for row in rows]


def session_view(record: dict[str, Any]) -> dict[str, Any]: #showing recommendations
    recommendations = {}
    for model_key in MODEL_KEYS:
        columns = track_columns(model_key)
        recommendations[model_key] = {
            "id": record[columns["id"]],
            "name": record[columns["name"]],
            "artist": record[columns["artist"]],
            "image_url": record[columns["image_url"]],
            "preview_url": record[columns["preview_url"]],
            "spotify_url": f"https://open.spotify.com/track/{record[columns['id']]}",
        }

    notes = json.loads(record["recommendation_notes"] or "{}")
    order = MODEL_KEYS.copy()
    random.Random(record["session_id"]).shuffle(order) #use session id as random seed
    return {
        "session_id": record["session_id"],
        "seed_track": {
            "id": record["seed_track_id"],
            "name": record["seed_track_name"],
            "artist": record["seed_artist_name"],
            "image_url": record["seed_image_url"],
        },
        "recommendations": recommendations,
        "notes": notes,
        "order": order,
    }


def update_session_ratings(session_id: str, ratings: dict[str, int], preferred_model: str) -> None:
    with engine.begin() as connection:
        connection.execute(
            update(sessions_table)
            .where(sessions_table.c.session_id == session_id)
            .values(
                rating_model_a=ratings["modelA"],
                rating_model_b=ratings["modelB"],
                rating_model_c=ratings["modelC"],
                rating_model_d=ratings["modelD"],
                rating_spotify=ratings["spotify"],
                preferred_model=preferred_model,
                submitted_at=now_utc(),
            )
        )


def admin_allowed() -> bool:
    password = os.getenv("ADMIN_PASSWORD")
    if not password:
        return False
    return browser_session.get("admin_allowed") is True #may be None

def model_rating_field(model_key: str) -> str:
    return {
        "modelA": "rating_model_a",
        "modelB": "rating_model_b",
        "modelC": "rating_model_c",
        "modelD": "rating_model_d",
        "spotify": "rating_spotify",
    }[model_key]


def admin_stats(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    submitted = [record for record in records if record.get("submitted_at")]
    output = []
    for model_key in MODEL_KEYS:
        rating_field = model_rating_field(model_key)
        ratings = [record[rating_field] for record in submitted if record.get(rating_field) is not None]
        avg = round(sum(ratings) / len(ratings), 2) if ratings else 0
        chosen_count = sum(1 for record in submitted if record.get("preferred_model") == model_key) # model chosen as the best
        choice_share = round((chosen_count / len(submitted)) * 100, 2) if submitted else 0 # percentage of favorite model
        distribution = {str(number): ratings.count(number) for number in range(1, 6)} #stats
        output.append(
            {
                "model": model_key,
                "label": MODEL_LABELS[model_key],
                "avg_rating": avg,
                "total_ratings": len(ratings),
                "chosen_count": chosen_count,
                "choice_share": choice_share,
                "distribution": distribution,
            }
        )
    return output


@app.route("/")
def home() -> str:
    query = request.args.get("q", "").strip()
    tracks: list[dict[str, Any]] = []
    if query:
        try:
            tracks = search_spotify(query)
        except Exception as error:
            flash(f"Spotify search failed: {error}", "error")
    return render_template("home.html", query=query, tracks=tracks)


@app.get("/api/search")
def api_search() -> Response:
    query = request.args.get("q", "").strip()
    try:
        tracks = search_spotify(query) if len(query) > 1 else []
        return jsonify({"tracks": tracks})
    except Exception as error:
        app.logger.warning("Spotify search failed: %s", error)
        return jsonify({"tracks": [], "error": "Spotify search failed"}), 502


@app.post("/start")
def start_session() -> Response:
    seed_track_id = request.form.get("track_id", "").strip()
    if not seed_track_id:
        flash("Choose a track first.", "error")
        return redirect(url_for("home"))

    try:
        seed = get_track_details(seed_track_id)
        recommendations, notes = generate_recommendations(seed_track_id)
        session_id = save_session(seed, recommendations, notes)
        return redirect(url_for("rate", session_id=session_id))
    except Exception as error:
        app.logger.exception("Could not create session")
        flash(f"Could not create recommendations: {error}", "error")
        return redirect(url_for("home"))


@app.route("/rate/<session_id>", methods=["GET", "POST"])
def rate(session_id: str) -> str | Response:
    record = get_session_record(session_id)
    if not record:
        return render_template("not_found.html"), 404

    if request.method == "POST":
        try:
            ratings = {
                model_key: int(request.form[f"rating_{model_key}"])
                for model_key in MODEL_KEYS
            }
            preferred_model = request.form["preferred_model"]
            if preferred_model not in MODEL_KEYS:
                raise ValueError("Invalid favorite model")
            update_session_ratings(session_id, ratings, preferred_model)
            return redirect(url_for("thank_you"))
        except Exception:
            flash("Please rate every recommendation and choose a favorite.", "error")

    return render_template(
        "rate.html",
        data=session_view(record),
        model_labels=MODEL_LABELS,
        show_model_names=show_model_names(),
    )


@app.route("/thank-you")
def thank_you() -> str:
    return render_template("thank_you.html")


@app.route("/admin", methods=["GET", "POST"])
def admin() -> str | Response:
    password = os.getenv("ADMIN_PASSWORD")
    if password and request.method == "POST":
        if request.form.get("password") == password:
            browser_session["admin_allowed"] = True
            return redirect(url_for("admin"))
        flash("That password did not work.", "error")

    if not admin_allowed():
        return render_template("admin_login.html")

    records = list_session_records()
    submitted_count = sum(1 for record in records if record.get("submitted_at"))
    return render_template(
        "admin.html",
        records=records,
        stats=admin_stats(records),
        model_labels=MODEL_LABELS,
        submitted_count=submitted_count,
    )


@app.route("/admin.csv")
def admin_csv() -> Response:
    if not admin_allowed():
        return redirect(url_for("admin"))

    records = list_session_records()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "Session ID",
            "Date",
            "Seed Track",
            "Seed Artist",
            "Preferred Model",
            "Model A Rating",
            "Model B Rating",
            "Model C Rating",
            "Model D Rating",
            "Spotify Rating",
        ]
    )
    for record in records:
        writer.writerow(
            [
                record["session_id"],
                record["created_at"],
                record["seed_track_name"],
                record["seed_artist_name"],
                MODEL_LABELS.get(record.get("preferred_model"), record.get("preferred_model") or ""),
                record.get("rating_model_a") or "",
                record.get("rating_model_b") or "",
                record.get("rating_model_c") or "",
                record.get("rating_model_d") or "",
                record.get("rating_spotify") or "",
            ]
        )

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=music_eval_results.csv"},
    )


@app.route("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=os.getenv("FLASK_DEBUG") == "1")

