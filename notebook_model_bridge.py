from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import requests
from scipy.sparse import hstack, load_npz
from sklearn.preprocessing import normalize

BASE_DIR = Path(__file__).resolve().parent #absolute path
RAPIDAPI_HOST = "spotify-extended-audio-features-api.p.rapidapi.com"
CONTENT_AUDIO_WEIGHT = 0.20
CONTENT_GENRE_WEIGHT = 0.80

NUMERIC_COLS = [
    "danceability",
    "energy",
    "valence",
    "tempo",
    "acousticness",
    "instrumentalness",
    "loudness",
    "speechiness",
    "liveness",
    "popularity",
]

API_FEATURE_COLS = [  # from RapidAPI audio-features endpoint, without popularity
    "danceability",
    "energy",
    "valence",
    "tempo",
    "acousticness",
    "instrumentalness",
    "loudness",
    "speechiness",
    "liveness",
]


@dataclass
class ModelData:
    df: pd.DataFrame
    scaler: Any
    tfidf: Any
    content_model: Any
    final_features: Any
    track_id_to_index: dict[str, int]
    svd_track_ids: list[str] # which songs are in SVD
    svd_track_to_index: dict[str, int] # find SVD row by track ID
    svd_vectors: np.ndarray #SVD vectors
    svd_catalog_indices: np.ndarray # which df row belongs to which SVD row
    hybrid_weights: dict[str, float]
    model_data_dir: Path


def default_model_data_dir() -> Path:
    return BASE_DIR / "model-data"

 
def required_file(model_data_dir: Path, filename: str) -> Path: # in model data
    path = model_data_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing model file: {path}")
    return path


def minmax(values: np.ndarray) -> np.ndarray: #normalization for scores in hybrid model
    low = values.min()
    high = values.max()
    if low == high:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) 
    return matrix / np.maximum(norms, 1e-12) #if a row is 0


def top_indices(scores: np.ndarray, top_k: int, excluded: set[int] | None = None) -> np.ndarray:
    scores = scores.copy()
    for index in excluded or set():
        if 0 <= index < len(scores):
            scores[index] = -np.inf

    ranked = np.argsort(-scores)
    return ranked[:top_k]


def track_ids_from_indices(df: pd.DataFrame, indices: np.ndarray) -> list[str]:
    return df.iloc[indices]["track_id"].astype(str).tolist()


def load_svd_model(model_data_dir: Path) -> tuple[list[str], dict[str, int], np.ndarray]:
    path = required_file(model_data_dir, "collaborative_svd_compact.npz")
    with np.load(path, allow_pickle=False) as model_file:
        track_ids = model_file["track_ids"].astype(str).tolist()
        vectors = np.asarray(model_file["item_factors_norm"], dtype=np.float32)

    if len(track_ids) != vectors.shape[0]:
        raise ValueError("Collaborative SVD rows do not match track IDs")

    track_to_index = {track_id: index for index, track_id in enumerate(track_ids)}
    return track_ids, track_to_index, normalize_rows(vectors)


def load_hybrid_weights(model_data_dir: Path) -> dict[str, float]:
    path = model_data_dir / "hybrid_config.json"
    if not path.exists():
        return {"content_weight": 0.55, "collaborative_weight": 0.35, "popularity_weight": 0.10}

    values = json.loads(path.read_text(encoding="utf-8"))
    return {
        "content_weight": float(values.get("content_weight", 0.55)),
        "collaborative_weight": float(values.get("collaborative_weight", 0.35)),
        "popularity_weight": float(values.get("popularity_weight", 0.10)),
    }


@lru_cache(maxsize=4) # load model files once and reuse them for later recommendations
def load_model_data(model_data_dir: str | None = None) -> ModelData: #model dir should be absolute path
    if model_data_dir:
        path = Path(model_data_dir)
        model_data_dir = path if path.is_absolute() else BASE_DIR / path
    else:
        model_data_dir = default_model_data_dir()

    df = pd.read_csv(required_file(model_data_dir, "spotify_tracks_final.csv"))

    scaler = joblib.load(required_file(model_data_dir, "scaler.pkl"))
    tfidf = joblib.load(required_file(model_data_dir, "tfidf.pkl"))
    content_model = joblib.load(required_file(model_data_dir, "content_knn_model.pkl"))
    final_features = load_npz(required_file(model_data_dir, "final_features_normalized.npz")).tocsr()

    if final_features.shape[0] != len(df):
        raise ValueError(
            "Model feature rows do not match the track catalog rows: ")

    svd_track_ids, svd_track_to_index, svd_vectors = load_svd_model(model_data_dir)
    track_id_to_index = {track_id: index for index, track_id in enumerate(df["track_id"])}
    svd_catalog_indices = np.asarray([track_id_to_index.get(track_id, -1) for track_id in svd_track_ids], dtype=int)

    return ModelData(
        df=df,
        scaler=scaler,
        tfidf=tfidf,
        content_model=content_model,
        final_features=final_features,
        track_id_to_index=track_id_to_index,
        svd_track_ids=svd_track_ids,
        svd_track_to_index=svd_track_to_index,
        svd_vectors=svd_vectors,
        svd_catalog_indices=svd_catalog_indices,
        hybrid_weights=load_hybrid_weights(model_data_dir),
        model_data_dir=model_data_dir,
    )


def rapidapi_headers(key: str) -> dict[str, str]:
    return {
        "x-rapidapi-key": key,
        "x-rapidapi-host": RAPIDAPI_HOST,
        "Content-Type": "application/json",
    }

def local_seed_features(data: ModelData, track_id: str) -> tuple[dict[str, Any], list[str]] | None:
    index = data.track_id_to_index.get(str(track_id))
    if index is None:
        return None

    row = data.df.iloc[index]
    features = {column: float(row[column]) for column in NUMERIC_COLS}
    genres = [str(row["track_genre"])]
    return features, genres


def fetch_audio_features(track_id: str, rapidapi_key: str | None = None) -> dict[str, Any]:
    key = rapidapi_key or os.getenv("RAPIDAPI_KEY")
    if not key:
        raise ValueError("RAPIDAPI_KEY is required when the seed track is not in the local catalog")

    url = f"https://{RAPIDAPI_HOST}/v1/audio-features/{track_id}"
    response = requests.get(url, headers=rapidapi_headers(key), timeout=20)
    if not response.ok:
        raise RuntimeError(f"RapidAPI audio-features failed with {response.status_code}: {response.text}")

    data = response.json()

    return {
        "id": track_id,
        "danceability": float(data["danceability"]),
        "energy": float(data["energy"]),
        "valence": float(data["valence"]),
        "tempo": float(data["tempo"]),
        "acousticness": float(data["acousticness"]),
        "instrumentalness": float(data["instrumentalness"]),
        "loudness": float(data["loudness"]),
        "speechiness": float(data["speechiness"]),
        "liveness": float(data["liveness"]),
    }

def fetch_optional_track_metadata(track_id: str, rapidapi_key: str | None = None) -> tuple[float | None, list[str]]:
    key = rapidapi_key or os.getenv("RAPIDAPI_KEY")
    if not key:
        return None, []

    market = os.getenv("RAPIDAPI_RECOMMENDATIONS_MARKET", "US")
    track_url = f"https://{RAPIDAPI_HOST}/v1/tracks/{track_id}"
    track_response = requests.get(track_url, headers=rapidapi_headers(key), params={"market": market}, timeout=20)
    if not track_response.ok:
        return None, []

    track_data = track_response.json()
    popularity = track_data.get("popularity")
    artist_genres = []

    for artist in (track_data.get("artists") or [])[:3]:
        artist_id = artist.get("id") 

        if artist_id:
            artist_url = f"https://{RAPIDAPI_HOST}/v1/artists/{artist_id}"
            artist_response = requests.get(artist_url, headers=rapidapi_headers(key), timeout=20)
            if artist_response.ok:
                artist_genres.extend(str(genre) for genre in artist_response.json().get("genres", []))

    return (float(popularity) if popularity is not None else None), artist_genres


def fetch_seed_features(track_id: str, rapidapi_key: str | None) -> tuple[dict[str, Any], list[str]]:
    features = fetch_audio_features(track_id, rapidapi_key)
    popularity, artist_genres = fetch_optional_track_metadata(track_id, rapidapi_key)
    if popularity is not None:
        features["popularity"] = popularity
    return features, artist_genres


def features_to_vector(features: dict[str, Any], artist_genres: list[str], data: ModelData) -> Any:
    missing = [column for column in API_FEATURE_COLS if column not in features]
    if missing:
        raise ValueError(f"Missing required audio feature(s): {missing}")

    row = {}
    for column in NUMERIC_COLS:
        if column == "popularity":
            row[column] = float(features.get("popularity", data.df["popularity"].median())) #take median in case api fails
        else:
            row[column] = float(features[column])

    scaled_numerical = data.scaler.transform(pd.DataFrame([row], columns=NUMERIC_COLS)) * CONTENT_AUDIO_WEIGHT
    genre_text = " ".join(artist_genres).strip()
    genre_vector = data.tfidf.transform([genre_text]) * CONTENT_GENRE_WEIGHT
    content_vector = hstack([scaled_numerical, genre_vector]).tocsr()
    return normalize(content_vector, norm="l2", axis=1, copy=True)


def similarity_to_query(feature_matrix: Any, query_vector: Any) -> np.ndarray:
    scores = feature_matrix @ query_vector.T
    return scores.toarray().ravel()

def track_is_in_svd(data: ModelData, track_id: str) -> bool:
    return str(track_id) in data.svd_track_to_index


def find_similar_dataset_track(data: ModelData, input_vector: Any) -> int:
    scores = similarity_to_query(data.final_features, input_vector)
    ranked_indices = np.argsort(-scores)

    for index in ranked_indices:
        track_id = str(data.df.iloc[int(index)]["track_id"])
        if track_is_in_svd(data, track_id):
            return int(index)

    raise ValueError("Could not find a similar track that exists in the SVD model")


def content_based_ids(data: ModelData, input_vector: Any, input_track_id: str, top_k: int) -> list[str]:
    neighbor_count = top_k + 5  #in case of geting used songs
    _, indices = data.content_model.kneighbors(input_vector, n_neighbors=neighbor_count)

    output = []
    for index in indices.ravel():
        track_id = str(data.df.iloc[int(index)]["track_id"])
        if track_id == input_track_id:
            continue
        output.append(track_id)

        if len(output) == top_k:
            break
    return output


def svd_seed_index(data: ModelData, seed_index: int) -> int:
    track_id = str(data.df.iloc[seed_index]["track_id"])

    if track_id not in data.svd_track_to_index:
        raise KeyError(f"Track is not present in the SVD model: {track_id}")

    return data.svd_track_to_index[track_id]

def svd_scores(data: ModelData, seed_index: int) -> tuple[np.ndarray, str]:
    inner_index = svd_seed_index(data, seed_index)
    scores = data.svd_vectors @ data.svd_vectors[inner_index]
    scores[inner_index] = -np.inf
    return scores, "exact SVD seed track"


def collaborative_ids(data: ModelData, seed_index: int, top_k: int) -> tuple[list[str], str]:
    scores, seed_note = svd_scores(data, seed_index)
    indices = top_indices(scores, top_k)
    ids = [data.svd_track_ids[index] for index in indices]
    return ids, "Collaborative SVD generated recommendations from the selected seed track."


def hybrid_ids(data: ModelData, query_vector: Any, seed_index: int, input_track_id: str, top_k: int) -> tuple[list[str], str]:
    content_scores = minmax(similarity_to_query(data.final_features, query_vector))
    popularity_scores = minmax(data.df["popularity"].to_numpy(dtype=float))
    raw_svd_scores, seed_note = svd_scores(data, seed_index)

    svd_df_ordered_scores = np.zeros(len(data.df), dtype=float)
    finite = np.isfinite(raw_svd_scores) # seed track not finite
    svd_df_ordered_scores[data.svd_catalog_indices[finite]] = minmax(raw_svd_scores[finite])

    content_weight = data.hybrid_weights["content_weight"]
    collaborative_weight = data.hybrid_weights["collaborative_weight"]
    popularity_weight = data.hybrid_weights["popularity_weight"]
    final_scores = (
        content_weight * content_scores
        + collaborative_weight * svd_df_ordered_scores
        + popularity_weight * popularity_scores
    )

    excluded = {seed_index}
    input_index = data.track_id_to_index.get((input_track_id))
    if input_index is not None:
        excluded.add(input_index)

    ids = track_ids_from_indices(data.df, top_indices(final_scores, top_k, excluded=excluded))
    note = (
        f"Hybrid combined content ({content_weight:.0%}), collaborative SVD ({collaborative_weight:.0%}), "
        f"and popularity ({popularity_weight:.0%}); SVD seed: {seed_note}."
    )
    return ids, note


def recommend(
    track_id: str,
    model_data_dir: Path | str | None = None,
    top_k: int = 10,
    rapidapi_key: str | None = None,
) -> dict[str, Any]:
    data = load_model_data(str(model_data_dir) if model_data_dir else None)
    seed_track_id = str(track_id)
    local_seed = local_seed_features(data, seed_track_id)

    if local_seed is not None:
        seed_features, artist_genres = local_seed
        feature_source = "local catalogue"
    else:
        seed_features, artist_genres = fetch_seed_features(seed_track_id, rapidapi_key)
        feature_source = "RapidAPI"
        
    seed_vector = features_to_vector(seed_features, artist_genres, data)
    collaborative_seed_index = data.track_id_to_index.get(seed_track_id)

    svd_seed_source = "selected seed track"
    if collaborative_seed_index is None:
        collaborative_seed_index = find_similar_dataset_track(data, seed_vector)
        svd_seed_source = "similar local catalogue track that exists in SVD"
    elif not track_is_in_svd(data, seed_track_id):
        collaborative_seed_index = find_similar_dataset_track(data, seed_vector)
        svd_seed_source = "similar local catalogue track that exists in SVD"


    model_a_ids = content_based_ids(data, seed_vector, seed_track_id, top_k)
    model_b_ids, model_b_note = collaborative_ids(data, collaborative_seed_index, top_k)
    model_c_ids, model_c_note = hybrid_ids(data, seed_vector, collaborative_seed_index, seed_track_id, top_k)

    seed_row = data.df.iloc[collaborative_seed_index]
    local_seed = {
        "track_id": str(seed_row["track_id"]),
        "track_name": str(seed_row["track_name"]),
        "artists": str(seed_row["artists"]),
    }

    return {
        "modelA": model_a_ids,
        "modelB": model_b_ids,
        "modelC": model_c_ids,
        "meta": {
            "source": "notebook_model_bridge",
            "modelDataDir": str(data.model_data_dir),
            "endpointStatus": "ok",
            "seedMatch": f"input track features were loaded from {feature_source}",
            "seed": local_seed,
            "notes": {
                "modelA": f"Content-based model used input track features from {feature_source}.",
                "modelB": f"{model_b_note} SVD seed: {svd_seed_source}.",
                "modelC": f"{model_c_note} SVD seed: {svd_seed_source}.",
            },
        },
    }
