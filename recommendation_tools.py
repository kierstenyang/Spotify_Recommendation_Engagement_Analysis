"""
Step 1: Recommendation tool functions.

These are the "hands" the AI agent will later call. They do all the actual
data retrieval from the cleaned Spotify dataset -- the LLM will never see
raw data it wasn't given through these functions, which is how we prevent
it from inventing songs that don't exist.

No LLM/API key needed for this file -- test it standalone first.
"""

import pandas as pd

df = pd.read_csv("dataset_cleaned.csv")


def search_tracks(danceability=None, energy=None, valence=None,
                   min_popularity=None, favorite_artists=None, top_n=20):
    """
    Find tracks matching target audio characteristics.

    Parameters are "target" values (0-1 scale for danceability/energy/valence).
    Tracks are ranked by closeness to the target profile (Euclidean distance),
    optionally filtered by minimum popularity or favorite artists.
    """
    candidates = df.copy()

    if favorite_artists:
        # Boost/filter toward tracks by favorite artists if any are present
        mask = candidates["artists"].str.contains(
            "|".join(favorite_artists), case=False, na=False
        )
        if mask.sum() >= top_n:
            candidates = candidates[mask]
        # if too few matches, fall back to the full dataset instead of erroring out

    if min_popularity is not None:
        candidates = candidates[candidates["popularity"] >= min_popularity]

    # Build a target profile using provided values, defaulting missing ones
    # to the dataset's overall mean so they don't distort the distance calc
    target = {
        "danceability": danceability if danceability is not None else df["danceability"].mean(),
        "energy": energy if energy is not None else df["energy"].mean(),
        "valence": valence if valence is not None else df["valence"].mean(),
    }

    candidates = candidates.copy()
    candidates["distance"] = (
        (candidates["danceability"] - target["danceability"]) ** 2 +
        (candidates["energy"] - target["energy"]) ** 2 +
        (candidates["valence"] - target["valence"]) ** 2
    ) ** 0.5

    results = candidates.sort_values("distance").head(top_n)

    return results[[
        "track_name", "artists", "danceability", "energy",
        "valence", "popularity", "track_genre"
    ]].to_dict(orient="records")


def get_track_info(track_name):
    """Return full details for a specific track by (partial, case-insensitive) name."""
    matches = df[df["track_name"].str.contains(track_name, case=False, na=False)]
    if matches.empty:
        return {"error": f"No track found matching '{track_name}'"}
    return matches.head(5)[[
        "track_name", "artists", "album_name", "popularity",
        "danceability", "energy", "valence", "tempo", "track_genre"
    ]].to_dict(orient="records")


def get_artist_stats(artist_name):
    """Return how frequently an artist appears in the dataset and their average stats."""
    matches = df[df["artists"].str.contains(artist_name, case=False, na=False)]
    if matches.empty:
        return {"error": f"No tracks found for artist '{artist_name}'"}
    return {
        "artist_query": artist_name,
        "track_count_in_dataset": len(matches),
        "avg_popularity": round(matches["popularity"].mean(), 1),
        "avg_danceability": round(matches["danceability"].mean(), 3),
        "avg_energy": round(matches["energy"].mean(), 3),
        "avg_valence": round(matches["valence"].mean(), 3),
        "top_tracks": matches.nlargest(3, "popularity")["track_name"].tolist(),
    }


# ---------- Quick standalone test (no API key needed) ----------
if __name__ == "__main__":
    print("=== Test: search_tracks (high energy, high danceability) ===")
    results = search_tracks(danceability=0.8, energy=0.8, top_n=5)
    for r in results:
        print(f"  {r['track_name']} — {r['artists']} (pop: {r['popularity']})")

    print("\n=== Test: get_track_info('Comedy') ===")
    print(get_track_info("Comedy"))

    print("\n=== Test: get_artist_stats('Bieber') ===")
    print(get_artist_stats("Bieber"))
