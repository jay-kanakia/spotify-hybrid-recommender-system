import streamlit as st
import pandas as pd
from scipy.sparse import load_npz
from pathlib import Path
from numpy import load as np_load

from src.data.content_filtering_data_transformation import content_recommendation
from src.data.collaborative_filtering_data_transformation import collaborative_recommendation
from src.data.Hybrid_recommendation import HybridRecommenderSystem


# ---------------- SAFE LOADERS ----------------

@st.cache_resource
def load_npz_data(path):
    return load_npz(path)

@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

@st.cache_resource
def load_npy(path):
    return np_load(path, allow_pickle=True)


# ---------------- MAIN ----------------

def main():

    root_path = Path(__file__).parent

    # ---------------- LOAD DATA ONCE ----------------
    transformed_data = load_npz_data(root_path/"data/processed/content_filtering_transformed_data.npz")
    transformed_hybrid_data = load_npz_data(root_path/"data/processed/hybrid_transformed_filtered_data.npz")
    interaction_mat = load_npz_data(root_path/"data/processed/interaction_matrix.npz")

    df_songs = load_csv(root_path/"data/cleaned/df_songs_cleaned.csv")
    df_filtered = load_csv(root_path/"data/filtered/Colab_filtered_data.csv")

    track_ids = load_npy(root_path/"data/track_ids.npy")


    # ---------------- SIDEBAR ----------------
    st.sidebar.title("📂 Navigation")

    section = st.sidebar.radio(
        "Go to:",
        ["🎧 Recommender", "🧠 How It Works", "📊 Dataset & Scale", "☁️ Architecture", "👨‍💻 About"]
    )


    # ---------------- RECOMMENDER ----------------
    if section == "🎧 Recommender":

        st.title("🎧 Spotify Hybrid Recommender")

        st.write("### Select a song 🎵")

        # ---------------- ORIGINAL SEARCH LOGIC ----------------
        song_list = df_songs['name'].to_list()
        artist_list = df_songs['artist'].to_list()

        song_options = [
            {"name": i, "artist": j}
            for i, j in zip(song_list, artist_list)
        ]

        label_to_song = {
            f"{s['name'].title()} by {s['artist'].title()}": s
            for s in song_options
        }

        selected = st.selectbox("Select a Song", list(label_to_song.keys()))

        song_name_org = label_to_song[selected]["name"]
        artist_name_org = label_to_song[selected]["artist"]
        song_name = song_name_org.lower()

        st.write("You selected:", selected)

        k = st.selectbox("How many recommendations?", [5, 10, 15, 20], index=1)


        # ---------------- FILTER CHECK ----------------
        exists = (df_filtered['name'] == song_name).any()

        if exists:
            filtering_type = st.selectbox(
                "Select filtering type:",
                ["Content-Based Filtering", "Collaborative Filtering", "Hybrid Recommender System"],
                index=2
            )

            diversity = st.slider("Diversity", 1, 10, 5)
            weight_content = 1 - (diversity / 10)

        else:
            filtering_type = st.selectbox(
                "Select filtering type:",
                ["Content-Based Filtering"],
                index=0
            )


        # ---------------- CONTENT ----------------
        if filtering_type == "Content-Based Filtering":
            if st.button("Get Recommendation"):

                recs = content_recommendation(song_name, df_songs, transformed_data, k)

                audio_url = df_songs[
                    df_songs['name'] == song_name
                ]['spotify_preview_url'].values[0]

                display(recs, song_name_org, artist_name_org, audio_url)


        # ---------------- COLLAB ----------------
        elif filtering_type == "Collaborative Filtering":
            if st.button("Get Recommendation"):

                recs = collaborative_recommendation(
                    song_name,
                    track_ids,
                    df_filtered,
                    interaction_mat,
                    k
                )

                audio_url = df_filtered[
                    df_filtered['name'] == song_name
                ]['spotify_preview_url'].values[0]

                display(recs, song_name_org, artist_name_org, audio_url)


        # ---------------- HYBRID ----------------
        elif filtering_type == "Hybrid Recommender System":
            if st.button("Get Recommendation"):

                model = HybridRecommenderSystem(
                    number_of_recommendation=k,
                    weight_content_based=weight_content
                )

                recs = model.give_recommmendations(
                    song_name,
                    df_filtered,
                    track_ids,
                    transformed_hybrid_data,
                    interaction_mat
                )

                audio_url = df_filtered[
                    df_filtered['name'] == song_name
                ]['spotify_preview_url'].values[0]

                display(recs, song_name_org, artist_name_org, audio_url)


# ---------------- DISPLAY FUNCTION (FIXED CORE ISSUE) ----------------

def display(recommendations, song_name=None, artist_name=None, audio_url=None):

    # 🎧 CURRENTLY PLAYING
    if song_name and artist_name:

        st.markdown("## 🎧 Currently Playing")
        st.markdown(f"### 🎵 {song_name.title()} by {artist_name.title()}")

        if audio_url:
            st.audio(audio_url)

        st.markdown("---")

    # 🔥 REMOVE CURRENT SONG FROM RECOMMENDATIONS (CRITICAL FIX)
    if song_name:
        recommendations = recommendations[
            recommendations['name'].str.lower() != song_name.lower()
        ]

    # 🔥 NEXT UP (TOP 10 ONLY, CLEAN)
    st.markdown("## 🔥 Next Up")

    recommendations = recommendations.head(10).reset_index(drop=True)

    for i, row in recommendations.iterrows():

        name = row['name'].title()
        artist = row['artist'].title()

        st.markdown(f"### {i+1}. {name} by {artist}")

        if "spotify_preview_url" in row:
            st.audio(row["spotify_preview_url"])

        st.markdown("---")


# ---------------- RUN ----------------

if __name__ == "__main__":
    main()