import streamlit as st
import pandas as pd
import logging

from scipy.sparse import load_npz
from src.data.content_filtering_data_transformation import content_recommendation
from src.data.collaborative_filtering_data_transformation import collaborative_recommendation
from src.data.Hybrid_recommendation import HybridRecommenderSystem
from pathlib import Path
from numpy import load

# create a logger
logger = logging.getLogger('streamlit_app')
logger.setLevel(logging.INFO)

# create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# create formatter
formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to handler
console_handler.setFormatter(formatter)

# add handler to logger
logger.addHandler(console_handler)

def load_npz_data(trans_data_path:Path):

    try:
        # load transformed data
        transformed_data = load_npz(trans_data_path)
    except FileNotFoundError:
        logger.error("Transformed data file not found at location")
    return transformed_data

def load_data(clean_data_path:Path)->pd.DataFrame:
    
    try:
        # load the data
        df = pd.read_csv(clean_data_path)
    except FileNotFoundError:
        logger.error("Data file not found at location")
    return df

def load_np(track_ids_path:Path):
    """
    load track ids
    """
    track_ids = load(track_ids_path,allow_pickle=True)

    return track_ids

def main():
    # root path
    root_path = Path(__file__).parent

    # transformed data path
    transformed_data_path = root_path/"data"/"processed"/"hybrid_transformed_filtered_data.npz"

    # load transformed data
    transformed_data = load_npz_data(transformed_data_path)
    logger.info("Transformed data loaded successfully")

    # cleaned data path
    cleaned_data_path = root_path/"data"/"cleaned"/"df_songs_cleaned.csv"

    # load clean data
    df_songs = load_data(cleaned_data_path)
    logger.info("Cleaned data loaded successfully")

    # load original data
    org_data_path = root_path/"data"/"raw"/"Music Info.csv"
    org_data = load_data(org_data_path)

    # filtered data path
    filtered_path = root_path/"data"/"filtered"/"Colab_filtered_data.csv"
    df_filtered = load_data(filtered_path)

    # interaction matrix path
    interaction_matrix_path = root_path/"data"/"processed"/"interaction_matrix.npz"

    # load interaction matrix
    interaction_mat = load_npz_data(interaction_matrix_path)

    # track_ids path
    track_id_path = root_path/"data"/"track_ids.npy"

    # load track_ids
    track_ids = load_np(track_id_path)

# ===========================================================================================

    # title
    st.title("Welcome to the Spotify Song Recommender!")

    # sub header
    st.write('### Enter the name of a song and the recommender will suggest similar songs 🎵🎧')

    # text input
    song_list = org_data['name'].to_list()
    artist_list = org_data['artist'].to_list()

    song_options = [
    {"name": i, "artist": j}
    for i, j in zip(song_list, artist_list)
    ]

    label_to_song = {
        f"{s['name']} by {s['artist']}": s
        for s in song_options
    }

    selected = st.selectbox("Select a Song", list(label_to_song.keys()))
    song_name_org = label_to_song[selected]["name"]
    artist_name = label_to_song[selected]["artist"]

    st.write('You entered :',selected)

    # lowercase the input
    song_name = song_name_org.lower()

    # select no. of recommendation
    k = st.selectbox('How many recommendation you need?',[5,10,15,20],index=1)

    # type of filtering
    filtering_type = st.selectbox('Select the type of filtering:', ['Content-Based Filtering', 'Collaborative Filtering','Hybrid Recommender System'],index=2)

    # Button
    if filtering_type == 'Content-Based Filtering':
        if st.button('Get Recommendation'):
            if (df_songs['name'] == song_name).any():
                st.write('Recommendation for', f"**{selected}**")
                recommendations = content_recommendation(song_name,df_songs,transformed_data,k)

                # Display recommendations
                for ind, recommendation in recommendations.iterrows():
                    rec_song_name = recommendation['name'].title()
                    rec_artist_name = recommendation['artist'].title()

                    if ind == 0:
                        st.markdown('## Currently Playing')
                        st.markdown(f'### **{rec_song_name}** by **{rec_artist_name}**')
                        st.audio(recommendation['spotify_preview_url'])
                        st.write('---')

                    elif ind == 1:
                        st.markdown('### Next Up 🎵')
                        st.markdown(f"### {ind}. **{rec_song_name}** by **{rec_artist_name}**")
                        st.audio(recommendation['spotify_preview_url'])
                        st.write('---')

                    else:
                        st.markdown(f"### {ind}. **{rec_song_name}** by **{rec_artist_name}**")
                        st.audio(recommendation['spotify_preview_url'])
                        st.write('---')

            else:
                st.write(f"Sorry, we couldn't find {song_name} in our database. Please try another song.")
                
    elif filtering_type == 'Collaborative Filtering':
        if st.button('Get Recommendation'):
            if (df_filtered['name'] == song_name).any():
                st.write('Recommendation for', f"**{selected}**")
                recommendations = collaborative_recommendation(song_name,track_ids,df_filtered,interaction_mat,k)

                # Display recommendations
                for ind, recommendation in recommendations.iterrows():
                    rec_song_name = recommendation['name'].title()
                    rec_artist_name = recommendation['artist'].title()

                    if ind == 0:
                        st.markdown('## Currently Playing')
                        st.markdown(f'### **{rec_song_name}** by **{rec_artist_name}**')
                        st.audio(recommendation['spotify_preview_url'])
                        st.write('---')

                    elif ind == 1:
                        st.markdown('### Next Up 🎵')
                        st.markdown(f"### {ind}. **{rec_song_name}** by **{rec_artist_name}**")
                        st.audio(recommendation['spotify_preview_url'])
                        st.write('---')

                    else:
                        st.markdown(f"### {ind}. **{rec_song_name}** by **{rec_artist_name}**")
                        st.audio(recommendation['spotify_preview_url'])
                        st.write('---')


            else:
                st.write(f"Sorry, we couldn't find {song_name} in our database. Please try another song.")
                
    elif filtering_type == "Hybrid Recommender System":
        if st.button('Get Recommmendation'):
            if (df_filtered['name'] == song_name).any():
                st.write('Recommendation for', f"**{selected}**")
                recommender = HybridRecommenderSystem(number_of_recommendation=k,weight_content_based=0.3)
                recommendations = recommender.give_recommmendations(song_name,df_filtered,track_ids,transformed_data,interaction_mat)
                
                # Display recommendations
                for ind, recommendation in recommendations.iterrows():
                    rec_song_name = recommendation['name'].title()
                    rec_artist_name = recommendation['artist'].title()

                    if ind == 0:
                        st.markdown('## Currently Playing')
                        st.markdown(f'### **{rec_song_name}** by **{rec_artist_name}**')
                        st.audio(recommendation['spotify_preview_url'])
                        st.write('---')

                    elif ind == 1:
                        st.markdown('### Next Up 🎵')
                        st.markdown(f"### {ind}. **{rec_song_name}** by **{rec_artist_name}**")
                        st.audio(recommendation['spotify_preview_url'])
                        st.write('---')

                    else:
                        st.markdown(f"### {ind}. **{rec_song_name}** by **{rec_artist_name}**")
                        st.audio(recommendation['spotify_preview_url'])
                        st.write('---')


            else:
                st.write(f"Sorry, we couldn't find {song_name} in our database. Please try another song.")




if __name__ == "__main__":
    main()
    




