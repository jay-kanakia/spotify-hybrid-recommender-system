import streamlit as st
import pandas as pd
import logging

from scipy.sparse import load_npz
from src.data.content_filtering_data_transformation import content_recommendation
from pathlib import Path

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

def load_transformed_data(trans_data_path:Path):

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
        logger.error("Clean data file not found at location")
    return df

def main():
    # root path
    root_path = Path(__file__).parent

    # transformed data path
    transformed_data_path = root_path/"data"/"processed"/"content_filtering_transformed_data.npz"

    # load transformed data
    transformed_data = load_transformed_data(transformed_data_path)
    logger.info("Transformed data loaded successfully")

    # cleaned data path
    cleaned_data_path = root_path/"data"/"cleaned"/"df_songs_cleaned.csv"

    # load clean data
    cleaned_data = load_data(cleaned_data_path)
    logger.info("Cleaned data loaded successfully")

    # load original data
    org_data_path = root_path/"data"/"raw"/"Music Info.csv"
    org_data = load_data(org_data_path)

    # title
    st.title("Welcome to the Spotify Song Recommender!")

    # sub header
    st.write('### Enter the name of a song and the recommender will suggest similar songs 🎵🎧')

    # text input
    song_list = org_data['name'].to_list()
    song_name_org = st.selectbox('Select a Song',song_list)
    st.write('You entered :',song_name_org)

    # lowercase the input
    song_name = song_name_org.lower()

    # select no. of recommendation
    k = st.selectbox('How many recommendation you need?',[5,10,15,20],index=1)

    # Button
    if st.button('Get Recommendation'):
        if (cleaned_data['name'] == song_name).any():
            st.write('Recommendation for', f"**{song_name_org}**")
            recommendations = content_recommendation(song_name,cleaned_data,transformed_data,k)

            # Display recommendations
            for ind, recommendation in recommendations.iterrows():
                song_name = recommendation['name'].title()
                artist_name = recommendation['artist'].title()

                if ind == 0:
                    st.markdown('## Currently Playing')
                    st.markdown(f'### **{song_name}** by **{artist_name}**')
                    st.audio(recommendation['spotify_preview_url'])
                    st.write('---')

                elif ind == 1:
                    st.markdown('### Next Up 🎵')
                    st.markdown(f"### {ind}. **{song_name}** by **{artist_name}**")
                    st.audio(recommendation['spotify_preview_url'])
                    st.write('---')

                else:
                    st.markdown(f"### {ind}. **{song_name}** by **{artist_name}**")
                    st.audio(recommendation['spotify_preview_url'])
                    st.write('---')

        else:
            st.write(f"Sorry, we couldn't find {song_name} in our database. Please try another song.")

if __name__ == "__main__":
    main()
    




