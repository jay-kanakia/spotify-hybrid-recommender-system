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
    transformed_data_path = root_path/"data"/"processed"/"content_filtering_transformed_data.npz"

    # load transformed data
    st.session_state.transformed_data = load_npz_data(transformed_data_path)

    # transformed hybrid data path
    transformed_hybrid_data_path = root_path/"data"/"processed"/"hybrid_transformed_filtered_data.npz"

    # load transformed hybrid data
    st.session_state.transformed_hybrid_data = load_npz_data(transformed_hybrid_data_path)
    logger.info("Transformed data loaded successfully")

    # cleaned data path
    cleaned_data_path = root_path/"data"/"cleaned"/"df_songs_cleaned.csv"

    # load clean data
    st.session_state.df_songs = load_data(cleaned_data_path)
    logger.info("Cleaned data loaded successfully")

    # load original data
    # org_data_path = root_path/"data"/"raw"/"Music Info.csv"
    # st.session_state.org_data = load_data(org_data_path)

    # filtered data path
    filtered_path = root_path/"data"/"filtered"/"Colab_filtered_data.csv"
    st.session_state.df_filtered = load_data(filtered_path)

    # interaction matrix path
    interaction_matrix_path = root_path/"data"/"processed"/"interaction_matrix.npz"

    # load interaction matrix
    st.session_state.interaction_mat = load_npz_data(interaction_matrix_path)

    # track_ids path
    track_id_path = root_path/"data"/"track_ids.npy"

    # load track_ids
    st.session_state.track_ids = load_np(track_id_path)

# ===========================================================================================
    st.sidebar.title("📂 Navigation")

    section = st.sidebar.radio(
        "Go to:",
        [
            "🎧 Recommender",
            "🧠 How It Works",
            "📊 Dataset & Scale",
            "☁️ Architecture",
            "👨‍💻 About"
        ]
    )

    if section == "🎧 Recommender":
        st.title("🎧 Spotify Hybrid Recommender")

        # your input + output UI here

        st.info("""
            💡 Try exploring artists like **Radiohead, Bob Dylan, or Metallica**

            🎧 Dataset highlights:
            - English songs released between **1960 – 2020**
            - Strong representation from the **2000–2010 era**
            - Prominent artists include *The Rolling Stones, Radiohead, Tom Waits, and Johnny Cash*
            - Dominant genres: **Rock, Electronic, Metal**, along with Pop, Rap, and Jazz
            """)
    
# ============================================================================================

        # title
        #st.title("Welcome to the Spotify Song Recommender!")

        # sub header
        st.write('### Enter the name of a song and the recommender will suggest similar songs 🎵🎧')

        # text input
        song_list = st.session_state.df_songs['name'].to_list()
        artist_list = st.session_state.df_songs['artist'].to_list()

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
        #artist_name = label_to_song[selected]["artist"]

        st.write('You entered :',selected)

        # lowercase the input
        song_name = song_name_org.lower()

        # select no. of recommendation
        k = st.selectbox('How many recommendation you need?',[5,10,15,20],index=1)

        if (st.session_state.df_filtered['name'] == song_name).any():

            # type of filtering
            filtering_type = st.selectbox('Select the type of filtering:', ['Content-Based Filtering', 'Collaborative Filtering','Hybrid Recommender System'],index=2)

            # diversity slider
            diversity  = st.slider(label='Diversity in Recommendation',
                                        min_value=1,
                                        max_value=10,
                                        step=1,
                                        value=5
                                        )
            
            # content based weight
            weight_content_based = 1 - (diversity/10)

        else:

            # type of filtering
            filtering_type = st.selectbox(label='Select the type of filtering:',options=['Content-Based Filtering'],index=0)

        # Button
        if filtering_type == 'Content-Based Filtering':
            if st.button('Get Recommendation'):
                if (st.session_state.df_songs['name'] == song_name).any():
                    st.write('Recommendation for', f"**{selected}**")
                    recommendations = content_recommendation(song_name,st.session_state.df_songs,st.session_state.transformed_data,k)

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
                if (st.session_state.df_filtered['name'] == song_name).any():
                    st.write('Recommendation for', f"**{selected}**")
                    recommendations = collaborative_recommendation(song_name,st.session_state.track_ids,st.session_state.df_filtered,st.session_state.interaction_mat,k)

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
                if (st.session_state.df_filtered['name'] == song_name).any():
                    st.write('Recommendation for', f"**{selected}**")
                    recommender = HybridRecommenderSystem(number_of_recommendation=k,weight_content_based=weight_content_based)
                    recommendations = recommender.give_recommmendations(song_name,st.session_state.df_filtered,st.session_state.track_ids,st.session_state.transformed_hybrid_data,st.session_state.interaction_mat)
                    
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

    elif section == "🧠 How It Works":
        st.header("🧠 Model Overview")

        with st.expander("🔍 Hybrid Recommendation Logic"):
            st.markdown("""
                This system combines **content-based** and **collaborative filtering** to generate high-quality recommendations:

                **🎯 Content-Based Filtering**
                - Uses audio features such as **tempo, loudness, key, mode**
                - Includes advanced attributes like **acousticness, instrumentalness, speechiness, liveness**
                - Computes similarity between songs using feature vectors

                **🤝 Collaborative Filtering**
                - Built on **10 million user–song interactions**
                - Covers **1 million users** and **30K songs**
                - Captures implicit listening patterns and user behavior

                **⚖️ Hybrid Ranking**
                - Combines both approaches to balance **accuracy** and **discovery**
                - Includes a **diversity slider** to control personalization vs exploration
                """)

        with st.expander("⚙️ Algorithms Used"):
            st.markdown("""
                - 📐 **Cosine Similarity** for content-based recommendations  
                - 🧮 **Sparse Matrix Representations** for efficient large-scale computation  
                - ⚡ **Dask** for parallel processing and scalability  

                **🚀 Optimization Highlight**
                - Reduced memory usage from **~60 GB → ~31 MB**
                - Achieved using **SciPy sparse matrices + distributed computation**
                """)            

    elif section == "📊 Dataset & Scale":
        st.header("📊 Dataset & Scale")

        st.markdown("""
            ### 📊 Dataset & Scale

            - 🎵 50,000 songs (content-based filtering)  
            - 🤝 30,000 songs (collaborative filtering)  
            - 👥 1,000,000 users  
            - 🔄 10,000,000 user–song interactions  
            """)

        st.success("✅ Designed to simulate real-world recommendation systems at scale")

    elif section == "☁️ Architecture":
        st.header("☁️ Deployment Architecture")

        st.markdown("""
            ### ☁️ Deployment Architecture

            - 🖥️ **AWS EC2 instances** running the Streamlit application  
            - 🔁 **Auto Scaling Group (ASG)** with launch templates to dynamically scale based on traffic/load  
            - 🌐 **Load Balancer** for efficient request distribution across instances  
            - 📦 **Stateless architecture** enabling horizontal scalability  

            **🚀 Deployment Strategy**
            - 🔄 **Blue-Green Deployment** ensures zero-downtime releases  
            - ✅ All instances run consistent, version-controlled code  
            - ↩️ **Rollback mechanism** in place for quick recovery in case of failure  
            """)

        st.info("🚀 This setup ensures reliability under varying user load")

    elif section == "👨‍💻 About":
        st.header("👨‍💻 About the Project")

        st.markdown("""
        **Jay Kanakia**  
        Spotify Hybrid Recommender System  

        🔗 GitHub: https://github.com/jay-kanakia  
        🌐 Portfolio: https://abc.com  
        💼 LinkedIn: https://www.linkedin.com/in/jaykanakia-mlops/?skipRedirect=true 
        """)

if __name__ == "__main__":
    main()
    




