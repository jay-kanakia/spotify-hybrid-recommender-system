import numpy as np
import pandas as pd
import joblib
import logging

from sklearn.preprocessing import MinMaxScaler,OneHotEncoder,StandardScaler
from category_encoders.count import CountEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import cosine_similarity
from content_filtering_data_cleaning import data_for_content_filtering
from scipy.sparse import save_npz
from pathlib import Path
from typing import Optional

# create a logger
logger = logging.getLogger('content_filtering_data_transformation')
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

def load_data(data_path:Path)->pd.DataFrame:

    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        logger.error('File not found at the location')

    return df

def train_transformer(df:pd.DataFrame)->ColumnTransformer:

    """
    Trains a ColumnTransformer on the provided data.
    The ColumnTransformer applies the following transformations:
    - Frequency Encoding using CountEncoder on specified columns.
    - One-Hot Encoding using OneHotEncoder on specified columns.
    - TF-IDF Vectorization using TfidfVectorizer on a specified column.
    - Standard Scaling using StandardScaler on specified columns.
    - Min-Max Scaling using MinMaxScaler on specified columns.
    Parameters:
    data (pd.DataFrame): The input data to be transformed.
    Returns:
    None
    """

    # cols to transform
    frequency_enode_cols = ['year']
    ohe_cols = ['artist',"time_signature","key"]
    tfidf_col = ['tags']
    standard_scale_cols = ["duration_ms","loudness","tempo"]
    min_max_scale_cols = ["danceability","energy","speechiness","acousticness","instrumentalness","liveness","valence"]

    # transformer 
    transformer = ColumnTransformer(transformers=[
        ("frequency_encode", CountEncoder(normalize=True,return_df=False), frequency_enode_cols),
        ("ohe", OneHotEncoder(handle_unknown="ignore"), ohe_cols),
        ("tfidf", TfidfVectorizer(max_features=85), tfidf_col[0]),
        ("standard_scale", StandardScaler(), standard_scale_cols),
        ("min_max_scale", MinMaxScaler(), min_max_scale_cols)
    ],remainder='passthrough',n_jobs=-1,force_int_remainder_cols=False)

    # fit the transformer
    transformer.fit(df)

    return transformer

def save_transformer(transformer:ColumnTransformer,model_path:Path)->None:

    """
    Saves:
    transformer.joblib: The trained ColumnTransformer object.
    """

    joblib.dump(transformer,model_path)

def transform_data(df:pd.DataFrame,transformer:ColumnTransformer)->pd.DataFrame:

    """
    Transforms the input data using a pre-trained transformer.
    Args:
        data (array-like): The data to be transformed.
    Returns:
        array-like: The transformed data.
    """

    transformed_data = transformer.transform(df)

    return transformed_data

def save_transformed_data(transformed_data:pd.DataFrame,trans_path:Path)->None:

    """
    Save the transformed data to a specified file path.

    Parameters:
    transformed_data (scipy.sparse.csr_matrix): The transformed data to be saved.
    save_path (str): The file path where the transformed data will be saved.

    Returns:
    None
    """

    # save the transformed data
    save_npz(trans_path,transformed_data)

def calculate_similarity_score(input_vector:np.array,transformed_data:pd.DataFrame)->np.array:

    """
    Calculate similarity scores between an input vector and a dataset using cosine similarity.
    Args:
        input_vector (array-like): The input vector for which similarity scores are to be calculated.
        data (array-like): The dataset against which the similarity scores are to be calculated.
    Returns:
        array-like: An array of similarity scores.
    """

    # calculate similarity score
    similarity_score = cosine_similarity(input_vector,transformed_data)

    return similarity_score

def content_recommendation(song_name:str,artist_name:str,songs_data:pd.DataFrame,transformed_data:np.array,k:Optional[int]=10)->pd.DataFrame:

    """
    Recommends top k songs similar to the given song based on content-based filtering.

    Parameters:
    song_name (str): The name of the song to base the recommendations on.
    artist_name (str): The name of the artist of the song.
    songs_data (DataFrame): The DataFrame containing song information.
    transformed_data (ndarray): The transformed data matrix for similarity calculations.
    k (int, optional): The number of similar songs to recommend. Default is 10.

    Returns:
    DataFrame: A DataFrame containing the top k recommended songs with their names, artists, and Spotify preview URLs.
    """

    # convert song name to lowercase
    song_name = song_name.lower()
    
    # convert the artist name to lowercase
    artist_name = artist_name.lower()

    # filter out the song from data
    song_row = songs_data.loc[((songs_data['name'] == song_name) & (songs_data['artist'] == artist_name)) ]

    # get the index of song
    song_index = song_row.index[0]

    # generate the input vector
    input_vector = transformed_data.loc[song_index]

    # calculate similarity scores
    similarity_score = cosine_similarity(input_vector,transformed_data  )

    # get the top k songs
    top_k_songs_indexes = np.argsort(similarity_score,axis=1).ravel()[-k-1:][::-1]
    
    # get the top k songs names
    top_k_songs_names = songs_data.iloc[top_k_songs_indexes]

    # print the top k songs
    top_k_list = top_k_songs_names[['name','artist','spotify_preview_url']].reset_index(drop=True)
    
    return top_k_list

def main():
    """
    Test the recommendations for a given song using content-based filtering.

    Parameters:
    data_path (str): The path to the CSV file containing the song data.

    Returns:
    None: Prints the top k recommended songs based on content similarity.
    """

    root_path = Path(__file__).parent.parent.parent

    file_path = root_path/"data"/"cleaned"/"df_songs_cleaned.csv"

    # load the data
    df = load_data(file_path)
    logger.info("Data loaded successfully")

    # clean the data
    filtered_data_content = data_for_content_filtering(df)
    logger.info("Data filtering successfully")

    # training a transformer
    transformer = train_transformer(filtered_data_content)
    logger.info("Tranformer object created")

    # save transformer
    model_dir = root_path/"models"
    model_dir.mkdir(exist_ok=True,parents=True)

    # save file name
    model_file_name = "transformer.joblib"

    # save file path
    model_path = model_dir/model_file_name    

    save_transformer(transformer,model_path=model_path)

    # transform the data
    transformed_data = transform_data(filtered_data_content,transformer)
    logger.info("Data transformed successfully")
    
    # save the transformed data

    trans_dir = root_path/"data"/"processed"
    trans_dir.mkdir(exist_ok=True,parents=True)

    trans_file_name = "content_filtering_transformed_data.npz"

    trans_path = trans_dir/trans_file_name

    save_transformed_data(transformed_data,trans_path)
    logger.info("Data saved successfully")

if __name__ == "__main__":
    main()