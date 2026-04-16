import pandas as pd
import numpy as np
import logging
import dask.dataframe as dd

from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import save_npz,csr_matrix
from pathlib import Path

# create a logging object
logger = logging.getLogger('Collabrative-based-filtering')
logger.setLevel(logging.INFO)

# create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# create formatter and add to handler
formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# add handler to logger
logger.addHandler(console_handler)

def load_songs_data(data_path:Path)->pd.DataFrame:

    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        logger.error('File not found at given loaction')

    return df

def load_users_data(data_path:Path)->pd.DataFrame:

    try:
        df = dd.read_csv(data_path)
    except FileNotFoundError:
        logger.error('File not found at given loaction')

    return df

def filtered_data(df_songs:pd.DataFrame,df_users:pd.DataFrame)->pd.DataFrame:

    """
    Filter the songs data for the given track ids
    """

    # get track ids from user data
    track_ids = df_users['track_id']

    # filtering songs
    df_songs = df_songs[df_songs['track_id'].isin(track_ids)]

    # reseting the index
    df_songs.reset_index(drop=True,inplace=True)

    return df_songs


def save_filtered_data(df:pd.DataFrame,save_filtered_data_path:Path)->None:
    
    """
    Save the data to a csv file
    """
    df.to_csv(save_filtered_data_path,index=False)


def interaction_matrix(df_user,track_id_path):
    
    # creating a copy
    df = df_user.copy()

    # convert string column to categorical
    df = df.categorize(columns = ['user_id','track_id'])

    # convert user_id and track_id to numeric indices
    user_mapping = df['user_id'].cat.codes
    track_mapping = df['track_id'].cat.codes

    # get the list of track_ids
    track_ids = df['track_id'].cat.categories.values

    # save the categories
    np.save(file=track_id_path,arr=track_ids,allow_pickle=True)

    # add user and track index to df
    df = df.assign(
        user_idx = user_mapping,
        track_idx = track_mapping
    )

    # create interaction matrix
    interaction_matrix = df.groupby(['track_idx','user_idx'])['playcount'].sum().reset_index()

    # compute interaction matrix
    interaction_matrix = interaction_matrix.compute()

    # get the indices to form sparse matrix
    row_indicies = interaction_matrix['track_idx']
    col_indicies = interaction_matrix['user_idx']
    values = interaction_matrix['playcount']

    # get the shape of sparse matrix
    n_tracks = row_indicies.nunique()
    n_users = col_indicies.nunique()

    # create sparse matrix
    interaction_matrix = csr_matrix(
        (values,(row_indicies,col_indicies)),
        shape=(n_tracks,n_users)
    )

    # return sparse matrix
    return interaction_matrix

def save_interaction_matrix(interaction_matrix,interaction_matrix_path):

    """
    Save the sparse matrix to a npz file
    """

    save_npz(interaction_matrix_path,interaction_matrix)

def collaborative_recommendation(song_name,artist_name,track_ids,df_songs,interaction_matrix,k=5):

    # lower case song and artist name
    song_name = song_name.lower()
    artist_name = artist_name.lower()

    # fetch the row from songs data
    song_row = df_songs.loc[(df_songs['name'] == song_name) & (df_songs['artist'] == artist_name)]

    # track id of input song
    input_track_id = song_row.loc['track_id'].values.item()

    # index value of track id
    input_track_index = np.where(track_ids == input_track_id)[0].item()

    # fetch the input vector
    input_vector = interaction_matrix[input_track_index]

    # get similarity scores
    similarity_score  = cosine_similarity(input_vector,interaction_matrix)

    # index values of recommendations
    recommendation_indices = np.argsort(similarity_score.ravel())[-k-1:][::-1]

    # get top k recommendations
    recommended_track_ids = track_ids(recommendation_indices)

    # get top scores
    top_scores = np.sort(similarity_score.ravel())[-k-1:][::-1]

    # get the songs from data
    temp_df = pd.DataFrame(
        {
            'track_ids' : recommendation_indices.tolist(),
            'scores' : top_scores
        }
    )

    top_k_songs = (
        df_songs
        .loc[df_songs['track_id'].isin(recommended_track_ids)]
        .merge(temp_df,on='track_id')
        .sort_values(by='scores',ascending=False)
        .drop(columns=['track_id','scores'])
        .reset_index(drop=True)
    )

    return top_k_songs

def main():

    # root path
    root_path = Path(__file__).parent.parent.parent

    # cleaned data path
    cleaned_data_path = root_path/"data"/"cleaned"/"df_songs_cleaned.csv"

    # load clean data
    df_songs = load_songs_data(data_path=cleaned_data_path)
    logger.info('Songs data loaded')

    # user data path
    user_data_path = root_path/"data"/"raw"/"User Listening History.csv"

    # load user data
    df_users = load_users_data(data_path=user_data_path)
    logger.info('User data loaded')

    # filtering the data
    df_filtered = filtered_data(df_songs,df_users)
    logger.info('User data filtered')

    # filtered data save path
    filter_data_path = root_path/"data"/"filtered"/"Colab_filtered_data.csv"

    # save filtered data
    save_filtered_data(df_filtered,filter_data_path)
    logger.info('filtered data saved')

    # track id save path
    track_id_path = root_path/"data"/"track_ids.npy"


    # create the interaction matrix
    interaction_mat = interaction_matrix(df_users,track_id_path)
    logger.info('interaction matrix created')

    # interaction matrix save path
    interaction_matrix_path = root_path/"data"/"processed"/"interaction_matrix.npz"

    # save interaction matrix
    save_interaction_matrix(interaction_mat,interaction_matrix_path)
    logger.info('interaction matrix saved')

if __name__ == "__main__":
    main()
