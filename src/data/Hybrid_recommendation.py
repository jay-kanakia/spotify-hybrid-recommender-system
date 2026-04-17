import pandas as pd
import numpy as np
import logging

from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from numpy import load
from scipy.sparse import load_npz

# create a logging object
logger = logging.getLogger('Hybrid recommender System')
logger.setLevel(logging.INFO)

# create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# create formatter and add to handler
formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# add handler to logger
logger.addHandler(console_handler)

def load_data(data_path:Path)->pd.DataFrame:

    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        logger.error('File not found error')
    
    return df

def load_np(track_ids_path:Path):
    """
    load track ids
    """
    track_ids = load(track_ids_path,allow_pickle=True)

    return track_ids

def load_npz_data(trans_data_path:Path):

    try:
        # load transformed data
        transformed_data = load_npz(trans_data_path)
    except FileNotFoundError:
        logger.error("Transformed data file not found at location")
    return transformed_data

class HybridRecommenderSystem:

    def __init__(self,
                 number_of_recommendation:int,
                 weight_content_based:float):
        
        self.number_of_recommendation = number_of_recommendation
        self.weight_content_based = weight_content_based
        self.weight_colab_based = 1 - weight_content_based

    def __calculate_content_based_similarity(self,song_name,df_filtered,transformed_matrix):
        
        # lower casing
        song_name = song_name.lower().strip()

        # fetch the row from songs data
        song_row = df_filtered[df_filtered['name'] == song_name]
        
        # index value of song
        index = song_row.index[0]
        print(index)
        
        # fetch the input vector
        input_vector = transformed_matrix[index]
        
        # get similarity scores
        content_similarity_score = cosine_similarity(input_vector,transformed_matrix)

        print("Shape of content :",content_similarity_score.shape)
       
       # return similarity score
        return content_similarity_score
        logger.info('content_similarity_score calculated')
    
    def __calculate_colab_based_similarity(self,song_name,track_ids,df_filtered,interaction_mat):

        # lower casing
        song_name = song_name.lower().strip()

        # fetch the row from songs data
        song_row = df_filtered[df_filtered['name'] == song_name]
        
        # track_id of input song
        input_track_id = song_row['track_id'].values.item()
       
        # index value of track_id
        index = np.where(track_ids == input_track_id)[0].item()
        
        # fetch the input vector
        input_vector  = interaction_mat[index]
        
        # get similarity scores
        colab_similarity_score = cosine_similarity(input_vector,interaction_mat)
    
        print("Shape of colab :",colab_similarity_score.shape)

       # return similarity score
        return colab_similarity_score
    
    def __normalize_similarities(self,similarity_scores):

        minimum = np.min(similarity_scores)
        maximum = np.max(similarity_scores)

        normalized_scores = (similarity_scores - minimum)/(maximum - minimum)

        return normalized_scores
    
    def __weighted_combination(self,content_similarity_score,colab_similarity_score):

        weighted_scores = (self.weight_content_based*content_similarity_score) + (self.weight_colab_based*colab_similarity_score)

        return weighted_scores
    
    def give_recommmendations(self,song_name,df_filtered,track_ids,transformed_matrix,interaction_mat):

        # calculate content based similarities
        content_based_similarity = self.__calculate_content_based_similarity(song_name,df_filtered,transformed_matrix)


        # calculate colab based similarities
        colab_based_similarity = self.__calculate_colab_based_similarity(song_name,track_ids,df_filtered,interaction_mat)

        # normalized content based similarity
        normalized_content_based_similarity = self.__normalize_similarities(content_based_similarity)

        # normalized colab based similarity
        normalized_colab_based_similarity = self.__normalize_similarities(colab_based_similarity)

        # weighted combination
        weighted_scores = self.__weighted_combination(normalized_content_based_similarity,normalized_colab_based_similarity)

        # index values of recommendation
        recommmendation_indices = np.argsort(weighted_scores.ravel())[-self.number_of_recommendation-1:][::-1]

        # get top k recommmendations
        recommeded_track_ids = track_ids[recommmendation_indices]

        # get top scores
        top_scores = np.sort(weighted_scores.ravel())[-self.number_of_recommendation-1:][::-1]

        # get the song from data
        temp_df = pd.DataFrame(
            {
                'track_id' : recommeded_track_ids.tolist(),
                'score' : top_scores
            }
        )

        top_k_songs = (
            df_filtered
            .loc[df_filtered['track_id'].isin(recommeded_track_ids)]
            .merge(temp_df,on='track_id')
            .sort_values(by='score',ascending=False)
            .drop(columns=['track_id','score'])
            .reset_index(drop=True)
        )

        return top_k_songs
    
def main():

    # root path
    root_path = Path(__file__).parent.parent.parent

    # filtered path
    filtered_path = root_path/"data"/"filtered"/"Colab_filtered_data.csv"

    # filtered data
    df_filtered = load_data(filtered_path)

    # track path
    track_path = root_path/"data"/"track_ids.npy"

    # load track_ids
    track_ids = load_np(track_path)

    # transformed data path
    trans_path = root_path/"data"/"processed"/"hybrid_transformed_filtered_data.npz"

    # load transformed data
    df_transformed = load_npz_data(trans_path)

    # interaction matrix path
    interaction_mat_path = root_path/"data"/"processed"/"interaction_matrix.npz"

    # load interaction matrix
    interaction_mat = load_npz_data(interaction_mat_path)

    recommmed = HybridRecommenderSystem(5,0.3)

    top_k_songs = recommmed.give_recommmendations('six day wonder',df_filtered,track_ids,df_transformed,interaction_mat)

    print(top_k_songs)

if __name__ == "__main__":
    main()