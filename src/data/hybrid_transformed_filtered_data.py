import pandas as pd
import logging
import joblib

from src.data.content_filtering_data_cleaning import data_for_content_filtering
from src.data.content_filtering_data_transformation import transform_data,save_transformed_data
from pathlib import Path
from sklearn.compose import ColumnTransformer

# create the logger
logger = logging.getLogger('Hybrid recommended transformed filtered data')
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
        logger.error('File not found error')
    
    return df

def load_transformer(transformer_path:Path)->ColumnTransformer:

    transformer = joblib.load(transformer_path)

    return transformer

def main():
    
    # root path
    root_path = Path(__file__).parent.parent.parent

    # filtered data path
    filtered_data_path = root_path/"data"/"filtered"/"Colab_filtered_data.csv"

    # load filtered data
    df_filtered = load_data(filtered_data_path)
    logger.info('Filtered data loaded')

    # cleaning filtered data
    df_filtered_cleaning = data_for_content_filtering(df_filtered)
    logger.info('Filtered data cleaning performed')

    # transformer path
    transformer_path = root_path/"models"/"transformer.joblib"

    # load transformer
    transformer = load_transformer(transformer_path)

    # transform the data into matrix
    df_transformed = transform_data(df_filtered_cleaning,transformer)
    logger.info('Filtered data transfomed')

    # transformed data save path
    transformed_data_path = root_path/"data"/"processed"/"hybrid_transformed_filtered_data.npz"

    # save the transformed data
    save_transformed_data(transformed_data=df_transformed,trans_path=transformed_data_path)
    logger.info('Transformed data saved successfully')

    

if __name__ == '__main__':
    main()