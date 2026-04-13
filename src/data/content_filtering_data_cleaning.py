import logging
import pandas as pd

from pathlib import Path

# create the logger
logger = logging.getLogger('content_filtering_data_ingestion')
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

def data_cleaning(df:pd.DataFrame)->pd.DataFrame:
    """
    Cleans the input DataFrame by performing the following operations:
    1. Removes duplicate rows based on the 'spotify_id' column.
    2. Drops the 'genre' and 'spotify_id' columns.
    3. Fills missing values in the 'tags' column with the string 'no_tags'.
    4. Converts the 'name', 'artist', and 'tags' columns to lowercase.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data to be cleaned.

    Returns:
    pd.DataFrame: The cleaned DataFrame.
    """
    return(
        df
        .drop_duplicates(subset='spotify_id')
        .drop(columns=['genre','spotify_id'])
        .fillna({'tags':'no_tags'})
        .assign(
            name = lambda x : x['name'].str.lower(),
            artist = lambda x : x['artist'].str.lower(),
            tags = lambda x : x['tags'].str.lower()
        )
        .reset_index(drop=True)
    )

def data_for_content_filtering(df:pd.DataFrame)->pd.DataFrame:
    """
    Cleans the input DataFrame by dropping specific columns.

    This function takes a DataFrame and removes the columns "track_id", "name",
    and "spotify_preview_url". It is intended to prepare the data for content based
    filtering by removing unnecessary features.

    Parameters:
    data (pandas.DataFrame): The input DataFrame containing songs information.

    Returns:
    pandas.DataFrame: A DataFrame with the specified columns removed.
    """
    return(
        df
        .drop(columns=["track_id","name","spotify_preview_url"])
    )

def save_data(df:pd.DataFrame,save_path:Path)->None:

    df.to_csv(save_path,index=False)


def main()->None:
    """
    Main function to load, clean, and save data.
    Parameters:
    data_path (str): The file path to the raw data CSV file.
    Returns:
    None
    """

    root_path = Path(__file__).parent.parent.parent

    data_path = root_path/"data"/"raw"/"Music Info.csv"


    # load the data
    df_songs = load_data(data_path)
    logger.info('Data loaded successfully')

    # clean the data
    df_songs = data_cleaning(df_songs)
    logger.info("Data cleaned successfully")

    # save the data
    save_path_dir = root_path/"data"/"cleaned"

    # check for directory
    save_path_dir.mkdir(exist_ok=True,parents=True)

    # file name
    save_file_name = "df_songs_cleaned.csv"

    # save path
    save_path = save_path_dir/save_file_name

    save_data(df_songs,save_path)
    logger.info("Data saved successfully")

if __name__ == "__main__":

    main()
    