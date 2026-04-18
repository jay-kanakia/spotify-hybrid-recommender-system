# set up the base image
FROM python:3.12-slim

# set the working directory
WORKDIR /app/

# copy the requirements file to workdir
COPY requirements.txt .

# install the requirements
RUN pip install --no-cache-dir -r requirements.txt

# copy data files
COPY ./data ./data


# COPY ./data/filtered/Colab_filtered_data.csv \
#      ./data/processed/interaction_matrix.npz \
#      ./data/processed/hybrid_transformed_filtered_data.npz \
#      ./data/track_ids.npy \
#      ./data/cleaned/df_songs_cleaned.csv \
#      ./data/processed/content_filtering_transformed_data.npz \
#      ./data/

# copy all required python scripts
COPY app.py .
COPY src/ src/ 

# COPY app.py \
#      ./src/data/collaborative_filtering_data_transformation.py \
#      ./src/data/hybrid_transformed_filtered_data.py \
#      ./src/data/Hybrid_recommendation.py \
#      ./src/data/content_filtering_data_cleaning.py \
#      ./src/data/content_filtering_data_transformation.py \
#      ./

# expose the port in the container
EXPOSE 8000

# run the streamlit app
CMD ["streamlit","run","app.py","--server.port","8000"]