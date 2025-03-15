import pandas as pd
import json
import os
import uuid
import asyncio
from model_client import openai_client
from constant import *
from tqdm import tqdm


# Function to get embeddings from OpenAI's Ada model
async def get_embeddings(text):
    response = await openai_client.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding


# Function to save embedding as a JSON file
async def save_embedding_as_json(DIR, embedding, embedding_id):
    file_path = os.path.join(DIR, f"{embedding_id}.json")
    with open(file_path, 'w') as f:
        json.dump(embedding, f)

async def main():
    for tc in ["TC4"]:

        # Load the Excel file
        file_path = f'../data/feedback_data_original/gpt_user_feedback - {tc}.xlsx'
        df = pd.read_excel(file_path)
        output_file_path = f'../data/feedback_data/gpt_user_feedback - {tc}.xlsx'
        query_embedding_dir = f'../data/feedback_data/query_embedding_json - {tc}/'
        response_embedding_dir = f'../data/feedback_data/response_embedding_json - {tc}/'

        if not os.path.exists(query_embedding_dir):
            os.makedirs(query_embedding_dir)

        if not os.path.exists(response_embedding_dir):
            os.makedirs(response_embedding_dir)

        # Apply the function to the user_query and response columns
        df['user_query_id'] = df['user_query'].apply(lambda x: str(uuid.uuid4()))
        df['response_id'] = df['response'].apply(lambda x: str(uuid.uuid4()))

        for idx, row in tqdm(df.iterrows(), total=len(df)):
            user_query_embedding = await get_embeddings(row['user_query'])
            response_embedding = await get_embeddings(row['response'])
            await save_embedding_as_json(query_embedding_dir, user_query_embedding, row['user_query_id'])
            await save_embedding_as_json(response_embedding_dir, response_embedding, row['response_id'])

        # Drop the original columns and rename the new ones
        df.drop(columns=['user_query', 'response'], inplace=True)
        df.rename(columns={'user_query_id': 'user_query', 'response_id': 'response'}, inplace=True)

        # Save the DataFrame to an Excel file
        df.to_excel(output_file_path, index=False)

if __name__ == '__main__':
    asyncio.run(main())
