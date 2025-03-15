import pandas as pd
import openai
import re
import time
import logging
import numpy as np
from dotenv import load_dotenv
import os
from model_client import openai_client
from constant import *
import json
from utils import get_token_usage, calculate_cost
from concurrent.futures import ThreadPoolExecutor
import functools
import asyncio

logger = logging.getLogger(__name__)


# Get OpenAI embeddings
@functools.lru_cache(maxsize=128)  # Cache results of this function to avoid redundant API calls
async def get_openai_embeddings(text):
    response = await openai_client.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL)

    return np.array(response.data[0].embedding)


# Calculate cosine similarity
def cosine_similarity(embeddings1, embeddings2):
    dot_product = np.dot(embeddings1, embeddings2.T)
    norm1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
    norm2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)
    return dot_product / (norm1 * norm2.T)


# Select top N cosine similarity
def select_top_cosine_similarity(embeddings_dict, query_embedding, top_n=COSINE_SIMILAR_FEEDBACK):
    feedback_embeddings = np.vstack(list(embeddings_dict.values()))
    similarities = cosine_similarity(feedback_embeddings, query_embedding.reshape(1, -1)).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    return [list(embeddings_dict.keys())[i] for i in top_indices]


# Save selected feedback
def save_selected_feedback(selected_feedback, output_path):
    selected_feedback.to_excel(output_path, index=False)

# Function to load embeddings concurrently
def load_embedding(embedding_id, embedding_dir):
    file_path = os.path.join(embedding_dir, f"{embedding_id}.json")
    with open(file_path, 'r') as f:
        embedding = np.array(json.load(f))
    return embedding_id, embedding

# Feedback extraction agent
def _generate_specific_prompt(specific_feedback):
    avg_verbosity = specific_feedback['verbosity'].mean()
    avg_accuracy = specific_feedback['accuracy'].mean()
    avg_completeness = specific_feedback['completeness'].mean()
    avg_quality = specific_feedback['quality'].mean()
    avg_relevancy = specific_feedback['relevancy'].mean()
    avg_emotional = specific_feedback['emotional'].mean()
    avg_safety = specific_feedback['safety'].mean()
    concatenated_free_text = specific_feedback['free_text'].str.cat(sep=' ')
    most_frequent_feedback = specific_feedback['feedback'].mode()[0]

    prompt = (
        f"specific query feedback - This is the feedback given by the user when a specific query as the current "
        f"question was fired by the user last time. This feedback is very important to reinforce the current "
        f"response as user has provided some feedback on the similar question last time. Based on the following "
        f"information for the specific feedback data:\n\n"
        f"Verbosity - comprehensiveness of the answer: {avg_verbosity:.2f}/10\n"
        f"Accuracy - correctness of the answer: {avg_accuracy:.2f}/10\n"
        f"Completeness - completeness of the answer: {avg_completeness:.2f}/10\n"
        f"Quality - quality in the terms of language and vocabulary used: {avg_quality:.2f}/10\n"
        f"Relevancy - relevancy of the response related to user query: {avg_relevancy:.2f}/10\n"
        f"Emotional - human-like emotional touch in the response: {avg_emotional:.2f}/10\n"
        f"Safety - safety of the response in terms of toxic/unsafe words used: {avg_safety:.2f}/10\n"
        f"Free_Text - free flowing natural language feedback by the user: {concatenated_free_text}\n"
        f"Feedback - overall boolean feedback by the user: {most_frequent_feedback}\n\n"
        "Generate a paragraph describing how to change the response for the current user query in natural "
        "language. For example, users might have given a low score on verbosity and quality but a high score on "
        "accuracy, so the feedback pattern can be: 'The response is pretty accurate, but I would have liked it "
        "more if it was more verbose with additional details and the quality of the response was better in terms "
        "of the language used. Conciseness was fine. No work needed there.'\n\n"
        "Follow the following rules:\n\nRule1: Be very concise and crisp while generating this paragraph. This "
        "should contain around 2-3 sentences and should have very specific instructions, nothing vague.\nRule2: "
        "It is not always necessary to change the response. If user has given a positive feedback in all aspects, "
        "then no need to forcefully generate a negative feedback, Just say that user is happy with the "
        "response."
    )
    return prompt


def _generate_general_prompt(complete_feedback):
    avg_verbosity = complete_feedback['verbosity'].mean()
    avg_accuracy = complete_feedback['accuracy'].mean()
    avg_completeness = complete_feedback['completeness'].mean()
    avg_quality = complete_feedback['quality'].mean()
    avg_relevancy = complete_feedback['relevancy'].mean()
    avg_emotional = complete_feedback['emotional'].mean()
    avg_safety = complete_feedback['safety'].mean()

    prompt = (
        f"general user feedback - This is the feedback received from the overall activity of the user. It is not "
        f"specific to the query which is currently asked, but more related to the alignment needed by the user for "
        f"his personal preferences. Based on the following information for the complete feedback data:\n\n"
        f"Verbosity - comprehensiveness of the answer: {avg_verbosity:.2f}/10\n"
        f"Accuracy - correctness of the answer: {avg_accuracy:.2f}/10\n"
        f"Completeness - completeness of the answer: {avg_completeness:.2f}/10\n"
        f"Quality - quality in the terms of language and vocabulary used: {avg_quality:.2f}/10\n"
        f"Relevancy - relevancy of the response related to user query: {avg_relevancy:.2f}/10\n"
        f"Emotional - human-like emotional touch in the response: {avg_emotional:.2f}/10\n"
        f"Safety - safety of the response in terms of toxic/unsafe words used: {avg_safety:.2f}/10\n\n"
        "Generate a paragraph describing the general user feedback patterns in natural language. Identify what "
        "kind of responses are liked by the user, for example some user might prefer a more emotional human like "
        "touch in the response, while others might prefer a more verbose response.\n\n"
        "Follow the following rules:\n\nRule1: Be very concise and crisp while generating this paragraph. This "
        "should contain around 2-3 sentences and should have very specific instructions, nothing vague.\nRule2: "
        "It is not always necessary to change the response. If user has given a positive feedback in all aspects, "
        "then no need to forcefully generate a negative feedback, Just say that user is happy with the "
        "response."
    )
    return prompt


class FeedbackExtractionAgent:
    def __init__(self, original_query, rewritten_query, insights, keywords, feedback_data_path,
                 filtered_feedback_data_path):
        self.original_query = original_query
        self.rewritten_query = rewritten_query
        self.insights = insights
        self.keywords = keywords
        self.feedback_data_path = feedback_data_path
        self.filtered_feedback_data_path = filtered_feedback_data_path
        self.query_embedding_dir = f'../data/feedback_data/query_embedding_json - {TEST_CASE}/'
        self.response_embedding_dir = f'../data/feedback_data/response_embedding_json - {TEST_CASE}/'

        start_time = time.time()
        logger.info("User Feedback Data Loading Agent started")

        self.feedback_data, self.query_embeddings, self.response_embeddings = self._load_feedback_data(
            feedback_data_path)

        end_time = time.time()
        logger.info(f"User Feedback Data Loading Agent -> Cost: {0} dollars, Time Taken: {end_time - start_time} seconds")

        self.model = MODEL_AGENT_MAPPING["feedback_extractor_agent"]

    def _load_feedback_data(self, feedback_data_path):
        feedback_data = pd.read_excel(feedback_data_path)

        with ThreadPoolExecutor() as executor:
            query_embedding_futures = {executor.submit(load_embedding, row['user_query'], self.query_embedding_dir): row['user_query'] for idx, row in feedback_data.iterrows()}
            response_embedding_futures = {executor.submit(load_embedding, row['response'], self.response_embedding_dir): row['response'] for idx, row in feedback_data.iterrows()}

        query_embeddings = {future.result()[0]: future.result()[1] for future in query_embedding_futures.keys()}
        response_embeddings = {future.result()[0]: future.result()[1] for future in response_embedding_futures.keys()}

        return feedback_data, query_embeddings, response_embeddings

    async def extract_feedback(self):
        start_time = time.time()
        logger.info("Feedback extractor Agent started")

        specific_feedback = await self._filter_specific_feedback()

        filtering_time = time.time()
        filter_feedback_time = filtering_time - start_time

        specific_query_feedback, general_user_feedback, cost = await self._generate_feedback(specific_feedback,
                                                                                       self.feedback_data)
        extraction_time = time.time()
        feedback_extraction_time = extraction_time - filtering_time

        end_time = time.time()
        total_time = end_time - start_time
        if cost is None:
            cost = 0
        logger.info(f"Filtering part of Agent -> Cost: 0 dollars, Filtering Time Taken: {filter_feedback_time} seconds")
        logger.info(
            f"Extractor part of Agent -> Cost: {cost} dollars, Feedback Time Taken: {feedback_extraction_time} seconds")
        logger.info(f"Feedback extractor Agent -> Cost: {cost} dollars, Time Taken: {total_time} seconds")

        return specific_query_feedback, general_user_feedback

    async def _filter_specific_feedback(self):
        query_embedding = await get_openai_embeddings(self.original_query)
        similarities = cosine_similarity(np.vstack(list(self.query_embeddings.values())),
                                         query_embedding.reshape(1, -1)).flatten()
        exact_match_index = np.argmax(similarities)
        if similarities[exact_match_index] == 1.0:  # Assuming exact match similarity score is 1.0
            exact_match_id = list(self.query_embeddings.keys())[exact_match_index]
            exact_match = self.feedback_data[self.feedback_data['user_query'] == exact_match_id]
            return exact_match.to_frame().T
        else:
            top_indices = select_top_cosine_similarity(self.query_embeddings, query_embedding)
            filtered_feedback_data = self.feedback_data.loc[self.feedback_data['user_query'].isin(top_indices)]
            save_selected_feedback(filtered_feedback_data, self.filtered_feedback_data_path)
            return filtered_feedback_data

    async def _generate_feedback(self, specific_feedback, complete_feedback):
        prompt = (f"You have to generate 2 types of feedbacks received from past user feedback data:\n\n"
                  f"1.) {_generate_specific_prompt(specific_feedback)}\n\n"
                  f"2.) {_generate_general_prompt(complete_feedback)} ")

        response = await openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system",
                 "content": "You are an assistant that helps analyze user feedback and generate specific query feedback patterns."},
                {"role": "user", "content": prompt}
            ],
            seed=26,
            tools=[
                {"type": "function",
                 "function": {"name": "feedback_post_processor",
                              "description": "Process specific query feedback",
                              "parameters": {
                                  "type": "object",
                                  "properties": {
                                      "specific_query_feedback": {"type": "string"},
                                      "general_user_feedback": {"type": "string"}
                                  },
                                  "required": ["specific_query_feedback", "general_user_feedback"]
                              }
                              }
                 }
            ],
            tool_choice="auto"
        )
        tokens_used = get_token_usage(response)

        cost = calculate_cost(input_tokens=tokens_used['input_tokens'], output_tokens=tokens_used['output_tokens'],
                              model=self.model)
        try:
            tool_calls = response.choices[0].message.tool_calls[0].function

            tool_function_name = tool_calls.name
            tool_function_id = response.choices[0].message.tool_calls[0].id
            arguments = json.loads(tool_calls.arguments)
            specific_query_feedback = arguments['specific_query_feedback']
            general_user_feedback = arguments['general_user_feedback']

            return specific_query_feedback, general_user_feedback, cost

        except Exception as e:
            logger.error("There is no tool returned from feedback extractor agent")
            return "", "", None
