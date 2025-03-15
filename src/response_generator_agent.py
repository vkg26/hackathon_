import time
import logging
from dotenv import load_dotenv
from model_client import openai_client
from constant import *
from utils import get_token_usage, calculate_cost


logger = logging.getLogger(__name__)


class ResponseGeneratorAgent:
    def __init__(self, user_query, rewritten_query, insights, keywords, filtered_context):
        self.user_query = user_query
        self.rewritten_query = rewritten_query
        self.insights = insights
        self.keywords = keywords
        self.filtered_context = filtered_context
        self.model = MODEL_AGENT_MAPPING["response_generator_agent"]

    async def generate_response(self):
        start_time = time.time()
        logger.info("Response Generator Agent started")

        response = await openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system",
                 "content": "You are an intelligent and helpful legal assistant capable of processing and "
                            "understanding complex legal documents, extracting information from them, and providing "
                            "answers based on the provided document text."},
                {"role": "user", "content": self._generate_prompt()}
            ],
            seed=26
        )
        end_time = time.time()
        time_taken = end_time - start_time

        generated_response = response.choices[0].message.content.strip()

        tokens_used = get_token_usage(response)
        cost = calculate_cost(input_tokens=tokens_used['input_tokens'], output_tokens=tokens_used['output_tokens'],
                              model=self.model)
        logger.info(
            f"Response Generator Agent -> Cost: {cost} dollars, Time Taken: {time_taken} seconds")

        return generated_response

    def _generate_prompt(self):
        prompt = (
            f"User Query: {self.user_query}\n"
            f"Rewritten Query: {self.rewritten_query}\n"
            f"Insights: {self.insights}\n"
            f"Keywords: {self.keywords}\n"
            "Answer the above user query from the filtered context provided below. Use rewritten query, "
            "insights and keywords for more clarification. If the answer to above query is explicitly present in the "
            "filtered context, only then give the result. Else if the answer to above query is not explicitly present "
            "in the given text, then say 'No answer found'."
            f"Filtered Context: {self.filtered_context}\n\n"
        )
        return prompt

