import re
import time
import logging
from dotenv import load_dotenv
from constant import *
from utils import get_token_usage, calculate_cost
from model_client import openai_client


logger = logging.getLogger(__name__)


class QueryInterpretationAgent:
    def __init__(self, query):
        self.query = query
        self.model = MODEL_AGENT_MAPPING["query_interpretation_agent"]

    async def interpret_query(self):
        start_time = time.time()
        logger.info(f"Query Interpretation Agent started")

        response = await openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an assistant that helps rewrite and analyze queries."},
                {"role": "user", "content": self._generate_prompt()}
            ],
            seed=26
        )

        end_time = time.time()
        time_taken = end_time - start_time

        response_text = response.choices[0].message.content.strip()
        # Process response to extract rewritten query, insights, and keywords
        rewritten_query, insights, keywords = self._process_response(response_text)

        tokens_used = get_token_usage(response)

        cost = calculate_cost(input_tokens=tokens_used['input_tokens'], output_tokens=tokens_used['output_tokens'],
                              model=self.model)
        logger.info(
            f"Query Interpretation Agent -> Cost: {cost} dollars, Time Taken: {time_taken} seconds")

        return rewritten_query, insights, keywords

    def _generate_prompt(self):
        prompt = (
            f"User query: {self.query}\n\n"
            "Rewrite the query in an elaborated format with correct spelling and grammar. "
            "Extract key insights and a list of relevant keywords from the query. "
            "Provide the output in the following format:\n\n"
            "Rewritten Query: <rewritten query>\n"
            "Insights: <insights>\n"
            "Keywords: [<keywords>]\n"
        )
        return prompt

    def _process_response(self, response_text):
        rewritten_query_match = re.search(r"Rewritten Query: (.+?)(?=\n|$)", response_text)
        insights_match = re.search(r"Insights:\s*(.*?)(?=Keywords:|$)", response_text, re.DOTALL)
        keywords_match = re.search(r"Keywords: \[(.+?)\]", response_text)

        rewritten_query = rewritten_query_match.group(1).strip() if rewritten_query_match else ""
        insights = insights_match.group(1).strip() if insights_match else ""
        keywords = keywords_match.group(1).strip() if keywords_match else ""

        return rewritten_query, insights, keywords