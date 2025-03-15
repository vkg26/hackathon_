import time
import logging
from dotenv import load_dotenv
from model_client import openai_client
from constant import *
from utils import get_token_usage, calculate_cost
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


class RAGAgent:
    def __init__(self, rewritten_query, insights, keywords, contract_text):
        self.rewritten_query = rewritten_query
        self.insights = insights
        self.keywords = keywords
        self.contract_text = contract_text
        self.model = MODEL_AGENT_MAPPING["rag_agent"]

    async def filter_context(self):
        start_time = time.time()
        logger.info("RAG Agent started")

        response = await openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system",
                 "content": "You are an assistant that helps filter relevant parts of a contract based on a query."},
                {"role": "user", "content": self._generate_prompt()}
            ],
            seed=26
        )

        end_time = time.time()
        time_taken = end_time - start_time

        filtered_context = response.choices[0].message.content.strip()

        tokens_used = get_token_usage(response)

        cost = calculate_cost(input_tokens=tokens_used['input_tokens'], output_tokens=tokens_used['output_tokens'],
                              model=self.model)
        logger.info(f"RAG Agent -> Cost: {cost} dollars, Time Taken: {time_taken} seconds")

        return filtered_context

    def _generate_prompt(self):
        prompt = (
            f"Rewritten Query: {self.rewritten_query}\n"
            f"Insights: {self.insights}\n"
            f"Keywords: {self.keywords}\n"
            f"Contract Text: {self.contract_text}\n\n"
            "Filter and provide only the relevant parts of the contract based on the rewritten query, insights, "
            "and keywords. Do not provide any explanations or answer to the user query. Only filter the relevant part "
            "and only return that part without any additional text. Your final response should be as it as from the "
            "provided contract text, nothing outside of it."
        )
        return prompt


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text
