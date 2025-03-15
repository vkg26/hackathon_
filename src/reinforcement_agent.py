import time
import logging
from dotenv import load_dotenv
from utils import get_token_usage, calculate_cost
from constant import *
from model_client import openai_client


logger = logging.getLogger(__name__)


class ReinforcementAgent:
    def __init__(self, user_query, rewritten_query, insights, keywords, filtered_context, generated_response,
                 specific_query_feedback, general_user_feedback):
        self.user_query = user_query
        self.rewritten_query = rewritten_query
        self.insights = insights
        self.keywords = keywords
        self.filtered_context = filtered_context
        self.generated_response = generated_response
        self.specific_query_feedback = specific_query_feedback
        self.general_user_feedback = general_user_feedback
        self.messages = []
        self.model = MODEL_AGENT_MAPPING["reinforcement_agent"]
        self.previous_actions = []

    async def formulate_response(self, previous_actions):
        self.previous_actions.extend(previous_actions)
        start_time = time.time()
        logger.info(f"Reinforcement Agent started")

        # Use GPT-4 to determine the appropriate action and arguments
        messages = self._generate_initial_messages()
        if self.messages:
            messages.extend(self.messages)

        response = await openai_client.chat.completions.create(
            model=self.model,
            messages=messages,
            seed=26,
            tools=[
                {"type": "function",
                 "function": {"name": "make_response_accurate",
                              "description": "Corrects the generated response if it is inaccurate based on feedback.",
                              "parameters": {
                                  "type": "object",
                                  "properties": {
                                      "user_query": {"type": "string"},
                                      "query_rewritten": {"type": "string"},
                                      "insights": {"type": "string"},
                                      "keywords": {"type": "string"},
                                      "response": {"type": "string"},
                                      "feedback": {"type": "string", "description": "part of the feedback which is relevant for this action only"}
                                  },
                                  "required": ["user_query", "query_rewritten", "insights", "keywords", "response", "feedback"]
                              }
                              }
                 },

                {"type": "function",
                 "function": {"name": "make_response_relevant",
                              "description": "Steers the generated response more towards the topic if needed based on feedback.",
                              "parameters": {
                                  "type": "object",
                                  "properties": {
                                      "user_query": {"type": "string"},
                                      "query_rewritten": {"type": "string"},
                                      "insights": {"type": "string"},
                                      "keywords": {"type": "string"},
                                      "response": {"type": "string"},
                                      "feedback": {"type": "string", "description": "part of the feedback which is relevant for this action only"}
                                  },
                                  "required": ["user_query", "query_rewritten", "insights", "keywords", "response", "feedback"]
                              }
                              }
                 },

                {"type": "function",
                 "function": {"name": "make_response_clear",
                              "description": "Cleans the generated response of ambiguous text or jargons for more clarity based on feedback.",
                              "parameters": {
                                  "type": "object",
                                  "properties": {
                                      "user_query": {"type": "string"},
                                      "query_rewritten": {"type": "string"},
                                      "insights": {"type": "string"},
                                      "keywords": {"type": "string"},
                                      "response": {"type": "string"},
                                      "feedback": {"type": "string", "description": "part of the feedback which is relevant for this action only"}
                                  },
                                  "required": ["user_query", "query_rewritten", "insights", "keywords", "response", "feedback"]
                              }
                              }
                 },

                {"type": "function",
                 "function": {"name": "make_response_verbose",
                              "description": "Elaborates the generated response with more details or make it concise and to the point based on feedback.",
                              "parameters": {
                                  "type": "object",
                                  "properties": {
                                      "user_query": {"type": "string"},
                                      "query_rewritten": {"type": "string"},
                                      "insights": {"type": "string"},
                                      "keywords": {"type": "string"},
                                      "response": {"type": "string"},
                                      "feedback": {"type": "string", "description": "part of the feedback which is relevant for this action only"}
                                  },
                                  "required": ["user_query", "query_rewritten", "insights", "keywords", "response", "feedback"]
                              }
                              }
                 },

                {"type": "function",
                 "function": {"name": "make_response_complete",
                              "description": "Completes the generated response with a complete answer including all details about what is asked in the question based on feedback.",
                              "parameters": {
                                  "type": "object",
                                  "properties": {
                                      "user_query": {"type": "string"},
                                      "query_rewritten": {"type": "string"},
                                      "insights": {"type": "string"},
                                      "keywords": {"type": "string"},
                                      "response": {"type": "string"},
                                      "feedback": {"type": "string", "description": "part of the feedback which is relevant for this action only"}
                                  },
                                  "required": ["user_query", "query_rewritten", "insights", "keywords", "response", "feedback"]
                              }
                              }
                 },

                {"type": "function",
                 "function": {"name": "make_response_quality",
                              "description": "Improves the quality of the generated response in terms of language and vocabulary used based on feedback.",
                              "parameters": {
                                  "type": "object",
                                  "properties": {
                                      "user_query": {"type": "string"},
                                      "query_rewritten": {"type": "string"},
                                      "insights": {"type": "string"},
                                      "keywords": {"type": "string"},
                                      "response": {"type": "string"},
                                      "feedback": {"type": "string", "description": "part of the feedback which is relevant for this action only"}
                                  },
                                  "required": ["user_query", "query_rewritten", "insights", "keywords", "response", "feedback"]
                              }
                              }
                 },

                {"type": "function",
                 "function": {"name": "make_response_emotional",
                              "description": "Improves the generated response to have a more human-like emotional touch based on feedback.",
                              "parameters": {
                                  "type": "object",
                                  "properties": {
                                      "user_query": {"type": "string"},
                                      "query_rewritten": {"type": "string"},
                                      "insights": {"type": "string"},
                                      "keywords": {"type": "string"},
                                      "response": {"type": "string"},
                                      "feedback": {"type": "string", "description": "part of the feedback which is relevant for this action only"}
                                  },
                                  "required": ["user_query", "query_rewritten", "insights", "keywords", "response", "feedback"]
                              }
                              }
                 },

                {"type": "function",
                 "function": {"name": "make_response_safe",
                              "description": "Makes the generated response safe in terms of toxic/unsafe words used based on feedback.",
                              "parameters": {
                                  "type": "object",
                                  "properties": {
                                      "user_query": {"type": "string"},
                                      "query_rewritten": {"type": "string"},
                                      "insights": {"type": "string"},
                                      "keywords": {"type": "string"},
                                      "response": {"type": "string"},
                                      "feedback": {"type": "string", "description": "part of the feedback which is relevant for this action only"}
                                  },
                                  "required": ["user_query", "query_rewritten", "insights", "keywords", "response", "feedback"]
                              }
                              }
                 },

                {"type": "function",
                 "function": {"name": "make_response_aligned",
                              "description": "Improves the generated response based on any other feedback received from user which cannot be covered in other actions.",
                              "parameters": {
                                  "type": "object",
                                  "properties": {
                                      "user_query": {"type": "string"},
                                      "query_rewritten": {"type": "string"},
                                      "insights": {"type": "string"},
                                      "keywords": {"type": "string"},
                                      "response": {"type": "string"},
                                      "feedback": {"type": "string", "description": "part of the feedback which is relevant for this action only"}
                                  },
                                  "required": ["user_query", "query_rewritten", "insights", "keywords", "response", "feedback"]
                              }
                              }
                 }
            ],
            tool_choice="auto"
        )

        end_time = time.time()
        time_taken = end_time - start_time
        tokens_used = get_token_usage(response)

        cost = calculate_cost(input_tokens=tokens_used['input_tokens'], output_tokens=tokens_used['output_tokens'],
                              model=self.model)
        logger.info(
            f"Reinforcement Agent -> Cost: {cost} dollars, Time Taken: {time_taken} seconds")

        return response

    def _generate_initial_messages(self):
        initial_message = {
            "role": "system",
            "content": "You are an assistant that helps formulate responses based on reinforcement learning principles."
        }
        user_message = {
            "role": "user",
            "content": (
                f"User Query: {self.user_query}\n"
                f"Rewritten Query: {self.rewritten_query}\n"
                f"Insights: {self.insights}\n"
                f"Keywords: {self.keywords}\n"
                f"Generated Response: {self.generated_response}\n"
                f"Specific Query Feedback: This is the feedback given by the user when a specific query as the "
                f"current question was fired by the user last time. This feedback is very important to reinforce the "
                f"current response as user has provided some feedback on the similar question last time. -> "
                f"{self.specific_query_feedback}\n\n"
                f"General User Feedback: this is the feedback received from the overall activity of the user. It is "
                f"not specific to the query which is currently asked, but more related to the alignment needed by the "
                f"user for his personal preferences. -> {self.general_user_feedback}\n\n"
                "Based on the above information, select the appropriate action from the following list  and provide the arguments required for this action:\n\n"
                "- make_response_accurate -> which corrects the generated response if it is inaccurate based on feedback\n"
                "- make_response_relevant -> which steers the generated response more towards the topic if needed based on feedback\n"
                "- make_response_clear -> which cleaned the generated response of ambiguous text or jargons for more clarity based on feedback\n"
                "- make_response_verbose -> which elaborates the generated response with more details or make it concise and to the point based on feedback\n"
                "- make_response_complete -> which completes the generated response with complete answer including complete details about what is asked in the question based on feedback\n"
                "- make_response_quality -> which improves the quality of the generated response answer in the terms of language and vocabulary used based on feedback\n"
                "- make_response_emotional -> which improves the generated response to have a more human-like emotional touch based on feedback\n"
                "- make_response_safe -> which makes the generated response safe in terms of toxic/unsafe words used based on feedback\n"
                "- make_response_aligned -> which improves the generated response based on any other feedback received from user which cannot be covered in other actions\n\n"
                "You need to follow below mentioned rules while selecting actions:\n"
                f"Rule 1: Do not select the same action multiple times. Check the previous actions selected list -> {self.previous_actions} and do not repeat the used action which is already present in this list. You will be penalized if you choose the action which is already present in this list. You can choose any action if this list is empty.\n"
                f"Rule 2: While passing the feedback argument for any action, first combine the specific and general feedback and then filter it to send only that part which is relevant for the selected action. Do not send the complete feedback blindly."
            )
        }

        return [initial_message, user_message]

    async def make_response_accurate(self, user_query, query_rewritten, insights, keywords, filtered_context, response, feedback):
        start_time = time.time()

        prompt = (
            f"Original Query: {user_query}\n"
            f"Rewritten Query: {query_rewritten}\n"
            f"Insights: {insights}\n"
            f"Keywords: {keywords}\n"
            f"Filtered Context: {filtered_context}\n"
            f"Response: {response}\n"
            f"Feedback: {feedback}\n\n"
            "Correct the generated response if it is inaccurate based on feedback. Only return the answer to the user "
            "query directly only from the filtered context provided. Do not give explanations or information about "
            "what this agent is trying to improve as it is a chatbot and user would only want to see the direct "
            "answer to the query."
        )
        response = await openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system",
                 "content": "You are an assistant that corrects the generated response if it is inaccurate based on feedback. "},
                {"role": "user", "content": prompt}
            ],
            seed=26
        )

        end_time = time.time()
        time_taken = end_time - start_time
        tokens_used = get_token_usage(response)

        cost = calculate_cost(input_tokens=tokens_used['input_tokens'], output_tokens=tokens_used['output_tokens'],
                              model=self.model)
        logger.info(
            f"Accuracy Action -> Cost: {cost} dollars, Time Taken: {time_taken} seconds")

        return response.choices[0].message.content.strip()

    async def make_response_relevant(self,  user_query, query_rewritten, insights, keywords, filtered_context, response, feedback):
        start_time = time.time()

        prompt = (
            f"Original Query: {user_query}\n"
            f"Rewritten Query: {query_rewritten}\n"
            f"Insights: {insights}\n"
            f"Keywords: {keywords}\n"
            f"Filtered Context: {filtered_context}\n"
            f"Response: {response}\n"
            f"Feedback: {feedback}\n\n"
            "Steer the generated response more towards the topic if needed based on feedback. Only return the answer "
            "to the user query directly. Do not give explanations or information about what this agent is trying to "
            "improve as it is a chatbot and user would only want to see the direct answer to the query."
        )
        response = await openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system",
                 "content": "You are an assistant that steers the generated response more towards the topic if needed based on feedback."},
                {"role": "user", "content": prompt}
            ],
            seed=26
        )

        end_time = time.time()
        time_taken = end_time - start_time
        tokens_used = get_token_usage(response)

        cost = calculate_cost(input_tokens=tokens_used['input_tokens'], output_tokens=tokens_used['output_tokens'],
                              model=self.model)
        logger.info(
            f"Relevancy Action -> Cost: {cost} dollars, Time Taken: {time_taken} seconds")

        return response.choices[0].message.content.strip()

    async def make_response_clear(self,  user_query, query_rewritten, insights, keywords, filtered_context, response, feedback):
        start_time = time.time()

        prompt = (
            f"Original Query: {user_query}\n"
            f"Rewritten Query: {query_rewritten}\n"
            f"Insights: {insights}\n"
            f"Keywords: {keywords}\n"
            f"Filtered Context: {filtered_context}\n"
            f"Response: {response}\n"
            f"Feedback: {feedback}\n\n"
            "Clean the generated response of ambiguous text or jargons for more clarity based on feedback. Only "
            "return the answer to the user query directly. Do not give explanations or information about what this "
            "agent is trying to improve as it is a chatbot and user would only want to see the direct answer to his "
            "query, not explanations or the process."
        )
        response = await openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system",
                 "content": "You are an assistant that cleans the generated response of ambiguous text or jargons for more clarity based on feedback."},
                {"role": "user", "content": prompt}
            ],
            seed=26

        )

        end_time = time.time()
        time_taken = end_time - start_time
        tokens_used = get_token_usage(response)

        cost = calculate_cost(input_tokens=tokens_used['input_tokens'], output_tokens=tokens_used['output_tokens'],
                              model=self.model)
        logger.info(
            f"Clarity Action -> Cost: {cost} dollars, Time Taken: {time_taken} seconds")

        return response.choices[0].message.content.strip()

    async def make_response_verbose(self,  user_query, query_rewritten, insights, keywords, filtered_context, response, feedback):
        start_time = time.time()

        prompt = (
            f"Original Query: {user_query}\n"
            f"Rewritten Query: {query_rewritten}\n"
            f"Insights: {insights}\n"
            f"Keywords: {keywords}\n"
            f"Filtered Context: {filtered_context}\n"
            f"Response: {response}\n"
            f"Feedback: {feedback}\n\n"
            "Elaborate the generated response with more details or make it concise and to the point based on "
            "feedback. Only return the answer to the user query directly. Do not give explanations or information "
            "about what this agent is trying to improve as it is a chatbot and user would only want to see the direct "
            "answer to the query."
        )
        response = await openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system",
                 "content": "You are an assistant that elaborates the generated response with more details or makes it concise and to the point based on feedback."},
                {"role": "user", "content": prompt}
            ],
            seed=26

        )

        end_time = time.time()
        time_taken = end_time - start_time
        tokens_used = get_token_usage(response)

        cost = calculate_cost(input_tokens=tokens_used['input_tokens'], output_tokens=tokens_used['output_tokens'],
                              model=self.model)
        logger.info(
            f"Verbosity Action -> Cost: {cost} dollars, Time Taken: {time_taken} seconds")

        return response.choices[0].message.content.strip()

    async def make_response_complete(self,  user_query, query_rewritten, insights, keywords, filtered_context, response, feedback):
        start_time = time.time()

        prompt = (
            f"Original Query: {user_query}\n"
            f"Rewritten Query: {query_rewritten}\n"
            f"Insights: {insights}\n"
            f"Keywords: {keywords}\n"
            f"Filtered Context: {filtered_context}\n"
            f"Response: {response}\n"
            f"Feedback: {feedback}\n\n"
            "Complete the generated response with a complete answer including all details about what is asked in the "
            "question based on feedback. Only return the answer to the user query directly. Do not give explanations "
            "or information about what this agent is trying to improve as it is a chatbot and user would only want to "
            "see the direct answer to the query."
        )
        response = await openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system",
                 "content": "You are an assistant that completes the generated response with a complete answer "
                            "including all details about what is asked in the question based on feedback."},
                {"role": "user", "content": prompt}
            ],
            seed=26

        )

        end_time = time.time()
        time_taken = end_time - start_time
        tokens_used = get_token_usage(response)

        cost = calculate_cost(input_tokens=tokens_used['input_tokens'], output_tokens=tokens_used['output_tokens'],
                              model=self.model)
        logger.info(
            f"Completeness Action -> Cost: {cost} dollars, Time Taken: {time_taken} seconds")

        return response.choices[0].message.content.strip()

    async def make_response_quality(self,  user_query, query_rewritten, insights, keywords, filtered_context, response, feedback):
        start_time = time.time()

        prompt = (
            f"Original Query: {user_query}\n"
            f"Rewritten Query: {query_rewritten}\n"
            f"Insights: {insights}\n"
            f"Keywords: {keywords}\n"
            f"Filtered Context: {filtered_context}\n"
            f"Response: {response}\n"
            f"Feedback: {feedback}\n\n"
            "Improve the quality of the generated response in terms of language and vocabulary used based on "
            "feedback. Try to include quality high quality richer and sophisticated words in your answer. Only return "
            "the answer to the user query directly. Do not give explanations or information about what this agent is "
            "trying to improve as it is a chatbot and user would only want to see the direct answer to the query."
        )
        response = await openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system",
                 "content": "You are an assistant that improves the quality of the generated response in terms of language and vocabulary used based on feedback."},
                {"role": "user", "content": prompt}
            ],
            seed=26

        )

        end_time = time.time()
        time_taken = end_time - start_time
        tokens_used = get_token_usage(response)

        cost = calculate_cost(input_tokens=tokens_used['input_tokens'], output_tokens=tokens_used['output_tokens'],
                              model=self.model)
        logger.info(
            f"Quality Action -> Cost: {cost} dollars, Time Taken: {time_taken} seconds")

        return response.choices[0].message.content.strip()

    async def make_response_emotional(self,  user_query, query_rewritten, insights, keywords, filtered_context, response, feedback):
        start_time = time.time()

        prompt = (
            f"Original Query: {user_query}\n"
            f"Rewritten Query: {query_rewritten}\n"
            f"Insights: {insights}\n"
            f"Keywords: {keywords}\n"
            f"Filtered Context: {filtered_context}\n"
            f"Response: {response}\n"
            f"Feedback: {feedback}\n\n"
            "Improve the generated response to have a more human-like emotional touch based on feedback. Only return "
            "the answer to the user query directly. Do not give explanations or information about what this agent is "
            "trying to improve as it is a chatbot and user would only want to see the direct answer to the query."
        )
        response = await openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system",
                 "content": "You are an assistant that improves the generated response to have a more human-like emotional touch based on feedback."},
                {"role": "user", "content": prompt}
            ],
            seed=26

        )

        end_time = time.time()
        time_taken = end_time - start_time
        tokens_used = get_token_usage(response)

        cost = calculate_cost(input_tokens=tokens_used['input_tokens'], output_tokens=tokens_used['output_tokens'],
                              model=self.model)
        logger.info(
            f"Emotional Touch Action -> Cost: {cost} dollars, Time Taken: {time_taken} seconds")

        return response.choices[0].message.content.strip()

    async def make_response_safe(self,  user_query, query_rewritten, insights, keywords, filtered_context, response, feedback):
        start_time = time.time()

        prompt = (
            f"Original Query: {user_query}\n"
            f"Rewritten Query: {query_rewritten}\n"
            f"Insights: {insights}\n"
            f"Keywords: {keywords}\n"
            f"Filtered Context: {filtered_context}\n"
            f"Response: {response}\n"
            f"Feedback: {feedback}\n\n"
            "Make the generated response safe in terms of toxic/unsafe words used based on feedback. Only return the "
            "answer to the user query directly. Do not give explanations or information about what this agent is "
            "trying to improve as it is a chatbot and user would only want to see the direct answer to the query."
        )
        response = await openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system",
                 "content": "You are an assistant that makes the generated response safe in terms of toxic/unsafe words used based on feedback."},
                {"role": "user", "content": prompt}
            ],
            seed=26

        )

        end_time = time.time()
        time_taken = end_time - start_time
        tokens_used = get_token_usage(response)

        cost = calculate_cost(input_tokens=tokens_used['input_tokens'], output_tokens=tokens_used['output_tokens'],
                              model=self.model)
        logger.info(
            f"Safety Action -> Cost: {cost} dollars, Time Taken: {time_taken} seconds")

        return response.choices[0].message.content.strip()

    async def make_response_aligned(self,  user_query, query_rewritten, insights, keywords, filtered_context, response, feedback):
        start_time = time.time()

        prompt = (
            f"Original Query: {user_query}\n"
            f"Rewritten Query: {query_rewritten}\n"
            f"Insights: {insights}\n"
            f"Keywords: {keywords}\n"
            f"Filtered Context: {filtered_context}\n"
            f"Response: {response}\n"
            f"Feedback: {feedback}\n\n"
            "Align the response to be more consistent with user expectations. Only return the answer to the user "
            "query directly. Do not give explanations or information about what this agent is trying to improve as it "
            "is a chatbot and user would only want to see the direct answer to the query."
        )
        response = await openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system",
                 "content": "You are an assistant that aligns responses to be more consistent with user expectations."},
                {"role": "user", "content": prompt}
            ],
            seed=26

        )

        end_time = time.time()
        time_taken = end_time - start_time
        tokens_used = get_token_usage(response)

        cost = calculate_cost(input_tokens=tokens_used['input_tokens'], output_tokens=tokens_used['output_tokens'],
                              model=self.model)
        logger.info(
            f"Alignment Action -> Cost: {cost} dollars, Time Taken: {time_taken} seconds")

        return response.choices[0].message.content.strip()
