import asyncio
from flask import Flask, request, jsonify, render_template
from query_interpretation_agent import QueryInterpretationAgent
from rag_agent import RAGAgent, extract_text_from_pdf
from response_generator_agent import ResponseGeneratorAgent
from feedback_extractor_agent import FeedbackExtractionAgent
from reinforcement_agent import ReinforcementAgent
import json
import pandas as pd
import os
import logging
import uuid
import time
from colorama import init, Fore, Back, Style
from utils import *
from flask_cors import CORS
from model_client import openai_client

# Initialize colorama
init(autoreset=True)

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(filename='../app_logs.log', level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@app.route('/')
def index():
    return render_template('index.html')


# Function to get embeddings from OpenAI's Ada model
async def get_embeddings(text):
    response = await openai_client.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding


# Function to save embedding as a JSON file
def save_embedding_as_json(DIR, embedding, embedding_id):
    file_path = os.path.join(DIR, f"{embedding_id}.json")
    with open(file_path, 'w') as f:
        json.dump(embedding, f)


async def query_interpretation(user_query):
    agent = QueryInterpretationAgent(user_query)
    return await agent.interpret_query()

async def rag_agent(rewritten_query, insights, keywords, contract_text):
    agent = RAGAgent(rewritten_query, insights, keywords, contract_text)
    return await agent.filter_context()

async def response_generator(user_query, rewritten_query, insights, keywords, filtered_context):
    agent = ResponseGeneratorAgent(user_query, rewritten_query, insights, keywords, filtered_context)
    return await agent.generate_response()

async def feedback_extractor(user_query, rewritten_query, insights, keywords, feedback_data_path,
                             filtered_feedback_data_path):
    agent = FeedbackExtractionAgent(user_query, rewritten_query, insights, keywords, feedback_data_path,
                                    filtered_feedback_data_path)
    return await agent.extract_feedback()


async def reinforcement_agent(user_query, rewritten_query, insights, keywords, filtered_context, generated_response,
                              specific_query_feedback, general_user_feedback):
    agent = ReinforcementAgent(user_query, rewritten_query, insights, keywords, filtered_context, generated_response,
                               specific_query_feedback, general_user_feedback)

    max_iterations = 2
    iterations = 0
    final_response = ""
    previous_actions = []
    while iterations < max_iterations:
        print(Fore.YELLOW + f" Reinforcement Agent Iteration {iterations + 1}")
        response = await agent.formulate_response(previous_actions)
        tool_calls = response.choices[0].message.tool_calls[0].function

        if tool_calls:
            tool_function_name = tool_calls.name
            tool_function_id = response.choices[0].message.tool_calls[0].id
            arguments = json.loads(tool_calls.arguments)
            arguments["filtered_context"] = agent.filtered_context

            if tool_function_name == "make_response_accurate":
                modified_response = await agent.make_response_accurate(**arguments)
            elif tool_function_name == "make_response_relevant":
                modified_response = await agent.make_response_relevant(**arguments)
            elif tool_function_name == "make_response_clear":
                modified_response = await agent.make_response_clear(**arguments)
            elif tool_function_name == "make_response_verbose":
                modified_response = await agent.make_response_verbose(**arguments)
            elif tool_function_name == "make_response_complete":
                modified_response = await agent.make_response_complete(**arguments)
            elif tool_function_name == "make_response_quality":
                modified_response = await agent.make_response_quality(**arguments)
            elif tool_function_name == "make_response_emotional":
                modified_response = await agent.make_response_emotional(**arguments)
            elif tool_function_name == "make_response_safe":
                modified_response = await agent.make_response_safe(**arguments)
            elif tool_function_name == "make_response_aligned":
                modified_response = await agent.make_response_aligned(**arguments)
            else:
                print(f"Error: function {tool_function_name} does not exist")
                break

            print(Fore.RED + "Agent selected:" + Fore.RESET + f"{tool_function_name}")
            print("Response modified according to user preference\n")

            # Append the result to the messages list
            agent.messages.append({
                "role": "assistant",
                "name": tool_function_name,
                "id": tool_function_id,
                "content": modified_response
            })

            previous_actions.append(tool_function_name)

            final_response = modified_response
        else:
            final_response = response.choices[0].message.content
            break

        iterations += 1

    if iterations == max_iterations:
        print(Fore.YELLOW + "Max iterations reached.\n")

    return final_response


async def main(user_query, contract_text, feedback_data_path, filtered_feedback_data_path):
    start_time = time.time()
    print(Fore.RED + "Received user query: " + Fore.RESET + f"{user_query}\n")  # Debugging statement

    # Step 1: Call Query Interpretation Agent
    rewritten_query, insights, keywords = await query_interpretation(user_query)
    print(Fore.YELLOW + "Query Analysis finished using Query Agent")  # Debugging statement
    print(Fore.RED + "Rewritten query: " + Fore.RESET + f"{rewritten_query}\n" + Fore.RED + "Insights: " + Fore.RESET + f"{insights}\n" + Fore.RED + "Keywords: " + Fore.RESET + f"{keywords}\n")  # Debugging statement

    # Step 2: Start Feedback Extraction Agent and RAG Agent in parallel using asyncio.gather()
    feedback_task = asyncio.create_task(
        feedback_extractor(user_query, rewritten_query, insights, keywords, feedback_data_path,
                           filtered_feedback_data_path))
    rag_task = asyncio.create_task(rag_agent(rewritten_query, insights, keywords, contract_text))

    # Gather the results of both tasks
    filtered_context, (specific_query_feedback, general_user_feedback) = await asyncio.gather(rag_task, feedback_task)

    print(Fore.YELLOW + "Filtered context extracted using RAG Agent\n")  # Debugging statement

    print(Fore.YELLOW + "Feedback extracted using Feedback Agent")  # Debugging statement
    print(Fore.RED + "Specific Query feedback: " +Fore.RESET + f"{specific_query_feedback}\n" + Fore.RED + "General User feedback: " + Fore.RESET + f"{general_user_feedback}\n")  # Debugging statement

    # Step 3: Call Response Generator Agent
    generated_response = await response_generator(user_query, rewritten_query, insights, keywords, filtered_context)
    print(Fore.YELLOW + "Response generated using Response Generator Agent")  # Debugging statement
    print(Fore.RED + "Generated response: " + Fore.RESET + f"{generated_response}\n")  # Debugging statement

    # Step 4: Call Reinforcement Agent
    final_response = await reinforcement_agent(user_query, rewritten_query, insights, keywords, filtered_context,
                                               generated_response, specific_query_feedback, general_user_feedback)

    end_time = time.time()
    total_time_taken = end_time - start_time

    logger.info(f"Total Time Taken: {total_time_taken} seconds")

    return generated_response, final_response


@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_query = data.get('query')

    # Path to your PDF file
    pdf_path = "../data/Sample_Contract.pdf"
    contract_text = extract_text_from_pdf(pdf_path)

    feedback_data_path = f"../data/feedback_data/gpt_user_feedback - {TEST_CASE}.xlsx"
    filtered_feedback_data_path = f"../data/filtered_feedback_data/gpt_user_feedback - {TEST_CASE}.xlsx"

    # Run the main function
    generated_response, final_response = asyncio.run(
        main(user_query, contract_text, feedback_data_path, filtered_feedback_data_path))

    return jsonify({'original_response': generated_response,
                    'reinforced_response': final_response})


@app.route('/feedback', methods=['POST'])
async def feedback():
    data = request.json
    user_query = data.get('user_query')
    selected_response = data.get('response')
    feedback = data.get('feedback')
    accuracy = data.get('accuracy')
    relevancy = data.get('relevancy')
    completeness = data.get('completeness')
    verbosity = data.get('verbosity')
    emotional = data.get('emotional')
    safety = data.get('safety')
    quality = data.get('quality')
    free_text = data.get('free_text')

    query_embedding_dir = f'../data/feedback_data/query_embedding_json - {TEST_CASE}/'
    response_embedding_dir = f'../data/feedback_data/response_embedding_json - {TEST_CASE}/'

    # Generate unique IDs for the embeddings
    user_query_id = str(uuid.uuid4())
    response_id = str(uuid.uuid4())

    # Get embeddings
    user_query_embedding = await get_embeddings(user_query)
    response_embedding = await get_embeddings(selected_response)

    # Save embeddings as JSON files
    save_embedding_as_json(query_embedding_dir, user_query_embedding, user_query_id)
    save_embedding_as_json(response_embedding_dir, response_embedding, response_id)

    # Prepare feedback data
    feedback_data = {
        'user_query': [user_query_id],
        'response': [response_id],
        'feedback': [feedback],
        'accuracy': [accuracy],
        'relevancy': [relevancy],
        'completeness': [completeness],
        'verbosity': [verbosity],
        'emotional': [emotional],
        'safety': [safety],
        'quality': [quality],
        'free_text': [free_text]
    }

    df = pd.DataFrame(feedback_data)

    feedback_file = f'../data/real_time_feedback_data/gpt_user_feedback_{TEST_CASE}.xlsx'
    current_feedback_data = f'../data/feedback_data/gpt_user_feedback_{TEST_CASE}.xlsx'

    try:
        existing_df = pd.read_excel(current_feedback_data)
        df = pd.concat([existing_df, df], ignore_index=True)
    except FileNotFoundError:
        pass

    df.to_excel(feedback_file, index=False)

    return jsonify({"message": "Feedback saved successfully"}), 200


if __name__ == '__main__':
    app.run(debug=True)
