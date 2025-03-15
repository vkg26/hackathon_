import asyncio
from dotenv import load_dotenv
import json
import time
from constant import *
from colorama import init, Fore, Back, Style

# Initialize colorama
init(autoreset=True)

# Import agent modules
from query_interpretation_agent import QueryInterpretationAgent
from rag_agent import RAGAgent, extract_text_from_pdf
from response_generator_agent import ResponseGeneratorAgent
from feedback_extractor_agent import FeedbackExtractionAgent
from reinforcement_agent import ReinforcementAgent

async def query_interpretation(user_query):
    agent = QueryInterpretationAgent(user_query)
    return await agent.interpret_query()

async def rag_agent(rewritten_query, insights, keywords, contract_text):
    agent = RAGAgent(rewritten_query, insights, keywords, contract_text)
    return await agent.filter_context()

async def response_generator(user_query, rewritten_query, insights, keywords, filtered_context):
    agent = ResponseGeneratorAgent(user_query, rewritten_query, insights, keywords, filtered_context)
    return await agent.generate_response()

async def feedback_extractor(user_query, rewritten_query, insights, keywords, feedback_data_path, filtered_feedback_data_path):
    agent = FeedbackExtractionAgent(user_query, rewritten_query, insights, keywords, feedback_data_path, filtered_feedback_data_path)
    return await agent.extract_feedback()

async def reinforcement_agent(user_query, rewritten_query, insights, keywords, filtered_context, generated_response, specific_query_feedback, general_user_feedback):
    agent = ReinforcementAgent(user_query, rewritten_query, insights, keywords, filtered_context, generated_response, specific_query_feedback, general_user_feedback)

    max_iterations = 2
    iterations = 0
    final_response = ""
    previous_actions = []
    while iterations < max_iterations:
        print(Fore.YELLOW + f"Reinforcement Agent Iteration {iterations + 1}")
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
            print(Fore.RED + "Reinforced response:" + Fore.RESET + f"{modified_response}")
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
        feedback_extractor(user_query, rewritten_query, insights, keywords, feedback_data_path, filtered_feedback_data_path))
    rag_task = asyncio.create_task(rag_agent(rewritten_query, insights, keywords, contract_text))

    # Gather the results of both tasks
    filtered_context, (specific_query_feedback, general_user_feedback) = await asyncio.gather(rag_task, feedback_task)
    print(Fore.YELLOW + "Filtered context extracted using RAG Agent: " + Fore.RESET + f"{filtered_context}\n")  # Debugging statement

    print(Fore.YELLOW + "Feedback extracted using Feedback Agent")  # Debugging statement
    print(Fore.RED + "Specific Query feedback: " +Fore.RESET + f"{specific_query_feedback}\n" + Fore.RED + "General User feedback: " + Fore.RESET + f"{general_user_feedback}\n")  # Debugging statement

    # Step 3: Call Response Generator Agent
    generated_response = await response_generator(user_query, rewritten_query, insights, keywords, filtered_context)
    print(Fore.YELLOW + "Response generated using Response Generator Agent")  # Debugging statement
    print(Fore.RED + "Generated response: " + Fore.RESET + f"{generated_response}\n")  # Debugging statement

    # Step 4: Call Reinforcement Agent
    final_response = await reinforcement_agent(user_query, rewritten_query, insights, keywords, filtered_context, generated_response, specific_query_feedback, general_user_feedback)

    end_time = time.time()
    total_time_taken = end_time - start_time

    print(Fore.RED + "Total Time Taken: " + Fore.RESET + f"{total_time_taken} seconds\n")

    return generated_response, final_response

if __name__ == '__main__':
    user_query = "Can you explain the indemnification clause?"
    # Path to your PDF file
    pdf_path = "../data/Sample_Contract.pdf"
    contract_text = extract_text_from_pdf(pdf_path)

    feedback_data_path = f"../data/feedback_data/gpt_user_feedback - {TEST_CASE}.xlsx"
    filtered_feedback_data_path = f"../data/filtered_feedback_data/gpt_user_feedback - {TEST_CASE}.xlsx"

    # Run the main function
    generated_response, final_response = asyncio.run(main(user_query, contract_text, feedback_data_path, filtered_feedback_data_path))