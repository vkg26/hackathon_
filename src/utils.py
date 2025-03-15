from constant import *


def calculate_cost(input_tokens, output_tokens, model):
    input_cost = input_tokens * INPUT_TOKEN_LLM_COSTS[model]
    output_cost = output_tokens * OUTPUT_TOKEN_LLM_COSTS[model]
    return input_cost + output_cost


def get_token_usage(response):
    # This function should parse the response to get the input and output tokens
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    total_tokens = input_tokens + output_tokens
    return {'input_tokens': input_tokens, 'output_tokens': output_tokens, 'total_tokens': total_tokens}
