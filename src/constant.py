COSINE_SIMILAR_FEEDBACK = 5
PROVIDER = "OPENAI"

if PROVIDER == "AZURE":
    EMBEDDING_MODEL = "texte3large" ## "ada002"
    GPT4_OMNI_MODEL = "gpt40"
    GPT35_MODEL = "gpt35"

    """https://portal.azure.com/#@icertis.com/resource/subscriptions/3019693c-fe55-455e-a247-e994cb81e074/overview
    https://portal.azure.com/#@icertis.com/resource/subscriptions/3019693c-fe55-455e-a247-e994cb81e074/resourceGroups/intaihctnopenai/providers/Microsoft.CognitiveServices/accounts/hackthonai/overview
    https://oai.azure.com/portal/4ed85998cad84a7da592f3713815a718/deployment?tenantid=78eff5bb-da38-47f0-a836-294c6d784112"""

    MODEL_AGENT_MAPPING = {
        'query_interpretation_agent': GPT4_OMNI_MODEL,
        'rag_agent': GPT4_OMNI_MODEL,
        'feedback_extractor_agent': GPT4_OMNI_MODEL,
        'response_generator_agent': GPT4_OMNI_MODEL,
        'reinforcement_agent': GPT4_OMNI_MODEL,
    }

    INPUT_TOKEN_LLM_COSTS = {
        'gpt40': 0.000005,  # Example cost per token
        'gpt35': 0.0000005  # Example cost per token
    }

    OUTPUT_TOKEN_LLM_COSTS = {
        'gpt40': 0.000015,  # Example cost per token
        'gpt35': 0.0000015  # Example cost per token
    }

elif PROVIDER == "OPENAI":
    EMBEDDING_MODEL = "text-embedding-3-large"
    GPT4_MODEL = "gpt-4"
    GPT4_OMNI_MODEL = "gpt-4o"
    GPT4_TURBO_MODEL = "gpt-4-turbo"
    GPT35_MODEL = "gpt-3.5-turbo-0125"

    MODEL_AGENT_MAPPING = {
        'query_interpretation_agent': GPT4_OMNI_MODEL,
        'rag_agent': GPT4_OMNI_MODEL,
        'feedback_extractor_agent': GPT4_OMNI_MODEL,
        'response_generator_agent': GPT4_OMNI_MODEL,
        'reinforcement_agent': GPT4_OMNI_MODEL,
    }

    INPUT_TOKEN_LLM_COSTS = {
        'gpt-4': 0.000030,  # Example cost per token
        'gpt-4o': 0.000005,  # Example cost per token
        'gpt-4-turbo': 0.000010,  # Example cost per token
        'gpt-3.5-turbo-0125': 0.0000005  # Example cost per token
    }

    OUTPUT_TOKEN_LLM_COSTS = {
        'gpt-4': 0.000060,  # Example cost per token
        'gpt-4o': 0.000015,  # Example cost per token
        'gpt-4-turbo': 0.000030,  # Example cost per token
        'gpt-3.5-turbo-0125': 0.0000015  # Example cost per token
    }

TEST_CASE = "TC3"


