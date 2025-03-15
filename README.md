# [USER First] Unified Solution for Enhanced Response & Feedback

## Overview
[USER First] is an intelligent multi-agent AI framework designed to enhance user confidence and satisfaction in AI-driven contract management copilots. By leveraging reinforcement learning principles, the system dynamically refines responses based on user feedback over time, ensuring improved query alignment and response personalization. This was developed as part of a hackathon submission organized by Icertis Solutions Pvt Ltd. 

## Features
- **Multi-Agent Architecture**: Consists of specialized agents for query interpretation, retrieval-augmented generation (RAG), response generation, feedback extraction, and reinforcement learning.
- **Reinforcement-Based Response Optimization**: Iteratively improves responses using past feedback.
- **OpenAI GPT-4o Integration**: Utilizes the latest OpenAI models for embeddings and response generation.
- **Flask-Based API**: Provides endpoints for querying and submitting feedback.
- **Asynchronous Processing**: Uses `asyncio` for efficient parallel execution of agents.

## Architecture
The framework is built around the following AI agents:
1. **Query Interpretation Agent**: Analyzes user input and refines the query for better comprehension.
2. **RAG Agent**: Extracts relevant contract information to generate context-aware responses.
3. **Response Generator Agent**: Constructs AI-driven answers based on the interpreted query and context.
4. **Feedback Extraction Agent**: Analyzes past user feedback to refine response generation.
5. **Reinforcement Agent**: Implements feedback-driven improvements using structured rules and AI-driven optimizations.
6. **Orchestrator**: Manages workflow execution between agents to ensure efficient response delivery.

## Installation
### Prerequisites
- Python 3.8+
- OpenAI API key (for GPT-4o usage)

### Steps
1. Clone the repository:
   ```sh
   git clone <repo_url>
   cd <repo_name>
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   - Create a `.env` file with the following variables:
     ```env
     OPENAI_API_KEY=your_api_key
     ```
4. Run the application:
   ```sh
   python app.py
   ```

## API Endpoints
### 1. Query Processing
**Endpoint:** `/chat`
**Method:** `POST`
**Request Body:**
```json
{
  "query": "What are the indemnification clauses in this contract?"
}
```
**Response:**
```json
{
  "original_response": "[Initial AI-generated response]",
  "reinforced_response": "[Improved response based on feedback]"
}
```

### 2. Feedback Submission
**Endpoint:** `/feedback`
**Method:** `POST`
**Request Body:**
```json
{
  "user_query": "What are the indemnification clauses?",
  "response": "[Generated response]",
  "feedback": "Needs more clarity",
  "accuracy": 7,
  "relevancy": 8,
  "completeness": 6,
  "verbosity": 5,
  "emotional": 4,
  "safety": 10,
  "quality": 7,
  "free_text": "Make it more detailed."
}
```
**Response:**
```json
{
  "message": "Feedback saved successfully"
}
```

## File Structure
```
├── app.py                     # Main application entry point (Flask API)
├── orchestrator.py             # Manages the execution of AI agents
├── query_interpretation_agent.py  # Analyzes and refines user queries
├── rag_agent.py                # Retrieves context from contracts
├── response_generator_agent.py  # Generates responses based on context
├── feedback_extractor_agent.py  # Extracts relevant feedback
├── reinforcement_agent.py       # Improves responses iteratively
├── model_client.py             # Handles OpenAI API calls
├── utils.py                    # Utility functions (cost calculation, logging, etc.)
├── constant.py                 # Configuration constants
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## Future Improvements
- Enhance personalization by leveraging user-specific preferences.
- Implement real-time fine-tuning with continuous learning.
- Expand support for additional contract types beyond MSAs, NDAs, and SOWs.

## Contact
For any queries or contributions, feel free to reach out on vinayakgoyaliitd26@gmail.com

