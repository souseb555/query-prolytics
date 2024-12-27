# QueryLytics

QueryLytics is an intelligent query processing system that combines natural language understanding with advanced retrieval capabilities. It features a multi-agent architecture for handling complex queries, document retrieval, and interactive feedback.

## ✨ Features

- **Intelligent Query Processing**: Advanced natural language understanding using OpenAI's GPT models.
- **Multi-Agent Architecture**: Specialized agents for different tasks:
  - **Main Agent**: Orchestrates query processing.
  - **Retrieval Agent**: Handles document search and retrieval.
  - **Probing Agent**: Manages interactive feedback sessions.
  - **Doc Chat Agent**: Processes document-specific queries.
- **Vector Search**: Efficient document retrieval using ChromaDB.
- **Interactive Feedback**: Dynamic feedback collection and response improvement.
- **API Integration**: FastAPI-based REST API for knowledge base operations.

## 🏰 Project Structure

```plaintext
querylytics/
├── apps/
│   └── knowledge_base/
│       └── app/
│           ├── api/
│           │   └── router.py  # FastAPI routes for KB operations
│           └── models/
├── shared/
│   └── infrastructure/
│       ├── agent/             # Agent framework
│       │   ├── base.py        # Base agent implementation
│       │   ├── main_agent.py  # Main orchestration agent
│       │   └── special/       # Specialized agents
│       ├── embedding_models/  # Embedding implementations
│       ├── language_models/   # LLM implementations
│       ├── tools/             # Utility tools
│       └── vector_store/      # Vector storage implementations
├── main.py                    # Application entry point
└── run.py                     # FastAPI server runner
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- MongoDB
- ChromaDB
- OpenAI API key

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/querylytics.git
   cd querylytics
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv querylytics_env
   source querylytics_env/bin/activate  # Linux/Mac
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:

   ```bash
   export MONGODB_URI=your_mongodb_uri
   export OPENAI_API_KEY=your_openai_api_key
   ```

### Running the Application

1. Start the FastAPI server:

   ```bash
   python run.py
   ```

2. Run the main application:

   ```bash
   python main.py
   ```

## 🔧 Configuration

Example configuration:

```python
config = MainAgentConfig(
    name="MainAgent",
    debug=True,
    llm=OpenAIGPTConfig(
        temperature=0.7
    )
)
```

## 📚 API Documentation

The Knowledge Base API provides the following endpoints:

### Documents

- `GET /kb/documents/{doc_id}`: Retrieve a specific document.
- `DELETE /kb/documents/{doc_id}`: Delete a document.
- `GET /kb/health`: Health check endpoint.

Example usage:

```python
import requests

# Get a document
response = requests.get("http://localhost:8000/kb/documents/123")
document = response.json()

# Health check
health = requests.get("http://localhost:8000/kb/health")
```

## 🔍 Core Components

### Main Agent

The Main Agent orchestrates the entire query processing workflow:
- Coordinates between specialized agents.
- Manages conversation state.
- Handles error recovery.

### Retrieval Agent

Handles document search and retrieval:
- Vector-based search using ChromaDB.
- Relevance scoring.
- Document ranking.

### Probing Agent

Manages interactive feedback sessions:
- Generates follow-up questions.
- Collects user feedback.
- Summarizes findings.

### Doc Chat Agent

Processes document-specific queries:
- Document chunking.
- Context-aware responses.
- Source attribution.

## 🛠️ Development

### Adding New Agents

1. Create a new agent class in `shared/infrastructure/agent/special/`.
2. Inherit from the base `Agent` class.
3. Implement required methods:
   - `__init__`
   - `handle_message`
   - Any specialized methods.

Example:

```python
class CustomAgent(Agent):
    def __init__(self, config: AgentConfig):
        super().__init__(config)

    def handle_message(self, message: str) -> str:
        # Implementation
        pass
```

### Testing

Run tests with:

```bash
python -m pytest tests/
```

## 🧑‍💻 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🖋️ License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for GPT models
- ChromaDB for vector storage
- FastAPI for the web framework
- Langroid for agentic framework

