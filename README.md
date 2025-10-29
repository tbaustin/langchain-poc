# Sourdough Recipe Assistant

An intelligent conversational AI assistant built with LangChain and LangGraph that provides personalized sourdough recipes based on user preferences and learns from feedback over time.

## Features

- 🍞 **Personalized Sourdough Recipes**: Get customized sourdough recipes with measurements in grams
- 🧠 **Long-term Memory**: Stores user feedback in a vector database to personalize future recommendations
- 💬 **Conversational Interface**: Interactive chat-based recipe requests and feedback collection
- 📝 **Recipe Constraints**: Automatically converts butter to margarine, honey to sugar, and optimizes for soft texture
- 🔄 **Persistent Conversations**: Maintains conversation history across sessions using checkpointing
- 🎯 **Semantic Search**: Retrieves relevant past feedback using FAISS vector similarity search

## Architecture

### Technology Stack

- **LLM**: Google Gemini (via `langchain-google-genai`)
- **Framework**: LangChain & LangGraph for stateful conversation workflows
- **Vector Store**: FAISS with Google's embedding model for semantic search
- **Memory**: SQLite for production, in-memory for development
- **Package Manager**: uv

### Core Components

#### 1. **State Management** (`src/state.py`)
Defines the conversation state using Pydantic models:
- `messages`: Conversation history with automatic message merging
- `retrieved_feedback`: Context from past user feedback

#### 2. **Graph Workflow** (`src/graph.py`)
Orchestrates the conversation flow using LangGraph:
```
get_user_input → check_end → call_model → ask_for_feedback → check_feedback
                    ↓                           ↓                     ↓
                   END                   provide_feedback ←─────────┘
```

#### 3. **Nodes** (`src/nodes.py`)
Individual processing units in the graph:
- `get_user_input`: Prompts user for recipe requests
- `call_model`: Generates AI response with context
- `retrieve_feedback`: Searches vector store for relevant past feedback (k=1)
- `save_feedback`: Stores feedback as documents with metadata (timestamp, recipe context)
- `ask_user_for_feedback`: Prompts for feedback collection
- `provide_feedback`: Collects and processes user feedback
- `check_end_connection`: Determines if conversation should end
- `check_feedback`: Routes to feedback or continuation flow

#### 4. **LLM Chain** (`src/chains.py`)
Configures the Google Gemini model with:
- Temperature: 0.7 (balanced creativity/consistency)
- Custom system prompt with recipe constraints
- Context injection for personalization

#### 5. **Vector Store** (`src/vector_store.py`)
FAISS-based semantic search:
- Uses Google's `embedding-001` model (768 dimensions)
- Stores feedback as `Document` objects with rich metadata
- In-memory document store for fast retrieval

## Setup

### Prerequisites

- Python 3.13+
- uv package manager
- Google Gemini API key

### Installation

1. **Clone the repository**
```bash
git clone <repo-url>
cd langchain-poc
```

2. **Install dependencies**
```bash
uv sync
```

3. **Install development tools (optional)**
```bash
uv sync --group dev
```

4. **Set up environment variables**

Create a `.env` file in the project root:
```env
GOOGLE_GENAI_API_KEY=your_api_key_here
GOOGLE_GENAI_MODEL=gemini-2.5-flash-lite
GOOGLE_GENAI_EMBEDDING_MODEL=models/embedding-001
ENV=development  # or "production" for SQLite persistence
```

## Usage

### Running the Application

```bash
uv run python main.py
```

### Example Conversation

```
Please provide your sourdough recipe request: I want a simple sandwich bread
Node 'call_model' executed:
  Content: Here's a sourdough recipe: Mix 500g bread flour, 350g water, 100g active starter...

Do you want to provide feedback on the recipe? (yes/no): yes
Please provide your feedback on the recipe: Too salty, reduce the salt
```

### Terminating the Conversation

Type any of these commands when prompted for input:
- `quit`
- `exit`
- `end`
- `no`

## How It Works

### Feedback Learning System

1. **User provides feedback** on a recipe
2. **Recipe context is extracted** from the previous AI response (first 200 chars)
3. **Feedback is stored** as a vector embedding with metadata:
   ```python
   Document(
       page_content="Too salty, reduce the salt",
       metadata={
           "type": "user_feedback",
           "timestamp": "2025-10-29T14:30:00",
           "recipe_context": "Here's a sourdough recipe: Mix..."
       }
   )
   ```
4. **Future requests** trigger semantic search (k=1) to find relevant feedback
5. **Retrieved feedback** is injected into the LLM prompt for personalization

### Temperature Settings

The LLM uses `temperature=0.7` for balanced behavior:
- **Low (0.0-0.3)**: Deterministic, factual responses
- **Medium (0.4-0.8)**: Balanced creativity and consistency ✅ **(current)**
- **High (0.9-2.0)**: Very creative but less reliable

### Vector Search Parameters

- **k=1**: Returns the single most relevant feedback
- Increases focus and reduces noise
- Suitable for targeted personalization

## Development

### Project Structure

```
langchain-poc/
├── main.py                 # Application entry point
├── pyproject.toml          # Dependencies and project metadata
├── uv.lock                 # Locked dependency versions
├── .env                    # Environment variables (gitignored)
├── memory.sqlite           # SQLite checkpoint storage (production)
└── src/
    ├── graph.py            # LangGraph workflow definition
    ├── nodes.py            # Graph node implementations
    ├── state.py            # State model definitions
    ├── chains.py           # LLM chain configuration
    └── vector_store.py     # FAISS vector store setup
```

### Code Quality

**Ruff** is configured for linting and formatting:

```bash
# Check for issues
uv run ruff check .

# Auto-fix issues
uv run ruff check --fix .

# Format code
uv run ruff format .
```

VS Code is configured for format-on-save with Ruff.

### Logging

The application uses Python's standard logging with INFO level:
- Graph initialization and node execution
- Feedback retrieval and storage operations
- Recipe context extraction (first 30 chars)
- Routing decisions

## Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_GENAI_API_KEY` | *required* | Google Gemini API key |
| `GOOGLE_GENAI_MODEL` | `gemini-2.5-flash-lite` | LLM model to use |
| `GOOGLE_GENAI_EMBEDDING_MODEL` | `models/embedding-001` | Embedding model (768-dim) |
| `ENV` | `development` | `development` (in-memory) or `production` (SQLite) |

### Recipe Constraints (Prompt Template)

The AI follows these constraints:
- ✅ Only sourdough recipes
- ✅ Measurements in grams
- ✅ Butter → Margarine conversion
- ✅ Honey → Granulated sugar conversion
- ✅ Soft texture preference (not chewy)
- ✅ Max 1000 characters per recipe
- ✅ Always starts with "Here's a sourdough recipe:"

## Vector Database Details

### FAISS Index

- **Type**: `IndexFlatL2` (L2 distance, exhaustive search)
- **Dimension**: 768 (determined by embedding model)
- **Storage**: In-memory (fast but non-persistent)

### Document Store

- **Type**: `InMemoryDocstore`
- **Purpose**: Stores original document content and metadata
- **Mapping**: `index_to_docstore_id` links FAISS positions to document IDs

### Alternatives

For production use with persistence, consider:
- **Qdrant**: High-performance Rust-based vector DB
- **Weaviate**: Feature-rich with built-in embeddings
- **Chroma**: Lightweight with persistence
- **Milvus**: Enterprise-scale vector database

## Troubleshooting

### Common Issues

**Issue**: `GOOGLE_GENAI_API_KEY not set`
**Solution**: Create a `.env` file with your API key

**Issue**: Vector store returns no feedback
**Solution**: Provide feedback on at least one recipe first to populate the store

**Issue**: Recipe context not extracted
**Solution**: Ensure AI responses contain "recipe" keyword (enforced by prompt)

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]