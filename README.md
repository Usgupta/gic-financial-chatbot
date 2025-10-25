# GIC Financial Chatbot

A sophisticated AI-powered chatbot for analyzing financial documents with advanced PDF processing, figure extraction, and intelligent question-answering capabilities. Built with Python, Gradio, OpenAI GPT-4, and Qdrant vector database.

## 🚀 Features

### Core Capabilities
- **PDF Document Processing**: Upload and analyze financial PDFs with intelligent text extraction
- **Figure & Chart Detection**: Automatically detect and extract figures, charts, diagrams, and graphs using DocLayout-YOLO
- **Multimodal Analysis**: Analyze both text content and visual elements using OpenAI Vision API
- **Intelligent Chunking**: Use LangExtract for semantic understanding and smart document chunking
- **Vector Search**: Store and retrieve document chunks using Qdrant vector database
- **Interactive Chat Interface**: Clean, user-friendly Gradio interface for document interaction

### Advanced Features
- **Multi-Query Search**: Generate query variations for comprehensive information retrieval
- **Citation Tracking**: Automatic source citations with page references
- **Figure Ranking**: Semantic similarity ranking for relevant figures and charts
- **Streaming Responses**: Real-time response streaming for better user experience
- **Error Handling**: Robust error handling with detailed logging and debugging

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PDF Upload    │───▶│  Text Extraction │───▶│  Smart Chunking │
│                 │    │  (PyPDF)         │    │  (LangExtract)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Figure Detection│───▶│ Vision Analysis  │───▶│ Vector Storage  │
│ (DocLayout-YOLO)│    │ (OpenAI Vision)  │    │ (Qdrant)        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │───▶│ Multi-Query      │───▶│ Response        │
│                 │    │ Search           │    │ Generation      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📋 Prerequisites

- **Python 3.11+**
- **OpenAI API Key** (for GPT-4 and Vision API)
- **Qdrant Cloud Account** (or local Qdrant instance)
- **poppler-utils** (for PDF to image conversion)

### System Dependencies

#### macOS
```bash
brew install poppler
```

#### Ubuntu/Debian
```bash
sudo apt-get install poppler-utils
```

#### Windows
Download poppler from [poppler-windows](https://github.com/oschwartz10612/poppler-windows) and add to PATH.

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd gic-financial-chatbot
```

### 2. Install Dependencies

The project uses `uv` for dependency management. Install dependencies with:

```bash
# Install uv if you don't have it
pip install uv

# Install project dependencies
uv sync
```

Alternatively, you can use pip:
```bash
pip install -r requirements.txt
```

### 3. Environment Setup

Create a `.env` file in the project root:

```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Qdrant Configuration
QDRANT_URL=your_qdrant_cloud_url
QDRANT_API_KEY=your_qdrant_api_key
```

### 4. Verify Installation

Test the installation by running the figure extraction test:

```bash
python test_figure_extraction.py --help
```

## 🚀 Quick Start

### 1. Start the Application
```bash
python main.py
```

The application will start and provide a local URL (typically `http://127.0.0.1:7860`).

### 2. Upload Documents
1. Open the web interface
2. Navigate to the "Upload PDFs" tab
3. Upload your financial PDF documents
4. Wait for processing to complete

### 3. Ask Questions
1. Switch to the "Ask Questions" tab
2. Type your questions about the uploaded documents
3. View responses with citations and related figures

## 📖 Usage Examples

### Basic Queries
```
"What is the company's revenue for Q3?"
"Show me the financial performance charts"
"What are the key risk factors mentioned?"
"Find information about market trends"
```

### Figure-Specific Queries
```
"Show me all charts and graphs"
"Display the revenue visualization"
"Find diagrams related to business strategy"
"What figures show the financial projections?"
```

### Advanced Queries
```
"Compare the revenue trends across different quarters"
"What do the charts say about market growth?"
"Summarize the key financial metrics from the figures"
```

## 🔧 Configuration

### Model Settings
The application uses several configurable parameters in `main.py`:

```python
# Figure extraction settings
CONFIDENCE_THRESHOLD = 0.25  # Detection confidence threshold
IMAGE_SIZE = 1024           # Image processing size
FIGURE_CLASSES = ['figure', 'picture', 'chart', 'diagram', 'graph', 'plot']

# Processing limits
max_pages_to_process = 5    # Limit figure extraction to first 5 pages
max_context_length = 8000   # Maximum context length for responses
```

### Qdrant Configuration
```python
COLLECTION_NAME = "pdf_documents"  # Vector collection name
VectorParams(size=1536, distance=Distance.COSINE)  # Embedding settings
```

### Check Collection Status
Use the "Check Collection" button in the web interface to verify document storage.

## 📁 Project Structure

```
gic-financial-chatbot/
├── main.py                          # Main application file
├── test_figure_extraction.py        # Figure extraction test script
├── pyproject.toml                   # Project dependencies
├── uv.lock                          # Dependency lock file
├── README.md                        # This file
├── uploaded_pdfs/                   # Directory for uploaded PDFs
├── extracted_figures/               # Directory for extracted figures
└── .env                             # Environment variables (create this)
```

## 🔍 Key Components

### Document Processing Pipeline
1. **PDF Upload**: Files are copied to `uploaded_pdfs/` directory
2. **Text Extraction**: PyPDF extracts text page by page
3. **Figure Detection**: DocLayout-YOLO identifies visual elements
4. **Vision Analysis**: OpenAI Vision API analyzes extracted figures
5. **Smart Chunking**: LangExtract creates semantic chunks
6. **Vector Storage**: Embeddings stored in Qdrant database

### Query Processing Pipeline
1. **Query Expansion**: Generate multiple query variations
2. **Vector Search**: Multi-query search in Qdrant
3. **Result Ranking**: Prioritize figures vs text based on query type
4. **Context Building**: Assemble relevant chunks with citations
5. **Response Generation**: GPT-4 generates responses with citations
6. **Figure Display**: Show relevant figures in gallery


## 🔒 Security Considerations

- **API Keys**: Never commit API keys to version control
- **File Uploads**: Validate PDF files before processing
- **Network Security**: Use HTTPS for production deployments
- **Data Privacy**: Consider data retention policies for uploaded documents

## 🚀 Deployment

### Local Development
```bash
python main.py
```


## 🙏 Acknowledgments

- **OpenAI**: For GPT-4 and Vision API capabilities
- **Qdrant**: For vector database functionality
- **DocLayout-YOLO**: For document layout analysis
- **LangExtract**: For intelligent text extraction
- **Gradio**: For the user interface framework


**Note**: This application requires active internet connectivity for model downloads and API calls. Ensure stable internet connection for optimal performance.