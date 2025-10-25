import os
import gradio as gr
import openai
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from pypdf import PdfReader
import uuid
from pathlib import Path
from dotenv import load_dotenv
import json
import langextract as lx
import cv2
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
from doclayout_yolo import YOLOv10
import torch
import requests
import base64
import urllib3
import re

load_dotenv()

# Create uploads directory
UPLOAD_DIR = Path("uploaded_pdfs")
UPLOAD_DIR.mkdir(exist_ok=True)

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Qdrant client with SSL verification disabled
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    https=True,
    verify=False,
    grpc_port=None,
    prefer_grpc=False
)

# Collection name
COLLECTION_NAME = "pdf_documents"

# Create collection if it doesn't exist
try:
    qdrant.get_collection(COLLECTION_NAME)
except Exception:
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )

# Figure extraction configuration
EXTRACTED_FIGURES_DIR = Path("extracted_figures")
EXTRACTED_FIGURES_DIR.mkdir(exist_ok=True)
MODEL_NAME = "doclayout_yolo_docstructbench_imgsz1024.pt"
CONFIDENCE_THRESHOLD = 0.25
IMAGE_SIZE = 1024
FIGURE_CLASSES = ['figure', 'picture', 'chart', 'diagram', 'graph', 'plot']

# Lazy load YOLO model
_yolo_model = None

def extract_text_by_page(pdf_path):
    """Extract text from PDF file page by page."""
    reader = PdfReader(pdf_path)
    pages_data = []
    
    for page_num, page in enumerate(reader.pages, 1):
        page_text = page.extract_text()
        if page_text.strip():  # Only include pages with content
            pages_data.append({
                'page_number': page_num,
                'text': page_text.strip(),
                'text_length': len(page_text.strip())
            })
    
    print(f"🔍 DEBUG: PDF extraction - {len(pages_data)} pages extracted")
    for page_data in pages_data[:3]:  # Show first 3 pages
        print(f"🔍 DEBUG: Page {page_data['page_number']}: {page_data['text_length']} chars")
    
    return pages_data

def download_model():
    """Download the DocLayout-YOLO model if it doesn't exist."""
    model_path = Path(MODEL_NAME)
    if model_path.exists():
        print(f"Model already exists: {MODEL_NAME}")
        return str(model_path)
    
    print("Downloading DocLayout-YOLO model...")
    model_url = "https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench/resolve/main/doclayout_yolo_docstructbench_imgsz1024.pt"
    
    try:
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Model downloaded successfully: {MODEL_NAME}")
        return str(model_path)
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("Make sure you have internet connection for model download")
        return None

def get_yolo_model():
    """Get YOLO model (lazy loading)."""
    global _yolo_model
    if _yolo_model is None:
        print("🖼️ DEBUG: Loading DocLayout-YOLO model...")
        try:
            model_path = download_model()
            if model_path:
                print(f"🖼️ DEBUG: Model path found: {model_path}")
                _yolo_model = YOLOv10(model_path)
                print(f"🖼️ DEBUG: Model loaded successfully: {MODEL_NAME}")
                print(f"🖼️ DEBUG: Model device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
            else:
                print("🖼️ DEBUG: Failed to download model")
                return None
        except Exception as e:
            print(f"🖼️ DEBUG: Error loading model: {e}")
            return None
    else:
        print("🖼️ DEBUG: Using cached YOLO model")
    return _yolo_model

def convert_pdf_to_images(pdf_path: Path) -> list:
    """Convert PDF pages to PIL Images."""
    print(f"🖼️ DEBUG: Converting PDF to images: {pdf_path.name}")
    print(f"🖼️ DEBUG: PDF path exists: {pdf_path.exists()}")
    print(f"🖼️ DEBUG: PDF size: {pdf_path.stat().st_size / 1024 / 1024:.2f} MB")
    try:
        images = convert_from_path(pdf_path, dpi=200)
        print(f"🖼️ DEBUG: Converted {len(images)} pages to images")
        for i, img in enumerate(images[:3]):  # Show first 3 images info
            print(f"🖼️ DEBUG: Page {i+1} image size: {img.size}")
        return images
    except Exception as e:
        print(f"🖼️ DEBUG: Error converting PDF: {e}")
        return []

def detect_figures(model, image: Image.Image) -> list:
    """Detect figures in a single page image."""
    print(f"🖼️ DEBUG: Detecting figures in image size: {image.size}")
    # Convert PIL to numpy array for YOLO
    image_np = np.array(image)
    print(f"🖼️ DEBUG: Image array shape: {image_np.shape}")
    
    # Run detection
    print(f"🖼️ DEBUG: Running YOLO prediction with imgsz={IMAGE_SIZE}, conf={CONFIDENCE_THRESHOLD}")
    results = model.predict(
        image_np, 
        imgsz=IMAGE_SIZE, 
        conf=CONFIDENCE_THRESHOLD,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        verbose=False
    )
    
    print(f"🖼️ DEBUG: YOLO prediction completed, results: {len(results) if results else 0}")
    
    detections = []
    if results and len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        print(f"🖼️ DEBUG: Found {len(boxes)} total detections")
        
        for i, box in enumerate(boxes):
            # Get class name
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])
            
            print(f"🖼️ DEBUG: Detection {i+1}: {class_name} (confidence: {confidence:.3f})")
            
            # Check if it's a figure-related class
            if class_name.lower() in FIGURE_CLASSES:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                detections.append({
                    'class_name': class_name,
                    'confidence': confidence,
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'detection_id': i
                })
                print(f"🖼️ DEBUG: ✓ Added figure detection: {class_name} at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
            else:
                print(f"🖼️ DEBUG: ✗ Skipped non-figure detection: {class_name}")
    else:
        print("🖼️ DEBUG: No detections found")
    
    print(f"🖼️ DEBUG: Returning {len(detections)} figure detections")
    return detections

def extract_and_save_figures(image: Image.Image, detections: list, 
                           page_num: int, pdf_name: str) -> list:
    """Extract and save detected figures."""
    print(f"🖼️ DEBUG: Extracting {len(detections)} figures from page {page_num + 1}")
    saved_figures = []
    
    for idx, detection in enumerate(detections):
        x1, y1, x2, y2 = detection['bbox']
        print(f"🖼️ DEBUG: Processing figure {idx + 1}: {detection['class_name']} at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
        
        # Crop the figure from the image
        figure_crop = image.crop((x1, y1, x2, y2))
        print(f"🖼️ DEBUG: Cropped figure size: {figure_crop.size}")
        
        # Generate filename
        figure_filename = f"{pdf_name}_page{page_num+1}_figure{idx+1}_{detection['class_name']}.png"
        figure_path = EXTRACTED_FIGURES_DIR / figure_filename
        print(f"🖼️ DEBUG: Saving figure to: {figure_path}")
        
        # Save the figure
        figure_crop.save(figure_path, "PNG")
        print(f"🖼️ DEBUG: ✓ Figure saved successfully")
        
        # Store metadata
        figure_metadata = {
            'filename': figure_filename,
            'page_number': page_num + 1,
            'class_name': detection['class_name'],
            'confidence': detection['confidence'],
            'bbox': detection['bbox'],
            'image_size': figure_crop.size,
            'pdf_name': pdf_name,
            'figure_path': str(figure_path)
        }
        
        saved_figures.append(figure_metadata)
        print(f"🖼️ DEBUG: ✓ Saved figure: {figure_filename} (confidence: {detection['confidence']:.3f})")
    
    print(f"🖼️ DEBUG: Extracted {len(saved_figures)} figures from page {page_num + 1}")
    return saved_figures

def analyze_figure_with_vision_api(image_path):
    """Analyze figure using OpenAI Vision API."""
    print(f"🖼️ DEBUG: Analyzing figure with Vision API: {image_path}")
    print(f"🖼️ DEBUG: Image file exists: {Path(image_path).exists()}")
    print(f"🖼️ DEBUG: Image file size: {Path(image_path).stat().st_size / 1024:.2f} KB")
    
    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        print(f"🖼️ DEBUG: Base64 encoded image length: {len(base64_image)} chars")
        print(f"🖼️ DEBUG: Sending request to OpenAI Vision API...")
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this figure from a financial document. Describe what it shows, key data points, trends, and insights. Be specific about numbers, labels, and visual elements."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]
            }],
            max_tokens=500
        )
        
        description = response.choices[0].message.content
        print(f"🖼️ DEBUG: ✓ Vision API response received ({len(description)} chars)")
        print(f"🖼️ DEBUG: Description preview: {description[:100]}...")
        return description
    except Exception as e:
        print(f"🖼️ DEBUG: ✗ Error analyzing figure with Vision API: {e}")
        return "Unable to analyze this figure."

def is_figure_query(query):
    """Detect if user is asking for a figure/chart/diagram."""
    figure_keywords = ['figure', 'chart', 'diagram', 'graph', 'image', 
                       'picture', 'visualization', 'plot', 'show me']
    is_figure = any(keyword in query.lower() for keyword in figure_keywords)
    print(f"🖼️ DEBUG: Query '{query}' -> is_figure_query: {is_figure}")
    return is_figure

def rank_figures_by_relevance(figures_found, query):
    """Rank figures by semantic similarity to the user's query using embeddings."""
    print(f"🔍 DEBUG: Ranking {len(figures_found)} figures by semantic similarity to query: '{query}'")
    
    if not figures_found:
        return []
    
    try:
        # Get embedding for the user query
        print(f"🔍 DEBUG: Creating embedding for query: '{query}'")
        query_embedding = get_embedding(query)
        
        if query_embedding == [0.0] * 1536:
            print("🔍 DEBUG: ✗ Failed to create query embedding, falling back to keyword matching")
            return rank_figures_by_keywords(figures_found, query)
        
        print(f"🔍 DEBUG: ✓ Query embedding created successfully")
        
        # Calculate similarity scores for each figure
        scored_figures = []
        
        for fig in figures_found:
            description = fig.get('description', '')
            if not description:
                print(f"🔍 DEBUG: Figure '{fig['filename']}' has no description, skipping")
                continue
            
            # Get embedding for figure description
            print(f"🔍 DEBUG: Creating embedding for figure description: '{description[:100]}...'")
            fig_embedding = get_embedding(description)
            
            if fig_embedding == [0.0] * 1536:
                print(f"🔍 DEBUG: ✗ Failed to create embedding for figure '{fig['filename']}', skipping")
                continue
            
            # Calculate cosine similarity
            similarity_score = calculate_cosine_similarity(query_embedding, fig_embedding)
            print(f"🔍 DEBUG: Figure '{fig['filename']}' similarity score: {similarity_score:.4f}")
            
            scored_figures.append((similarity_score, fig))
        
        # Sort by similarity score (highest first) and take top 3
        scored_figures.sort(key=lambda x: x[0], reverse=True)
        top_figures = [fig for score, fig in scored_figures[:3]]
        
        print(f"🔍 DEBUG: Selected top {len(top_figures)} figures by semantic similarity:")
        for i, fig in enumerate(top_figures):
            print(f"🔍 DEBUG:   {i+1}. {fig['filename']} (similarity: {scored_figures[i][0]:.4f})")
        
        return top_figures
        
    except Exception as e:
        print(f"🔍 DEBUG: ✗ Error in semantic ranking: {e}")
        print(f"🔍 DEBUG: Falling back to keyword-based ranking")
        return rank_figures_by_keywords(figures_found, query)

def calculate_cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    import numpy as np
    
    # Convert to numpy arrays
    a = np.array(vec1)
    b = np.array(vec2)
    
    # Calculate cosine similarity
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    similarity = dot_product / (norm_a * norm_b)
    return float(similarity)

def rank_figures_by_keywords(figures_found, query):
    """Fallback keyword-based ranking method."""
    print(f"🔍 DEBUG: Using keyword-based ranking as fallback")
    
    query_lower = query.lower()
    query_words = set(query_lower.split())
    
    scored_figures = []
    
    for fig in figures_found:
        score = 0
        description = fig.get('description', '').lower()
        filename = fig.get('filename', '').lower()
        
        # Basic keyword matching
        description_words = set(description.split())
        common_words = query_words.intersection(description_words)
        score += len(common_words) * 2
        
        filename_words = set(filename.split('_'))
        filename_matches = query_words.intersection(filename_words)
        score += len(filename_matches) * 1.5
        
        # Add base score for having a valid path
        if fig.get('path') and Path(fig['path']).exists():
            score += 1
        
        scored_figures.append((score, fig))
        print(f"🔍 DEBUG: Figure '{fig['filename']}' keyword score: {score}")
    
    # Sort by score and take top 3
    scored_figures.sort(key=lambda x: x[0], reverse=True)
    top_figures = [fig for score, fig in scored_figures[:3]]
    
    print(f"🔍 DEBUG: Selected top {len(top_figures)} figures by keywords:")
    for i, fig in enumerate(top_figures):
        print(f"🔍 DEBUG:   {i+1}. {fig['filename']} (score: {scored_figures[i][0]})")
    
    return top_figures
    
def smart_chunk_with_langextract_page_by_page(pages_data, pdf_name):
    """Use LangExtract to create intelligent chunks with semantic understanding, processing each page individually."""
    print(f"\n🔍 DEBUG: Starting LangExtract page-by-page processing for {pdf_name}")
    print(f"📄 DEBUG: Processing {len(pages_data)} pages individually")
    
    all_chunks = []
    
    for page_data in pages_data:
        page_number = page_data['page_number']
        page_text = page_data['text']
        
        print(f"\n📄 DEBUG: Processing Page {page_number} ({len(page_text)} chars)")
        print(f"📄 DEBUG: Page {page_number} preview: {page_text[:200]}...")
        
        try:
            # Define chunking prompt for LangExtract - focused on financial document extraction
            chunking_prompt = f"""
            Extract key information from this financial document page. Identify and extract meaningful chunks of information.
            Focus on extracting:
            1. Financial metrics (revenue, profit, costs, investments)
            2. Business operations and strategies
            3. Key announcements and developments
            4. Risk factors and forward-looking statements
            5. Company background and history
            
            Each extraction should be a coherent piece of information that can stand alone.
            This is page {page_number} of the document.
            """
            
            # Define examples for LangExtract using proper ExampleData objects
            examples = [
                lx.data.ExampleData(
                    text="DoorDash reported Q3 revenue of $2.2 billion, up 27% year-over-year. The company's marketplace revenue grew to $1.8 billion, driven by increased order volume and higher average order values.",
                    extractions=[
                        lx.data.Extraction(
                            extraction_class="financial",
                            extraction_text="DoorDash reported Q3 revenue of $2.2 billion, up 27% year-over-year.",
                            attributes={"metric": "revenue", "period": "Q3", "growth": "27%"}
                        ),
                        lx.data.Extraction(
                            extraction_class="financial", 
                            extraction_text="The company's marketplace revenue grew to $1.8 billion, driven by increased order volume and higher average order values.",
                            attributes={"metric": "marketplace_revenue", "amount": "$1.8 billion", "drivers": ["order_volume", "average_order_values"]}
                        )
                    ]
                )
            ]
            
            print(f"🚀 DEBUG: Calling lx.extract for Page {page_number}")
            
            result = lx.extract(
                text_or_documents=page_text,
                prompt_description=chunking_prompt,
                examples=examples,
                model_id="gpt-4o",
                api_key=os.getenv("OPENAI_API_KEY"),
                fence_output=True,
                use_schema_constraints=False
            )
            
            print(f"✅ DEBUG: LangExtract call completed for Page {page_number}")
            
            # Process the structured data into chunks
            page_chunks = []
            
            if hasattr(result, 'extractions') and result.extractions:
                print(f"📊 DEBUG: Found {len(result.extractions)} extractions from Page {page_number}")
                
                for i, extraction in enumerate(result.extractions):
                    chunk_text = getattr(extraction, 'extraction_text', None)
                    extraction_class = getattr(extraction, 'extraction_class', 'general')
                    attributes = getattr(extraction, 'attributes', {})
                    
                    if chunk_text and len(chunk_text.strip()) > 50:  # Minimum chunk size
                        # Determine topic and importance based on extraction class and attributes
                        topic = extraction_class.title() if extraction_class else 'General Content'
                        importance = 'High' if extraction_class in ['financial', 'revenue', 'profit'] else 'Medium'
                        
                        page_chunks.append({
                            'text': chunk_text,
                            'topic': topic,
                            'importance': importance,
                            'type': extraction_class,
                            'chunk_id': f"{pdf_name}_page_{page_number}_chunk_{i}",
                            'source_location': f"Page {page_number}",
                            'page_number': page_number,
                            'attributes': attributes
                        })
                        print(f"✅ DEBUG: ✓ Page {page_number} chunk {i+1}: {extraction_class} - {len(chunk_text)} chars")
                    else:
                        print(f"❌ DEBUG: ✗ Skipped Page {page_number} chunk {i+1}: text too short")
            else:
                print(f"❌ DEBUG: No extractions found for Page {page_number}")
            
            all_chunks.extend(page_chunks)
            print(f"📊 DEBUG: Page {page_number} contributed {len(page_chunks)} chunks")
            
        except Exception as e:
            print(f"\n❌ DEBUG: LangExtract failed for Page {page_number}:")
            print(f"❌ DEBUG: Exception: {str(e)}")
            # Continue with next page instead of failing completely
            continue
    
    print(f"\n🔍 DEBUG: Final chunk count: {len(all_chunks)} from {len(pages_data)} pages")
    return all_chunks

def extract_page_number(text):
    """Extract page number from text that contains [PAGE X] markers."""
    page_match = re.search(r'\[PAGE (\d+)\]', text)
    return int(page_match.group(1)) if page_match else None

def smart_chunk_with_langextract(text, pdf_name):
    """Use LangExtract to create intelligent chunks with semantic understanding."""
    print(f"\n🔍 DEBUG: Starting LangExtract processing for {pdf_name}")
    print(f"📄 DEBUG: Input text length: {len(text)} characters")
    print(f"📄 DEBUG: Input text preview: {text[:200]}...")
    
    try:
        # Define chunking prompt for LangExtract - focused on financial document extraction
        chunking_prompt = """
        Extract key information from this financial document. Identify and extract meaningful chunks of information.
        Focus on extracting:
        1. Financial metrics (revenue, profit, costs, investments)
        2. Business operations and strategies
        3. Key announcements and developments
        4. Risk factors and forward-looking statements
        5. Company background and history
        
        Each extraction should be a coherent piece of information that can stand alone.
        """
        
        print(f"📝 DEBUG: Chunking prompt: {chunking_prompt[:100]}...")
        
        # Define examples for LangExtract using proper ExampleData objects
        examples = [
            lx.data.ExampleData(
                text="DoorDash reported Q3 revenue of $2.2 billion, up 27% year-over-year. The company's marketplace revenue grew to $1.8 billion, driven by increased order volume and higher average order values.",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="financial",
                        extraction_text="DoorDash reported Q3 revenue of $2.2 billion, up 27% year-over-year.",
                        attributes={"metric": "revenue", "period": "Q3", "growth": "27%"}
                    ),
                    lx.data.Extraction(
                        extraction_class="financial", 
                        extraction_text="The company's marketplace revenue grew to $1.8 billion, driven by increased order volume and higher average order values.",
                        attributes={"metric": "marketplace_revenue", "amount": "$1.8 billion", "drivers": ["order_volume", "average_order_values"]}
                    )
                ]
            )
        ]
        
        print(f"📚 DEBUG: Examples: {examples}")
        print(f"🔑 DEBUG: OpenAI API Key exists: {bool(os.getenv('OPENAI_API_KEY'))}")
        print(f"🔑 DEBUG: OpenAI API Key preview: {os.getenv('OPENAI_API_KEY')[:10]}..." if os.getenv('OPENAI_API_KEY') else "None")
        
        # Use LangExtract with OpenAI model
        print(f"🚀 DEBUG: Calling lx.extract with:")
        print(f"   - text_or_documents: {len(text[:4000])} chars")
        print(f"   - text preview: {text[:4000][:200]}...")
        print(f"   - text contains PAGE markers: {'[PAGE' in text[:4000]}")
        print(f"   - model_id: gpt-4o")
        print(f"   - fence_output: True")
        print(f"   - use_schema_constraints: False")
        
        result = lx.extract(
            text_or_documents=text[:4000],  # Limit to avoid token limits
            prompt_description=chunking_prompt,
            examples=examples,
            model_id="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
            fence_output=True,
            use_schema_constraints=False
        )
        
        print(f"✅ DEBUG: LangExtract call completed successfully!")
        
        # Debug: Print the actual result structure
        print(f"\n🔍 DEBUG: LangExtract result analysis:")
        print(f"📊 DEBUG: Result type: {type(result)}")
        print(f"📊 DEBUG: Result dir: {dir(result)}")
        print(f"📊 DEBUG: Result str: {str(result)[:500]}...")
        
        # Check if result is a dict
        if isinstance(result, dict):
            print(f"📊 DEBUG: Result is a dict with keys: {list(result.keys())}")
            for key, value in result.items():
                print(f"📊 DEBUG:   {key}: {type(value)} = {str(value)[:100]}...")
        
        # Process the structured data into chunks
        chunks = []
        
        print(f"\n🔍 DEBUG: Checking for extractions attribute...")
        print(f"📊 DEBUG: hasattr(result, 'extractions'): {hasattr(result, 'extractions')}")
        
        if hasattr(result, 'extractions'):
            print(f"📊 DEBUG: result.extractions type: {type(result.extractions)}")
            print(f"📊 DEBUG: result.extractions value: {result.extractions}")
            
            if result.extractions:
                print(f"📊 DEBUG: Found {len(result.extractions)} extractions from LangExtract")
                
                for i, extraction in enumerate(result.extractions):
                    print(f"\n🔍 DEBUG: Processing extraction {i+1}:")
                    print(f"📊 DEBUG:   Extraction type: {type(extraction)}")
                    print(f"📊 DEBUG:   Extraction dir: {dir(extraction)}")
                    
                    # Access the extraction text and metadata
                    chunk_text = getattr(extraction, 'extraction_text', None)
                    extraction_class = getattr(extraction, 'extraction_class', 'general')
                    attributes = getattr(extraction, 'attributes', {})
                    provenance = getattr(extraction, 'provenance', [])
                    
                    print(f"📊 DEBUG:   chunk_text: {chunk_text[:100] if chunk_text else 'None'}...")
                    print(f"📊 DEBUG:   chunk_text contains PAGE marker: {'[PAGE' in chunk_text if chunk_text else False}")
                    print(f"📊 DEBUG:   extraction_class: {extraction_class}")
                    print(f"📊 DEBUG:   attributes: {attributes}")
                    print(f"📊 DEBUG:   provenance: {provenance}")
                    
                    if chunk_text and len(chunk_text.strip()) > 50:  # Minimum chunk size
                        # Extract page number from chunk text
                        page_number = extract_page_number(chunk_text)
                        
                        # Determine topic and importance based on extraction class and attributes
                        topic = extraction_class.title() if extraction_class else 'General Content'
                        importance = 'High' if extraction_class in ['financial', 'revenue', 'profit'] else 'Medium'
                        
                        # Create user-friendly source location with page number
                        source_location = f"Page {page_number}" if page_number else "Unknown page"
                        
                        chunks.append({
                            'text': chunk_text,
                            'topic': topic,
                            'importance': importance,
                            'type': extraction_class,
                            'chunk_id': f"{pdf_name}_chunk_{i}",
                            'source_location': source_location,
                            'page_number': page_number,
                            'attributes': attributes,
                            'provenance': provenance
                        })
                        print(f"✅ DEBUG: ✓ LangExtract chunk {i+1}: {extraction_class} - Page {page_number} - {len(chunk_text)} chars")
                    else:
                        print(f"❌ DEBUG: ✗ Skipped extraction {i+1}: text too short or None")
            else:
                print(f"❌ DEBUG: result.extractions is empty or None")
        else:
            print(f"❌ DEBUG: result has no 'extractions' attribute")
        
        print(f"\n🔍 DEBUG: Final chunk count: {len(chunks)}")
        
        if chunks:
            print(f"✅ DEBUG: LangExtract successfully created {len(chunks)} intelligent chunks")
            return chunks
        else:
            print(f"❌ DEBUG: LangExtract returned no chunks")
            return []
        
    except Exception as e:
        print(f"\n❌ DEBUG: LangExtract failed with exception:")
        print(f"❌ DEBUG: Exception type: {type(e)}")
        print(f"❌ DEBUG: Exception message: {str(e)}")
        print(f"❌ DEBUG: Exception args: {e.args}")
        import traceback
        print(f"❌ DEBUG: Full traceback:")
        traceback.print_exc()
        print(f"❌ DEBUG: LangExtract processing failed")
        return []


def generate_query_variations(original_query):
    """Generate multiple query variations using LLM for comprehensive RAG coverage."""
    print(f"\n🔍 DEBUG: Generating query variations for: '{original_query}'")
    
    try:
        expansion_prompt = f"""
        Given the following user question, generate 4 different query variations that would help retrieve comprehensive information from financial documents. Each variation should approach the question from a different angle:

        Original question: "{original_query}"

        Generate variations that:
        1. Use different terminology/synonyms
        2. Ask for specific metrics/data points
        3. Focus on different aspects (financial, operational, strategic, etc.)
        4. Use broader or more specific phrasing

        Return ONLY a JSON array of 4 query strings, no other text.
        Example format: ["query 1", "query 2", "query 3", "query 4"]
        """
        
        print(f"📝 DEBUG: Query expansion prompt length: {len(expansion_prompt)}")
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert at generating diverse query variations for information retrieval. Return only valid JSON arrays."},
                {"role": "user", "content": expansion_prompt}
            ],
            temperature=0.7  # Higher temperature for more diverse variations
        )
        
        print(f"✅ DEBUG: Query expansion response received")
        print(f"📊 DEBUG: Response content: {response.choices[0].message.content}")
        
        # Parse the response
        import json
        try:
            response_content = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks if present
            if response_content.startswith('```json'):
                response_content = response_content[7:]  # Remove ```json
            if response_content.startswith('```'):
                response_content = response_content[3:]   # Remove ```
            if response_content.endswith('```'):
                response_content = response_content[:-3]  # Remove trailing ```
            
            response_content = response_content.strip()
            print(f"📊 DEBUG: Cleaned response content: {response_content}")
            
            query_variations = json.loads(response_content)
            print(f"✅ DEBUG: Successfully parsed {len(query_variations)} query variations")
            
            # Add original query to the list
            all_queries = [original_query] + query_variations
            print(f"📊 DEBUG: Total queries (including original): {len(all_queries)}")
            
            for i, query in enumerate(all_queries):
                print(f"📊 DEBUG: Query {i+1}: {query}")
            
            return all_queries
            
        except json.JSONDecodeError as e:
            print(f"❌ DEBUG: Failed to parse query variations as JSON: {e}")
            print(f"❌ DEBUG: Raw response: {response.choices[0].message.content}")
            # Fallback to original query only
            return [original_query]
            
    except Exception as e:
        print(f"\n❌ DEBUG: Query expansion failed with exception:")
        print(f"❌ DEBUG: Exception: {str(e)}")
        # Fallback to original query only
        return [original_query]

def get_embedding(text):
    """Get embedding for text using OpenAI."""
    # Truncate text if it's too long (safety check)
    max_tokens = 8000  # Leave some buffer for the embedding model
    if len(text) > max_tokens:
        text = text[:max_tokens]
    
    try:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error creating embedding: {e}")
        return [0.0] * 1536

def multi_query_search(queries, collection_name, limit_per_query=2):
    """Search Qdrant with multiple queries and merge results."""
    print(f"\n🔍 DEBUG: Multi-query search with {len(queries)} queries")
    
    all_results = []
    seen_chunk_ids = set()
    
    for i, query in enumerate(queries):
        print(f"📊 DEBUG: Processing query {i+1}/{len(queries)}: '{query}'")
        
        try:
            # Get query embedding
            query_embedding = get_embedding(query)
            
            # Search Qdrant for relevant chunks
            search_results = qdrant.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=limit_per_query
            )
            
            print(f"📊 DEBUG: Query {i+1} returned {len(search_results)} results")
            
            # Add unique results to our collection
            for hit in search_results:
                chunk_id = hit.payload.get('chunk_id', str(hit.id))
                if chunk_id not in seen_chunk_ids:
                    all_results.append(hit)
                    seen_chunk_ids.add(chunk_id)
                    print(f"✅ DEBUG: Added unique result from query {i+1}: {chunk_id}")
                else:
                    print(f"🔄 DEBUG: Skipped duplicate result from query {i+1}: {chunk_id}")
            
        except Exception as e:
            print(f"❌ DEBUG: Error processing query {i+1}: {e}")
            continue
    
    print(f"📊 DEBUG: Multi-query search completed: {len(all_results)} unique results from {len(queries)} queries")
    return all_results

def upload_pdf(file):
    """Process and upload PDF to Qdrant."""
    print(f"\n📤 DEBUG: ===== STARTING PDF UPLOAD =====")
    print(f"📤 DEBUG: File: {file}")
    
    if file is None:
        print("📤 DEBUG: No file provided")
        return "No file uploaded"
    
    # Gradio passes file path as string
    import shutil
    import os
    
    # Get filename from the file path
    filename = os.path.basename(file)
    pdf_path = UPLOAD_DIR / filename
    print(f"📤 DEBUG: Processing file: {filename}")
    print(f"📤 DEBUG: PDF path: {pdf_path}")
    
    # Copy file to our uploads directory
    try:
        shutil.copy2(file, pdf_path)
        print(f"📤 DEBUG: ✓ File copied successfully")
    except shutil.SameFileError:
        # File is already in the right location, just use it
        print(f"📤 DEBUG: ✓ File already in correct location")
        pass
    
    print(f"📤 DEBUG: Starting text extraction...")
    # Extract text page by page and create smart chunks
    pages_data = extract_text_by_page(pdf_path)
    pdf_name = os.path.basename(file)
    print(f"📤 DEBUG: ✓ Text extraction completed: {len(pages_data)} pages")
    
    # Extract figures from PDF (with timeout protection)
    print(f"\n🖼️ DEBUG: ===== STARTING FIGURE EXTRACTION FOR {pdf_name} =====")
    model = get_yolo_model()
    if model is None:
        print("🖼️ DEBUG: ⚠️ Warning: Could not load YOLO model, skipping figure extraction")
        figure_chunks = []
    else:
        print("🖼️ DEBUG: ✓ YOLO model loaded successfully")
        # Convert PDF to images
        images = convert_pdf_to_images(pdf_path)
        if not images:
            print("🖼️ DEBUG: ⚠️ Warning: Could not convert PDF to images, skipping figure extraction")
            figure_chunks = []
        else:
            print(f"🖼️ DEBUG: ✓ PDF converted to {len(images)} images")
            figure_chunks = []
            total_figures = 0
            
            # Process each page for figures (limit to first 5 pages for faster processing)
            max_pages_to_process = min(5, len(images))  # Limit to first 5 pages
            print(f"🖼️ DEBUG: Processing first {max_pages_to_process} pages out of {len(images)} total pages")
            
            for page_num, image in enumerate(images[:max_pages_to_process]):
                print(f"\n🖼️ DEBUG: ===== PROCESSING PAGE {page_num + 1}/{max_pages_to_process} FOR FIGURES =====")
                
                # Detect figures on this page
                detections = detect_figures(model, image)
                
                if detections:
                    print(f"🖼️ DEBUG: ✓ Found {len(detections)} figures on page {page_num + 1}")
                    # Extract and save figures
                    saved_figures = extract_and_save_figures(image, detections, page_num, pdf_name)
                    
                    # Analyze each figure with Vision API and create chunks
                    for fig_idx, figure_metadata in enumerate(saved_figures):
                        print(f"🖼️ DEBUG: ===== ANALYZING FIGURE {fig_idx + 1} WITH VISION API =====")
                        figure_description = analyze_figure_with_vision_api(figure_metadata['figure_path'])
                        
                        # Create figure chunk
                        figure_chunk = {
                            'text': figure_description,
                            'topic': "Figure/Chart/Diagram",
                            'importance': 'High',
                            'type': 'figure',
                            'chunk_id': f"{pdf_name}_figure_{page_num+1}_{fig_idx+1}",
                            'source_location': f"Page {page_num + 1}",
                            'page_number': page_num + 1,
                            'is_figure': True,
                            'figure_path': figure_metadata['figure_path'],
                            'figure_filename': figure_metadata['filename'],
                            'figure_class': figure_metadata['class_name'],
                            'figure_confidence': figure_metadata['confidence'],
                            'figure_bbox': figure_metadata['bbox']
                        }
                        figure_chunks.append(figure_chunk)
                        total_figures += 1
                        print(f"🖼️ DEBUG: ✓ Created figure chunk: {figure_metadata['filename']}")
                        print(f"🖼️ DEBUG: ✓ Chunk description length: {len(figure_description)} chars")
                else:
                    print(f"🖼️ DEBUG: ✗ No figures detected on page {page_num + 1}")
            
            print(f"\n🖼️ DEBUG: ===== FIGURE EXTRACTION COMPLETED =====")
            print(f"🖼️ DEBUG: ✓ Total figures extracted: {total_figures}")
            print(f"🖼️ DEBUG: ✓ Total figure chunks created: {len(figure_chunks)}")
    
    print(f"📤 DEBUG: Starting text chunking...")
    chunks = smart_chunk_with_langextract_page_by_page(pages_data, pdf_name)
    print(f"📤 DEBUG: ✓ Text chunking completed: {len(chunks)} chunks")
    
    # Combine text and figure chunks
    all_chunks = chunks + figure_chunks
    print(f"📤 DEBUG: ✓ Combined chunks: {len(all_chunks)} total ({len(chunks)} text + {len(figure_chunks)} figures)")
    
    # If LangExtract fails, return error instead of fallback
    if not chunks:
        print("📤 DEBUG: ✗ LangExtract failed")
        return f"Failed to process {pdf_name} with LangExtract. Please check the document content."
    
    if not all_chunks:
        print("📤 DEBUG: ✗ No chunks created")
        return f"No readable text or figures found in {pdf_name}"
    
    # Create embeddings and store in Qdrant
    print(f"📤 DEBUG: Starting embedding creation and Qdrant storage...")
    points = []
    successful_chunks = 0
    
    print(f"📤 DEBUG: Processing {len(all_chunks)} total chunks from {pdf_name} ({len(chunks)} text + {len(figure_chunks)} figures)")
    
    for i, chunk_data in enumerate(all_chunks):
        try:
            chunk_text = chunk_data['text']
            print(f"📤 DEBUG: Processing chunk {i+1}/{len(all_chunks)} (topic: {chunk_data['topic']}, length: {len(chunk_text)})")
            embedding = get_embedding(chunk_text)
            # Skip if embedding is zero vector (error occurred)
            if embedding != [0.0] * 1536:
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "text": chunk_text,
                        "pdf_name": pdf_name,
                        "chunk_index": i,
                        "pdf_path": str(pdf_path),
                        "topic": chunk_data['topic'],
                        "importance": chunk_data['importance'],
                        "type": chunk_data['type'],
                        "chunk_id": chunk_data['chunk_id'],
                        "source_location": chunk_data['source_location'],
                        "page_number": chunk_data.get('page_number'),
                        "is_figure": chunk_data.get('is_figure', False),
                        "figure_path": chunk_data.get('figure_path'),
                        "figure_filename": chunk_data.get('figure_filename'),
                        "figure_class": chunk_data.get('figure_class'),
                        "figure_confidence": chunk_data.get('figure_confidence'),
                        "figure_bbox": chunk_data.get('figure_bbox')
                    }
                )
                points.append(point)
                successful_chunks += 1
                print(f"📤 DEBUG: ✓ Chunk {i+1} ({chunk_data['topic']}) processed successfully")
            else:
                print(f"📤 DEBUG: ✗ Chunk {i+1} failed - zero embedding")
        except Exception as e:
            print(f"📤 DEBUG: ✗ Error processing chunk {i+1}: {e}")
            continue
    
    if points:
        try:
            print(f"📤 DEBUG: Storing {successful_chunks} chunks in Qdrant...")
            qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
            print(f"📤 DEBUG: ✓ Successfully stored {successful_chunks} chunks in Qdrant")
            result_message = f"Successfully uploaded {os.path.basename(file)} with {successful_chunks} chunks ({len(chunks)} text + {len(figure_chunks)} figures)"
            print(f"📤 DEBUG: ===== UPLOAD COMPLETED =====")
            print(f"📤 DEBUG: Result: {result_message}")
            return result_message
        except Exception as e:
            print(f"📤 DEBUG: ✗ Error storing in Qdrant: {e}")
            return f"Error storing chunks in database: {e}"
    else:
        print("📤 DEBUG: ✗ No points to store")
        return f"Failed to process any chunks from {os.path.basename(file)}. Please check the PDF content."

def query_rag_with_figures(message, history):
    """Query the RAG system with streaming response and figure information."""
    print(f"\n🔍 DEBUG: ===== STARTING QUERY RAG WITH FIGURES =====")
    print(f"🔍 DEBUG: Query: '{message}'")
    print(f"🔍 DEBUG: History length: {len(history) if history else 0}")
    
    if not message.strip():
        print("🔍 DEBUG: Empty query, returning empty response")
        return "", []
    
    # Debug: Check collection info
    try:
        collection_info = qdrant.get_collection(COLLECTION_NAME)
        print(f"🔍 DEBUG: Collection points count: {collection_info.points_count}")
    except Exception as e:
        print(f"🔍 DEBUG: Error getting collection info: {e}")
        return "Error accessing document collection. Please try uploading a PDF first.", []
    
    # Generate query variations for comprehensive coverage
    print(f"🔍 DEBUG: Generating query variations...")
    expanded_queries = generate_query_variations(message)
    print(f"🔍 DEBUG: Generated {len(expanded_queries)} query variations")
    
    # Search Qdrant with multiple queries
    print(f"🔍 DEBUG: Searching Qdrant with multiple queries...")
    search_results = multi_query_search(expanded_queries, COLLECTION_NAME, limit_per_query=2)
    
    print(f"🔍 DEBUG: Search results count: {len(search_results)}")
        
    if not search_results:
        print("🔍 DEBUG: No search results found")
        return "No relevant documents found. Please upload some PDFs first.", []
    
    # Check if this is a figure query and prioritize figure results
    is_figure_request = is_figure_query(message)
    print(f"🔍 DEBUG: Is figure query: {is_figure_request}")
    
    # Separate figure and text results
    figure_results = []
    text_results = []
    
    for hit in search_results:
        if hit.payload.get('is_figure', False):
            figure_results.append(hit)
        else:
            text_results.append(hit)
    
    print(f"🔍 DEBUG: Results breakdown: {len(figure_results)} figures, {len(text_results)} text")
    
    # Prioritize figure results if user is asking for figures
    if is_figure_request and figure_results:
        search_results = figure_results + text_results
        print(f"🔍 DEBUG: ✓ Prioritizing {len(figure_results)} figure results")
    elif not is_figure_request and figure_results:
        search_results = text_results + figure_results
        print(f"🔍 DEBUG: ✓ Prioritizing {len(text_results)} text results")
    
    # Build context from search results with citations
    print(f"🔍 DEBUG: Building context from search results...")
    context_parts = []
    citations = []
    figures_found = []  # Store figure information
    total_length = 0
    max_context_length = 8000  # Limit context to ~8000 characters
    
    for i, hit in enumerate(search_results):
        print(f"🔍 DEBUG: Processing hit {i+1}/{len(search_results)}")
        
        # Try different ways to access the text
        chunk_text = None
        chunk_metadata = {}
        if hasattr(hit, 'payload') and hit.payload:
            if isinstance(hit.payload, dict) and "text" in hit.payload:
                chunk_text = hit.payload["text"]
                chunk_metadata = hit.payload
            elif hasattr(hit.payload, 'text'):
                chunk_text = hit.payload.text
                chunk_metadata = hit.payload
        
        if chunk_text:
            print(f"🔍 DEBUG: Found text chunk: {len(chunk_text)} characters")
            
            # Check if this is a figure result
            is_figure = chunk_metadata.get('is_figure', False)
            print(f"🔍 DEBUG: Is figure chunk: {is_figure}")
            
            if is_figure:
                # Store figure information for later use
                figure_info = {
                    'path': chunk_metadata.get('figure_path'),
                    'description': chunk_text,
                    'page': chunk_metadata.get('page_number'),
                    'source': chunk_metadata.get('pdf_name'),
                    'filename': chunk_metadata.get('figure_filename'),
                    'class': chunk_metadata.get('figure_class'),
                    'confidence': chunk_metadata.get('figure_confidence')
                }
                figures_found.append(figure_info)
                print(f"🔍 DEBUG: ✓ Found figure: {figure_info['filename']} on page {figure_info['page']}")
            
            if total_length + len(chunk_text) > max_context_length:
                print(f"🔍 DEBUG: Chunk too large, skipping (would exceed {max_context_length} limit)")
                break
            
            # Create citation reference
            citation_ref = f"[{i+1}]"
            citations.append({
                'ref': citation_ref,
                'text': chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text,
                'source': chunk_metadata.get('source_location', 'Unknown source'),
                'topic': chunk_metadata.get('topic', 'General'),
                'pdf_name': chunk_metadata.get('pdf_name', 'Unknown document'),
                'page_number': chunk_metadata.get('page_number')
            })
            
            # Add citation reference to the chunk
            cited_chunk = f"{chunk_text} {citation_ref}"
            context_parts.append(cited_chunk)
            total_length += len(cited_chunk)
            print(f"🔍 DEBUG: ✓ Added chunk to context with citation {citation_ref}. Total length now: {total_length}")
        else:
            print(f"🔍 DEBUG: ✗ No text found in hit {i+1}")
    
    context = "\n\n".join(context_parts)
    
    # Debug: Print context being sent to LLM
    print(f"🔍 DEBUG: Context length: {len(context)} characters")
    print(f"🔍 DEBUG: Context preview: {context[:200]}...")
    print(f"🔍 DEBUG: User question: {message}")
    
    # Create messages for OpenAI chat with citation instructions
    citation_instructions = """
    IMPORTANT: The context includes citation references like [1], [2], etc. 
    When you reference information from the context, you MUST include the citation reference in your response.
    For example: "DoorDash reported revenue of $2.2 billion [1]..."
    """
    
    messages = [
        {
            "role": "system",
            "content": f"You are a helpful assistant that answers questions based ONLY on the provided context from PDF documents. {citation_instructions} You MUST use the information from the context to answer questions and include appropriate citation references. If the context contains relevant information, provide a detailed answer based on that information with proper citations. If the context doesn't contain enough information to answer the question, say so clearly."
        },
        {
            "role": "user",
            "content": f"Here is the context from PDF documents with citation references:\n\n{context}\n\nBased on this context, please answer the following question: {message}\n\nRemember to include citation references [1], [2], etc. when referencing information from the context."
        }
    ]
    
    # Stream response from OpenAI
    print(f"🔍 DEBUG: Sending request to OpenAI GPT-4o...")
    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        stream=True
    )
    
    response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            response += chunk.choices[0].delta.content
    
    print(f"🔍 DEBUG: ✓ OpenAI response received ({len(response)} chars)")
    
    # Rank and select top 3 most relevant figures
    print(f"🔍 DEBUG: Ranking {len(figures_found)} figures by relevance...")
    top_figures = rank_figures_by_relevance(figures_found, message)
    print(f"🔍 DEBUG: Selected top {len(top_figures)} figures")
    
    # Add citations section after streaming is complete
    if citations:
        print(f"🔍 DEBUG: Adding {len(citations)} citations to response")
        citations_text = "\n\n**Sources:**\n"
        for citation in citations:
            print(f"🔍 DEBUG: Citation {citation['ref']}: page_number={citation['page_number']}, pdf_name={citation['pdf_name']}")
            page_info = f" (Page {citation['page_number']})" if citation['page_number'] else " (Page unknown)"
            pdf_name = citation['pdf_name'].replace('.pdf', '') if citation['pdf_name'] else 'Unknown Document'
            citations_text += f"{citation['ref']} {pdf_name}{page_info} - {citation['topic']}\n"
        
        # Add figure information if figures were found (show top 3 only)
        if figures_found:
            print(f"🔍 DEBUG: Adding top {len(top_figures)} figures to response")
            citations_text += "\n**Top Related Figures:**\n"
            for i, fig in enumerate(top_figures):
                citations_text += f"📊 {fig['filename']} (Page {fig['page']}) - {fig['class']}\n"
        
        response += citations_text
    elif figures_found:
        # If only figures found, add figure information (show top 3 only)
        print(f"🔍 DEBUG: Adding top {len(top_figures)} figures to response (no citations)")
        figures_text = "\n\n**Top Related Figures:**\n"
        for fig in top_figures:
            figures_text += f"📊 {fig['filename']} (Page {fig['page']}) - {fig['class']}\n"
        response += figures_text
    
    # Prepare figure paths for gallery (top 3 only)
    figure_paths = [fig['path'] for fig in top_figures if fig['path'] and Path(fig['path']).exists()]
    print(f"🔍 DEBUG: Returning {len(figure_paths)} top figure paths for gallery")
    print(f"🔍 DEBUG: Top figure paths: {figure_paths}")
    
    print(f"🔍 DEBUG: ===== QUERY RAG WITH FIGURES COMPLETED =====")
    return response, figure_paths

def query_rag(message, history):
    """Query the RAG system with streaming response."""
    if not message.strip():
        return ""
    
    # Debug: Check collection info
    try:
        collection_info = qdrant.get_collection(COLLECTION_NAME)
        print(f"Collection points count: {collection_info.points_count}")
    except Exception as e:
        print(f"Error getting collection info: {e}")
        return "Error accessing document collection. Please try uploading a PDF first."
    
    # Generate query variations for comprehensive coverage
    expanded_queries = generate_query_variations(message)
    
    # Search Qdrant with multiple queries
    search_results = multi_query_search(expanded_queries, COLLECTION_NAME, limit_per_query=2)
    
    print(f"Search results count: {len(search_results)}")
        
    if not search_results:
        return "No relevant documents found. Please upload some PDFs first."
    
    # Check if this is a figure query and prioritize figure results
    is_figure_request = is_figure_query(message)
    print(f"🖼️ Is figure query: {is_figure_request}")
    
    # Separate figure and text results
    figure_results = []
    text_results = []
    
    for hit in search_results:
        if hit.payload.get('is_figure', False):
            figure_results.append(hit)
        else:
            text_results.append(hit)
    
    print(f"📊 Results breakdown: {len(figure_results)} figures, {len(text_results)} text")
    
    # Prioritize figure results if user is asking for figures
    if is_figure_request and figure_results:
        search_results = figure_results + text_results
        print(f"🖼️ Prioritizing {len(figure_results)} figure results")
    elif not is_figure_request and figure_results:
        search_results = text_results + figure_results
        print(f"📝 Prioritizing {len(text_results)} text results")
    
    # Debug: Print search results structure
    print(f"First search result type: {type(search_results[0])}")
    print(f"First search result: {search_results[0]}")
    print(f"First search result payload: {search_results[0].payload}")
    print(f"First search result payload type: {type(search_results[0].payload)}")
    
    # Build context from search results with citations
    context_parts = []
    citations = []
    figures_found = []  # Store figure information
    total_length = 0
    max_context_length = 8000  # Limit context to ~8000 characters
    
    for i, hit in enumerate(search_results):
        print(f"Processing hit {i+1}: {hit}")
        print(f"Hit payload: {hit.payload}")
        
        # Try different ways to access the text
        chunk_text = None
        chunk_metadata = {}
        if hasattr(hit, 'payload') and hit.payload:
            if isinstance(hit.payload, dict) and "text" in hit.payload:
                chunk_text = hit.payload["text"]
                chunk_metadata = hit.payload
            elif hasattr(hit.payload, 'text'):
                chunk_text = hit.payload.text
                chunk_metadata = hit.payload
        
        if chunk_text:
            print(f"Found text chunk: {len(chunk_text)} characters")
            print(f"🔍 DEBUG: Chunk metadata page_number: {chunk_metadata.get('page_number')}")
            print(f"🔍 DEBUG: Chunk metadata keys: {list(chunk_metadata.keys())}")
            
            # Check if this is a figure result
            is_figure = chunk_metadata.get('is_figure', False)
            
            if is_figure:
                # Store figure information for later use
                figure_info = {
                    'path': chunk_metadata.get('figure_path'),
                    'description': chunk_text,
                    'page': chunk_metadata.get('page_number'),
                    'source': chunk_metadata.get('pdf_name'),
                    'filename': chunk_metadata.get('figure_filename'),
                    'class': chunk_metadata.get('figure_class'),
                    'confidence': chunk_metadata.get('figure_confidence')
                }
                figures_found.append(figure_info)
                print(f"🖼️ Found figure: {figure_info['filename']} on page {figure_info['page']}")
            
            if total_length + len(chunk_text) > max_context_length:
                print(f"Chunk too large, skipping (would exceed {max_context_length} limit)")
                break
            
            # Create citation reference
            citation_ref = f"[{i+1}]"
            citations.append({
                'ref': citation_ref,
                'text': chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text,
                'source': chunk_metadata.get('source_location', 'Unknown source'),
                'topic': chunk_metadata.get('topic', 'General'),
                'pdf_name': chunk_metadata.get('pdf_name', 'Unknown document'),
                'page_number': chunk_metadata.get('page_number')
            })
            
            # Add citation reference to the chunk
            cited_chunk = f"{chunk_text} {citation_ref}"
            context_parts.append(cited_chunk)
            total_length += len(cited_chunk)
            print(f"Added chunk to context with citation {citation_ref}. Total length now: {total_length}")
        else:
            print(f"No text found in hit {i+1}")
    
    context = "\n\n".join(context_parts)
    
    # Debug: Print context being sent to LLM
    print(f"Context length: {len(context)} characters")
    print(f"Context preview: {context[:200]}...")
    print(f"User question: {message}")
    
    # Create messages for OpenAI chat with citation instructions
    citation_instructions = """
    IMPORTANT: The context includes citation references like [1], [2], etc. 
    When you reference information from the context, you MUST include the citation reference in your response.
    For example: "DoorDash reported revenue of $2.2 billion [1]..."
    """
    
    messages = [
        {
            "role": "system",
            "content": f"You are a helpful assistant that answers questions based ONLY on the provided context from PDF documents. {citation_instructions} You MUST use the information from the context to answer questions and include appropriate citation references. If the context contains relevant information, provide a detailed answer based on that information with proper citations. If the context doesn't contain enough information to answer the question, say so clearly."
        },
        {
            "role": "user",
            "content": f"Here is the context from PDF documents with citation references:\n\n{context}\n\nBased on this context, please answer the following question: {message}\n\nRemember to include citation references [1], [2], etc. when referencing information from the context."
        }
    ]
    
    # Stream response from OpenAI
    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        stream=True
    )
    
    response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            response += chunk.choices[0].delta.content
            yield response
    
    # Add citations section after streaming is complete
    if citations:
        citations_text = "\n\n**Sources:**\n"
        for citation in citations:
            print(f"🔍 DEBUG: Citation {citation['ref']}: page_number={citation['page_number']}, pdf_name={citation['pdf_name']}")
            page_info = f" (Page {citation['page_number']})" if citation['page_number'] else " (Page unknown)"
            pdf_name = citation['pdf_name'].replace('.pdf', '') if citation['pdf_name'] else 'Unknown Document'
            citations_text += f"{citation['ref']} {pdf_name}{page_info} - {citation['topic']}\n"
        
        # Add figure information if figures were found
        if figures_found:
            citations_text += "\n**Related Figures:**\n"
            for fig in figures_found:
                citations_text += f"📊 {fig['filename']} (Page {fig['page']}) - {fig['class']}\n"
        
        yield response + citations_text
    elif figures_found:
        # If only figures found, add figure information
        figures_text = "\n\n**Related Figures:**\n"
        for fig in figures_found:
            figures_text += f"📊 {fig['filename']} (Page {fig['page']}) - {fig['class']}\n"
        yield response + figures_text
    else:
        yield response

# Create Gradio interface
with gr.Blocks(title="GIC Financial Docs Assistant") as demo:
    gr.Markdown("# GIC Financial Docs Assistant")
    gr.Markdown("Upload PDFs and ask questions about their content!")
    
    with gr.Tab("Upload PDFs"):
        file_input = gr.File(
            label="Upload PDF",
            file_types=[".pdf"],
            type="filepath"
        )
        upload_button = gr.Button("Upload PDF")
        upload_status = gr.Textbox(label="Status", interactive=False)
        
        def check_collection():
            try:
                collection_info = qdrant.get_collection(COLLECTION_NAME)
                return f"Collection has {collection_info.points_count} documents"
            except Exception as e:
                return f"Error: {e}"
        
        check_button = gr.Button("Check Collection")
        collection_status = gr.Textbox(label="Collection Status", interactive=False)
        
        upload_button.click(
            upload_pdf,
            inputs=file_input,
            outputs=upload_status
        )
        check_button.click(
            check_collection,
            outputs=collection_status
        )
    
    with gr.Tab("Ask Questions"):
        chatbot = gr.Chatbot(
            label="Chat with your PDFs",
            height=500,
            type="messages"
        )
        figure_gallery = gr.Gallery(
            label="Related Figures",
            columns=2,
            rows=2,
            height=300,
            show_label=True
        )
        msg = gr.Textbox(
            label="Ask a question about your uploaded PDFs",
            placeholder="What is this document about? Try asking for charts or figures!"
        )
        clear = gr.Button("Clear")
        
        def user(user_message, history):
            # history is a list of message dicts when type="messages"
            history = history or []
            return "", history + [{"role": "user", "content": user_message}]
        
        def bot(history):
            # Expect history as list[{"role":..., "content":...}]
            print(f"\n🤖 DEBUG: ===== BOT FUNCTION CALLED =====")
            history = history or []
            if not history or history[-1].get("role") != "user":
                print("🤖 DEBUG: No user message found, returning empty")
                yield history, []
                return

            user_message = history[-1]["content"]
            print(f"🤖 DEBUG: User message: '{user_message}'")
            messages = history + [{"role": "assistant", "content": ""}]

            # Use the new function that returns both response and figures
            print(f"🤖 DEBUG: Calling query_rag_with_figures...")
            response, figure_paths = query_rag_with_figures(user_message, messages)
            print(f"🤖 DEBUG: Response length: {len(response)} chars")
            print(f"🤖 DEBUG: Figure paths count: {len(figure_paths)}")
            
            messages[-1]["content"] = response
            print(f"🤖 DEBUG: Returning messages and figure paths")
            yield messages, figure_paths
        
        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, [chatbot], [chatbot, figure_gallery]
        )
        clear.click(lambda: ([], "", []), None, [chatbot, msg, figure_gallery], queue=False)

if __name__ == "__main__":
    demo.launch(share=True)