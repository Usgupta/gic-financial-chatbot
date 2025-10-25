#!/usr/bin/env python3
"""
Vision Model Figure Extraction Test Script

This script uses DocLayout-YOLO to detect and extract figures, tables, and charts 
from PDF documents. It processes PDFs in the uploaded_pdfs/ directory and saves 
extracted figures as separate image files with metadata.

Usage:
    python test_figure_extraction.py                    # Process all PDFs in uploaded_pdfs/
    python test_figure_extraction.py path/to/file.pdf  # Process specific PDF

Integration Notes for main.py:
- This script demonstrates the figure extraction pipeline
- For integration: modify extract_text_by_page() to also extract figures
- Store figure embeddings in Qdrant alongside text embeddings
- Use multimodal retrieval (CLIP embeddings) for figure search
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import cv2
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
from doclayout_yolo import YOLOv10
import torch
import requests
import os

# Configuration
EXTRACTED_FIGURES_DIR = Path("extracted_figures")
UPLOADED_PDFS_DIR = Path("uploaded_pdfs")
MODEL_NAME = "doclayout_yolo_docstructbench_imgsz1024.pt"
CONFIDENCE_THRESHOLD = 0.25
IMAGE_SIZE = 1024

# Figure-related class labels in DocLayout-YOLO
FIGURE_CLASSES = ['figure', 'picture', 'chart', 'diagram', 'graph', 'plot']

def setup_directories():
    """Create necessary directories for output."""
    EXTRACTED_FIGURES_DIR.mkdir(exist_ok=True)
    print(f"Created output directory: {EXTRACTED_FIGURES_DIR}")

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
        sys.exit(1)

def load_model():
    """Load the DocLayout-YOLO model."""
    print("Loading DocLayout-YOLO model...")
    try:
        # Download model if not exists
        model_path = download_model()
        model = YOLOv10(model_path)
        print(f"Model loaded successfully: {MODEL_NAME}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have internet connection for model download")
        sys.exit(1)

def convert_pdf_to_images(pdf_path: Path) -> List[Image.Image]:
    """Convert PDF pages to PIL Images."""
    print(f"Converting PDF to images: {pdf_path.name}")
    try:
        images = convert_from_path(pdf_path, dpi=200)
        print(f"Converted {len(images)} pages to images")
        return images
    except Exception as e:
        print(f"Error converting PDF: {e}")
        return []

def detect_figures(model, image: Image.Image) -> List[Dict[str, Any]]:
    """Detect figures in a single page image."""
    # Convert PIL to numpy array for YOLO
    image_np = np.array(image)
    
    # Run detection
    results = model.predict(
        image_np, 
        imgsz=IMAGE_SIZE, 
        conf=CONFIDENCE_THRESHOLD,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        verbose=False
    )
    
    detections = []
    if results and len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        for i, box in enumerate(boxes):
            # Get class name
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            
            # Check if it's a figure-related class
            if class_name.lower() in FIGURE_CLASSES:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                
                detections.append({
                    'class_name': class_name,
                    'confidence': confidence,
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'detection_id': i
                })
    
    return detections

def extract_and_save_figures(image: Image.Image, detections: List[Dict[str, Any]], 
                           page_num: int, pdf_name: str) -> List[Dict[str, Any]]:
    """Extract and save detected figures."""
    saved_figures = []
    
    for idx, detection in enumerate(detections):
        x1, y1, x2, y2 = detection['bbox']
        
        # Crop the figure from the image
        figure_crop = image.crop((x1, y1, x2, y2))
        
        # Generate filename
        figure_filename = f"{pdf_name}_page{page_num+1}_figure{idx+1}_{detection['class_name']}.png"
        figure_path = EXTRACTED_FIGURES_DIR / figure_filename
        
        # Save the figure
        figure_crop.save(figure_path, "PNG")
        
        # Store metadata
        figure_metadata = {
            'filename': figure_filename,
            'page_number': page_num + 1,
            'class_name': detection['class_name'],
            'confidence': detection['confidence'],
            'bbox': detection['bbox'],
            'image_size': figure_crop.size,
            'pdf_name': pdf_name
        }
        
        saved_figures.append(figure_metadata)
        print(f"  Saved figure: {figure_filename} (confidence: {detection['confidence']:.3f})")
    
    return saved_figures

def process_pdf(pdf_path: Path, model) -> Dict[str, Any]:
    """Process a single PDF file."""
    print(f"\n{'='*60}")
    print(f"Processing PDF: {pdf_path.name}")
    print(f"{'='*60}")
    
    # Convert PDF to images
    images = convert_pdf_to_images(pdf_path)
    if not images:
        return {'error': 'Failed to convert PDF to images'}
    
    pdf_name = pdf_path.stem
    all_figures = []
    total_pages = len(images)
    
    # Process each page
    for page_num, image in enumerate(images):
        print(f"\nProcessing page {page_num + 1}/{total_pages}...")
        
        # Detect figures on this page
        detections = detect_figures(model, image)
        
        if detections:
            print(f"  Found {len(detections)} figures on page {page_num + 1}")
            # Extract and save figures
            saved_figures = extract_and_save_figures(image, detections, page_num, pdf_name)
            all_figures.extend(saved_figures)
        else:
            print(f"  No figures detected on page {page_num + 1}")
    
    # Save metadata
    metadata = {
        'pdf_name': pdf_name,
        'pdf_path': str(pdf_path),
        'total_pages': total_pages,
        'total_figures': len(all_figures),
        'figures': all_figures,
        'processing_timestamp': str(Path().cwd()),
        'model_used': MODEL_NAME,
        'confidence_threshold': CONFIDENCE_THRESHOLD
    }
    
    metadata_filename = f"{pdf_name}_metadata.json"
    metadata_path = EXTRACTED_FIGURES_DIR / metadata_filename
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nSummary for {pdf_name}:")
    print(f"  Pages processed: {total_pages}")
    print(f"  Figures extracted: {len(all_figures)}")
    print(f"  Metadata saved: {metadata_filename}")
    
    return metadata

def main():
    """Main function to process PDFs."""
    parser = argparse.ArgumentParser(description='Extract figures from PDFs using DocLayout-YOLO')
    parser.add_argument('pdf_path', nargs='?', help='Path to specific PDF file (optional)')
    args = parser.parse_args()
    
    print("Vision Model Figure Extraction Test Script")
    print("=" * 50)
    
    # Setup
    setup_directories()
    model = load_model()
    
    # Determine which PDFs to process
    if args.pdf_path:
        pdf_path = Path(args.pdf_path)
        if not pdf_path.exists():
            print(f"Error: PDF file not found: {pdf_path}")
            sys.exit(1)
        pdf_files = [pdf_path]
    else:
        # Process all PDFs in uploaded_pdfs directory
        if not UPLOADED_PDFS_DIR.exists():
            print(f"Error: Directory not found: {UPLOADED_PDFS_DIR}")
            sys.exit(1)
        
        pdf_files = list(UPLOADED_PDFS_DIR.glob("*.pdf"))
        if not pdf_files:
            print(f"No PDF files found in {UPLOADED_PDFS_DIR}")
            sys.exit(1)
        
        print(f"Found {len(pdf_files)} PDF files to process")
    
    # Process each PDF
    all_results = []
    total_figures = 0
    
    for pdf_file in pdf_files:
        result = process_pdf(pdf_file, model)
        if 'error' not in result:
            all_results.append(result)
            total_figures += result['total_figures']
    
    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"PDFs processed: {len(all_results)}")
    print(f"Total figures extracted: {total_figures}")
    print(f"Output directory: {EXTRACTED_FIGURES_DIR}")
    
    if total_figures > 0:
        print(f"\nExtracted figures are saved in: {EXTRACTED_FIGURES_DIR}")
        print("Each PDF has a corresponding metadata JSON file with detailed information.")
    
    print("\nIntegration Notes:")
    print("- Modify extract_text_by_page() in main.py to include figure extraction")
    print("- Store figure embeddings in Qdrant using CLIP or similar vision encoders")
    print("- Implement multimodal retrieval for combined text + figure search")

if __name__ == "__main__":
    main()
