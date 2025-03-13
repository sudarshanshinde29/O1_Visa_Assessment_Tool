# utils/document_processor.py
import requests
import io
import os
from typing import BinaryIO, Optional, Union
from environs import Env

# Load environment variables
env = Env()
env.read_env()  # Read .env file if it exists
MISTRAL_API_KEY = env("MISTRAL_API_KEY", None)  # Allow fallback to None if not set

def extract_text_with_mistral_ocr(
    file_input: Union[BinaryIO, str], 
    is_url: bool = False
) -> str:
    """
    Extract text from a document using Mistral's OCR API.
    
    Args:
        file_input: Either a file-like object or a URL string
        is_url: Flag indicating if the input is a URL
        
    Returns:
        Extracted text in markdown format
    """
    if not MISTRAL_API_KEY:
        raise ValueError("MISTRAL_API_KEY environment variable is not set")
    
    url = "https://api.mistral.ai/v1/ocr"
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Prepare the payload based on input type
    if is_url:
        payload = {
            "model": "mistral-ocr-latest",
            "id": "mistral-ocr-latest",
            "document": {
                "type": "document_url",
                "document_url": file_input,
                "document_name": "resume.pdf",
            }
        }
        response = requests.post(url, json=payload, headers=headers)
    else:
        # For file uploads, we need to use multipart/form-data
        # First, prepare multipart form with the file
        file_content = file_input.read()
        files = {
            'file': ('resume.pdf', file_content, 'application/pdf')
        }
        data = {
            'model': 'mistral-ocr-2503',
        }
        # Remove content-type from headers for multipart/form-data
        upload_headers = {
            "Authorization": f"Bearer {MISTRAL_API_KEY}"
        }
        response = requests.post(url, files=files, data=data, headers=upload_headers)
    
    # Process the response
    if response.status_code == 200:
        result = response.json()
        
        # Compile all the markdown content from each page
        full_text = ""
        if result.get('pages') and len(result.get('pages')) > 0:
            for page in result['pages']:
                # We're not including page markers in the text since
                # we want continuous text for the resume analysis
                full_text += page.get('markdown', '') + "\n\n"
        return full_text
    else:
        error_msg = f"OCR API Error {response.status_code}: {response.text}"
        raise Exception(error_msg)

def extract_text_from_pdf(file_object: BinaryIO) -> str:
    """
    Extract text from a PDF file using Mistral OCR.
    Falls back to PyPDF if Mistral OCR fails or API key is not set.
    
    Args:
        file_object: File-like object containing the PDF
        
    Returns:
        Extracted text
    """
    try:
        # Try to use Mistral OCR
        if MISTRAL_API_KEY:
            # Reset file pointer to beginning
            file_object.seek(0)
            return extract_text_with_mistral_ocr(file_object)
        else:
            raise ValueError("MISTRAL_API_KEY not set, falling back to PyPDF")
    except Exception as e:
        # Fallback to PyPDF if Mistral OCR fails
        print(f"Mistral OCR failed: {str(e)}. Falling back to PyPDF.")
        
        # Reset file pointer to beginning
        file_object.seek(0)
        
        # Use PyPDF as fallback
        import pypdf
        text = ""
        pdf_reader = pypdf.PdfReader(file_object)
        
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n\n"
        
        return text

def extract_text_from_url(url: str) -> str:
    """
    Extract text from a PDF at a given URL using Mistral OCR.
    
    Args:
        url: URL pointing to a PDF document
        
    Returns:
        Extracted text
    """
    try:
        # Use Mistral OCR with URL
        if MISTRAL_API_KEY:
            return extract_text_with_mistral_ocr(url, is_url=True)
        else:
            raise ValueError("MISTRAL_API_KEY not set")
    except Exception as e:
        # For URLs, we need to download the file first for PyPDF fallback
        print(f"Mistral OCR with URL failed: {str(e)}. Downloading file for PyPDF.")
        
        # Download the file
        response = requests.get(url)
        if response.status_code == 200:
            # Create in-memory file-like object
            file_object = io.BytesIO(response.content)
            
            # Use PyPDF
            import pypdf
            text = ""
            pdf_reader = pypdf.PdfReader(file_object)
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n\n"
            print ("text", text)
            return text
        else:
            raise Exception(f"Failed to download PDF from URL: {response.status_code}")
