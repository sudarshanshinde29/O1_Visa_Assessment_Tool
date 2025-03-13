# app.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import uvicorn
from io import BytesIO
from typing import Dict, Any, Optional
from agents.agent_manager import AgentManager
from agents.resume_agent import process_resume
from agents.mapping_agent import map_resume_to_criteria
from utils.document_processor import extract_text_from_pdf, extract_text_from_url
from pydantic import BaseModel
import logging
from agents.agent_manager import AgentManager



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="O-1A Visa Assessment API")

# Instantiate once, maybe at the module level:
agent_manager = AgentManager()

class URLInput(BaseModel):
    url: str

@app.post("/process-resume/")
async def process_resume_endpoint(file: UploadFile = File(...)):
    """
    Process a resume PDF and extract structured information.
    """
    try:
        # Check if file is a PDF
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Read the file
        contents = await file.read()
        
        # Extract text from PDF
        raw_text = extract_text_from_pdf(BytesIO(contents))
        
        # Process the resume
        structured_resume = process_resume(raw_text)
        
        return JSONResponse(content={"structured_resume": structured_resume})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing resume: {str(e)}")

@app.post("/process-resume-from-url/")
async def process_resume_from_url(input_data: URLInput):
    """
    Process a resume PDF from a URL and extract structured information.
    """
    try:
        url = input_data.url
        logger.info(f"Processing URL: {url}")
        # Extract text from PDF URL
        raw_text = extract_text_from_url(url)
        logger.info("Extracted text from URL")
        # Process the resume
        structured_resume = process_resume(raw_text)
        logger.info("Processed resume text into structured data")
        return JSONResponse(content={"structured_resume": structured_resume})
    
    except Exception as e:
        logger.error(f"Error processing resume from URL: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing resume from URL: {str(e)}")

@app.post("/map-criteria/")
async def map_criteria_endpoint(structured_resume: Dict[str, Any]):
    """
    Map structured resume data to O-1A criteria.
    """
    try:
        # Map resume to criteria
        criteria_mapping = map_resume_to_criteria(structured_resume)
        
        return JSONResponse(content={"criteria_mapping": criteria_mapping})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error mapping criteria: {str(e)}")

@app.post("/process-and-map/")
async def process_and_map_endpoint(file: UploadFile = File(...)):
    """
    Process a resume PDF and map to O-1A criteria in one step.
    """
    try:
        # Check if file is a PDF
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Read the file
        contents = await file.read()
        
        # Extract text from PDF
        raw_text = extract_text_from_pdf(BytesIO(contents))
        
        # Process the resume
        structured_resume = process_resume(raw_text)
        
        # Map resume to criteria
        criteria_mapping = map_resume_to_criteria(structured_resume)
        
        return JSONResponse(content={
            "structured_resume": structured_resume,
            "criteria_mapping": criteria_mapping
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing and mapping: {str(e)}")

@app.post("/process-and-map-from-url/")
async def process_and_map_from_url_endpoint(input_data: URLInput):
    """
    Process a resume PDF from a URL and map to O-1A criteria in one step.
    """
    try:
        url = input_data.url
        
        # Extract text from PDF URL
        raw_text = extract_text_from_url(url)
        
        # Process the resume
        structured_resume = process_resume(raw_text)
        
        # Map resume to criteria
        criteria_mapping = map_resume_to_criteria(structured_resume)
        
        return JSONResponse(content={
            "structured_resume": structured_resume,
            "criteria_mapping": criteria_mapping
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing and mapping from URL: {str(e)}")


@app.post("/full-assessment/")
async def full_assessment(file: UploadFile = File(...)):
    try:
        # Process document
        contents = await file.read()
        raw_text = extract_text_from_pdf(BytesIO(contents))
        
        # Structure resume
        structured_resume = process_resume(raw_text)
        
        # Map criteria
        criteria_mapping = map_resume_to_criteria(structured_resume)
        
        # Coordinate assessment with agent manager (which handles all agents)
        result = agent_manager.coordinate_assessment(
            structured_resume,
            criteria_mapping
        )
        
        return JSONResponse(content={
            "structured_resume": structured_resume,
            "criteria_mapping": criteria_mapping,
            "assessment_result": result
        })
    
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.get("/agent-status/")
async def get_agent_status():
    """Check the status of all agents in the system."""
    try:
        status = agent_manager.get_all_agents_status()
        return JSONResponse(content={"status": status})
    except Exception as e:
        raise HTTPException(500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
