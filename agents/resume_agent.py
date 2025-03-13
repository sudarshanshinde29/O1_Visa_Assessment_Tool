# agents/resume_agent.py
from typing import Dict, Any, Annotated, TypedDict
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import langgraph as lg
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
import google.generativeai as genai
import os
from environs import Env
import json
import re

# Configure environment
env = Env()
env.read_env()  # Read .env file if it exists
GOOGLE_API_KEY = env("GOOGLE_API_KEY", None)

# Configure the API with your key
genai.configure(api_key=GOOGLE_API_KEY)

# Define state for the Resume Structuring Agent
class ResumeAgentState(TypedDict):
    raw_text: str
    structured_resume: Dict[str, Any]
    error: str
    retry_count: int

# Define structured resume schema
class StructuredResume(BaseModel):
    personalInfo: Dict[str, Any] = Field(
        description="Basic information about the candidate including name, contact details, etc."
    )
    education: list = Field(
        description="Academic background with degrees, institutions, dates, and relevant details"
    )
    workExperience: list = Field(
        description="Professional history with company names, titles, dates, responsibilities, and achievements"
    )
    publications: list = Field(
        description="Scholarly works with titles, publication venues, dates, and citation information"
    )
    awards: list = Field(
        description="Recognitions received with names, dates, issuers, and significance"
    )
    memberships: list = Field(
        description="Professional association memberships with organization names and details"
    )
    pressAndMedia: list = Field(
        description="Media coverage about the candidate with publication names, dates, and descriptions"
    )
    judgingExperience: list = Field(
        description="Experience judging the work of others including roles, organizations, and details"
    )
    contributions: list = Field(
        description="Original contributions to the field with descriptions and significance"
    )
    skills: list = Field(
        description="Technical and professional capabilities relevant to the candidate's field"
    )
    additionalInfo: Dict[str, Any] = Field(
        description="Other relevant professional activities or information"
    )

# Create the Resume Structuring Agent
def create_resume_structuring_agent(model_name: str = "gemini-2.0-flash"):
    # Initialize the LLM with proper error handling
    try:
        llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.7)
        print(f"Initialized {model_name}")
    except Exception as e:
        print(f"Error initializing {model_name}: {str(e)}")
        llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
        print("Falling back to gemini-pro")
    
    # Define the system prompt
    system_prompt = """
    You are the Resume Structuring Agent for an O-1A visa assessment system. Your task is to transform raw text extracted from a resume into a clean, well-structured JSON format.

    YOUR TASKS:
    1. Organize raw resume text into standard resume sections
    2. Normalize dates, titles, and organization names
    3. Identify and structure accomplishments within each role/position
    4. Extract key skills, technologies, and domain expertise
    5. Preserve all relevant information while removing formatting artifacts
    6. Handle incomplete or ambiguous information appropriately
    7. Make reasonable inferences when information is implied but not explicit

    Pay special attention to extracting information that may be relevant to O-1A visa criteria:
    - Awards and recognitions
    - Professional memberships
    - Media coverage about the candidate
    - Judging or reviewer roles
    - Original contributions to their field
    - Publications and scholarly articles
    - Leadership or critical roles in organizations
    - Salary or compensation information

    Ensure your output is comprehensive and well-structured as it will be used for downstream analysis.
    """

    # Define the nodes in the graph
    def preprocess_resume(state: ResumeAgentState) -> ResumeAgentState:
        """Preprocess the raw resume text for better structuring."""
        raw_text = state["raw_text"]
        
        # Implement any necessary text preprocessing here
        # For example, removing extraneous whitespace, fixing common OCR issues, etc.
        processed_text = raw_text.strip().replace("\n\n\n", "\n\n")
        
        return {"raw_text": processed_text, "structured_resume": {}, "error": "", "retry_count": 0}



    def extract_json_from_response(response_text: str) -> dict:
        # If it's clean JSON already, just parse it
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass  # Try to clean it up below
        
        # Remove any code fences using regex
        cleaned_text = re.sub(r"```(json)?", "", response_text).strip().strip("`").strip()
        
        # Try loading again
        try:
            return json.loads(cleaned_text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e}\nContent: {cleaned_text}")


    def structure_resume(state: ResumeAgentState) -> ResumeAgentState:
        """Structure the resume text into a standardized JSON format."""
        try:
            raw_text = state["raw_text"]
            
            # Create prompt for the LLM
            user_prompt = f"""
            Please convert the following resume text into a structured JSON format that matches the provided schema.
            Extract all relevant information and organize it into the appropriate sections.ONLY return a JSON object. DO NOT include markdown, code fences, or explanations.
            
            RESUME TEXT:
            {raw_text}
            
            OUTPUT SCHEMA:
            ```
            {StructuredResume.schema_json(indent=2)}
            ```
            
            Return ONLY the JSON object without any additional explanations or markdown formatting.
            """
            
            # Get response from LLM
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            # Get response from LLM - Gemini format
            response = llm.invoke(messages)

            # Extract JSON from response
            response_text = response.content
            print("Raw response content:", response.content)

            # Use the cleaned JSON extractor
            structured_resume = extract_json_from_response(response_text)
            
            # structured_resume = json.loads(json_text)
            
            return {"raw_text": raw_text, "structured_resume": structured_resume, "error": "", "retry_count": state.get("retry_count", 0)}
        except Exception as e:
            return {"raw_text": raw_text, "structured_resume": {}, "error": str(e), "retry_count": state.get("retry_count", 0)}

    def validate_resume_structure(state: ResumeAgentState) -> ResumeAgentState:
        """Validate the structured resume and fix any issues."""
        # Don't just pass through error states
        structured_resume = state.get("structured_resume", {})
        
        # Ensure all required fields are present
        required_fields = [
            "personalInfo", "education", "workExperience", "publications", 
            "awards", "memberships", "pressAndMedia", "judgingExperience",
            "contributions", "skills", "additionalInfo"
        ]
        
        for field in required_fields:
            if field not in structured_resume:
                structured_resume[field] = [] if field != "personalInfo" and field != "additionalInfo" else {}
        
        return {
            "raw_text": state["raw_text"],
            "structured_resume": structured_resume,
            "error": "",  # Clear any error after validation
            "retry_count": state.get("retry_count", 0)
        }

    # Define the edges of the graph
    def decide_next_step(state: ResumeAgentState) -> str:
        """Decide the next step based on the current state."""
        if state["error"]:
            return "handle_error"
        return "validate_resume_structure"

    def should_end(state: ResumeAgentState) -> bool:
        """Determine if the process should end."""
        # Add a counter to the state to track retry attempts
        if "retry_count" not in state:
            state["retry_count"] = 0
        else:
            state["retry_count"] += 1
        
        # End if no error or if we've tried too many times
        return not state["error"] or state["retry_count"] > 3

    def handle_error(state: ResumeAgentState) -> ResumeAgentState:
        """Handle errors in the resume structuring process."""
        raw_text = state["raw_text"]
        error = state["error"]
        retry_count = state.get("retry_count", 0)
        
        # Try to recover from the error with a simpler approach
        try:
            # Create a more direct prompt for the LLM
            user_prompt = f"""
            I encountered an error while processing this resume: {error}
            
            Please convert the following resume text into a simpler JSON format with these fields:
            - personalInfo (object with name, contact, etc.)
            - education (array of education items)
            - workExperience (array of work items)
            - publications (array of publications)
            - awards (array of awards)
            - other (any other relevant information)
            
            RESUME TEXT:
            {raw_text}
            
            Return ONLY the JSON object.
            """
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            response = llm.invoke(user_prompt)
            
            # Extract JSON from response with improved error handling
            response_text = response.content
            
            if "```json" in response_text:
                json_text = response_text.split("``````")[0].strip()
            elif "``` " in response_text:
                # Find content between first and last triple backticks
                parts = response_text.split("```") # Split on triple backticks
                if len(parts) >= 3:  # At least one complete code block
                    json_text = parts[1].strip() 
                else:
                    json_text = response_text.strip() 
            else:
                json_text = response_text.strip()
            
            simplified_resume = json.loads(json_text)
            
            # Expand the simplified resume to match our schema
            structured_resume = {
                "personalInfo": simplified_resume.get("personalInfo", {}),
                "education": simplified_resume.get("education", []),
                "workExperience": simplified_resume.get("workExperience", []),
                "publications": simplified_resume.get("publications", []),
                "awards": simplified_resume.get("awards", []),
                "memberships": [],
                "pressAndMedia": [],
                "judgingExperience": [],
                "contributions": [],
                "skills": simplified_resume.get("skills", []),
                "additionalInfo": simplified_resume.get("other", {})
            }
            
            return {
                "raw_text": raw_text,
                "structured_resume": structured_resume,
                "error": "",
                "retry_count": retry_count + 1
            }
        except Exception as e:
            # If recovery fails, return an error and a minimal structure
            return {
                "raw_text": raw_text,
                "structured_resume": {
                    "personalInfo": {},
                    "education": [],
                    "workExperience": [],
                    "publications": [],
                    "awards": [],
                    "memberships": [],
                    "pressAndMedia": [],
                    "judgingExperience": [],
                    "contributions": [],
                    "skills": [],
                    "additionalInfo": {"error": f"Failed to structure resume: {str(e)}"}
                },
                "error": f"Failed to recover from error: {str(e)}",
                "retry_count": retry_count + 1
            }

    # Create the graph
    workflow = StateGraph(ResumeAgentState)
    
    # Add nodes
    workflow.add_node("preprocess_resume", preprocess_resume)
    workflow.add_node("structure_resume", structure_resume)
    workflow.add_node("validate_resume_structure", validate_resume_structure)
    workflow.add_node("handle_error", handle_error)
    
    # Add edges
    workflow.add_edge("preprocess_resume", "structure_resume")
    workflow.add_conditional_edges(
        "structure_resume",
        decide_next_step,
        {
            "validate_resume_structure": "validate_resume_structure",
            "handle_error": "handle_error"
        }
    )
    # Instead of unconditionally going back to validate, have handle_error go to END if it can't fix the error
    workflow.add_conditional_edges(
        "handle_error",
        lambda state: not state["error"],  # Check if error was fixed
        {
            True: "validate_resume_structure",
            False: END  # End if couldn't fix the error
        }
    )
    workflow.add_edge("validate_resume_structure", END)  # Always end after validation
        
    # Set entry point
    workflow.set_entry_point("preprocess_resume")
    
    # Compile the graph
    resume_agent = workflow.compile()
    
    return resume_agent

# Function to process a resume
def process_resume(raw_text: str) -> Dict[str, Any]:
    """Process a resume from raw text to structured format."""
    # Create the agent
    resume_agent = create_resume_structuring_agent()
    
    # Initialize the state
    initial_state = {"raw_text": raw_text, "structured_resume": {}, "error": "", "retry_count": 0}
    
    # Run the agent
    final_state = resume_agent.invoke(initial_state)
    
    # Return the structured resume
    return final_state["structured_resume"]
