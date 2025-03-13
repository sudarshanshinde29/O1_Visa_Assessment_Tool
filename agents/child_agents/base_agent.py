# agents/child_agents/base_agent.py
from typing import Dict, Any, Annotated, TypedDict
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
import langgraph as lg
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
import re
import os
from environs import Env
import json
from langchain_core.messages import HumanMessage

# Configure environment
env = Env()
env.read_env()  # Read .env file if it exists
GOOGLE_API_KEY = env("GOOGLE_API_KEY", None)

# Configure the API with your key
genai.configure(api_key=GOOGLE_API_KEY)


# Define the state for child agents
class ChildAgentState(TypedDict):
    resume_data: Dict[str, Any]
    criterion_mapping: Dict[str, Any]
    assessment: Dict[str, Any]
    error: str

def create_child_agent_template(criterion, system_prompt):
    """Create a child agent for a specific criterion"""
    
    # Initialize the Gemini model
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
    
    # Define the nodes in the graph
    def analyze_criterion(state: ChildAgentState) -> ChildAgentState:
        """Analyze the resume for the specific criterion"""
        try:
            resume_data = state["resume_data"]
            criterion_mapping = state["criterion_mapping"]
            
            # Create prompt combining system prompt and user instructions
            prompt = f"""
            {system_prompt}
            
            Please analyze this resume data for evidence of {criterion}.
            
            RESUME DATA:
            ```
            {json.dumps(resume_data, indent=2)}
            ```
            
            INITIAL CRITERION MAPPING:
            ```
            {json.dumps(criterion_mapping, indent=2)}
            ```
            
            Provide a detailed assessment of how the candidate meets or fails to meet this criterion.
            Include:
            1. All evidence items that support this criterion
            2. An analysis of the strength of each piece of evidence
            3. An overall assessment of the evidence strength (None, Weak, Moderate, Strong)
            4. Detailed justification for your assessment
            
            Format your response as JSON with the following structure:
            {{
              "criterion": "{criterion}",
              "evidence_items": [
                {{
                  "description": "Description of the evidence",
                  "source": "Location in resume",
                  "strength": "Weak|Moderate|Strong"
                }}
              ],
              "evidence_strength": "None|Weak|Moderate|Strong",
              "justification": "Detailed explanation of assessment"
            }}
            
            Return ONLY the JSON without any additional explanation.
            """
            
            # Get response from Gemini
            messages = [HumanMessage(content=prompt)]
            response = llm.invoke(messages)
            response_text = response.content.strip()
            
            # Look for JSON pattern in the response
            json_match = re.search(r'```(?:json)?(.*?)```', response_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(1)
            else:
                json_text = response_text.strip()
            
            # Clean up non-JSON content
            json_text = re.sub(r'^[^{]*', '', json_text)
            json_text = re.sub(r'[^}]*$', '', json_text)
            
            assessment = json.loads(json_text)
            
            return {
                "resume_data": resume_data,
                "criterion_mapping": criterion_mapping,
                "assessment": assessment,
                "error": ""
            }
        except Exception as e:
            return {
                "resume_data": state.get("resume_data", {}),
                "criterion_mapping": state.get("criterion_mapping", {}),
                "assessment": {},
                "error": f"Error analyzing {criterion}: {str(e)}"
            }
    
    def validate_assessment(state: ChildAgentState) -> ChildAgentState:
        """Validate the assessment and fix any issues"""
        if state["error"]:
            return state
        
        try:
            assessment = state["assessment"]
            
            # Ensure all required fields are present
            required_fields = ["criterion", "evidence_items", "evidence_strength", "justification"]
            for field in required_fields:
                if field not in assessment:
                    if field == "criterion":
                        assessment[field] = criterion
                    elif field == "evidence_items":
                        assessment[field] = []
                    elif field == "evidence_strength":
                        assessment[field] = "None"
                    elif field == "justification":
                        assessment[field] = f"No detailed justification provided for {criterion}."
            
            # Validate evidence_strength value
            valid_strengths = ["None", "Weak", "Moderate", "Strong"]
            if assessment["evidence_strength"] not in valid_strengths:
                # Default to "None" if invalid
                assessment["evidence_strength"] = "None"
            
            return {
                "resume_data": state["resume_data"],
                "criterion_mapping": state["criterion_mapping"],
                "assessment": assessment,
                "error": ""
            }
        except Exception as e:
            return {
                **state,
                "error": f"Error validating assessment: {str(e)}"
            }
    
    def handle_error(state: ChildAgentState) -> ChildAgentState:
        """Handle errors in the child agent process"""
        resume_data = state["resume_data"]
        criterion_mapping = state["criterion_mapping"]
        error = state["error"]
        
        # Create a default assessment with error information
        default_assessment = {
            "criterion": criterion,
            "evidence_items": [],
            "evidence_strength": "None",
            "justification": f"Error occurred during assessment: {error}"
        }
        
        return {
            "resume_data": resume_data,
            "criterion_mapping": criterion_mapping,
            "assessment": default_assessment,
            "error": ""  # Clear the error since we've handled it
        }
    
    # Define the edges of the graph
    def decide_next_step(state: ChildAgentState) -> str:
        """Decide the next step based on the current state"""
        if state["error"]:
            return "handle_error"
        return "validate_assessment"
    
    # Create the graph
    workflow = StateGraph(ChildAgentState)
    
    # Add nodes
    workflow.add_node("analyze_criterion", analyze_criterion)
    workflow.add_node("validate_assessment", validate_assessment)
    workflow.add_node("handle_error", handle_error)
    
    # Add edges
    workflow.add_conditional_edges(
        "analyze_criterion",
        decide_next_step,
        {
            "validate_assessment": "validate_assessment",
            "handle_error": "handle_error"
        }
    )
    workflow.add_edge("validate_assessment", END)
    workflow.add_edge("handle_error", END)
    
    # Set entry point
    workflow.set_entry_point("analyze_criterion")
    
    # Compile the graph
    child_agent = workflow.compile()
    
    return child_agent
