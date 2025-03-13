# agents/mapping_agent.py
from typing import Dict, Any, List, Annotated, TypedDict, Optional
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import langgraph as lg
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
import google.generativeai as genai
import os
from environs import Env

# Configure environment
env = Env()
env.read_env()  # Read .env file if it exists
GOOGLE_API_KEY = env("GOOGLE_API_KEY", None)

# Configure the API with your key
genai.configure(api_key=GOOGLE_API_KEY)

# Define state for the Experience Mapping Agent
class MappingAgentState(TypedDict):
    structured_resume: Dict[str, Any]
    criteria_mapping: Dict[str, Any]
    error: str

# Define structure for mapped criteria
class CriterionEvidence(BaseModel):
    criterion: str = Field(description="Name of the O-1A criterion")
    relevantItems: List[Dict[str, Any]] = Field(description="List of resume elements matching this criterion")
    context: str = Field(description="Additional context needed to understand these elements")
    potentialStrength: str = Field(description="Initial assessment of evidence strength (weak, moderate, strong)")

# Create the Experience Mapping Agent
def create_experience_mapping_agent(model_name: str = "gemini-2.0-flash"):
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
    You are the Experience Mapping Agent for an O-1A visa assessment system. Your task is to analyze a structured resume and map specific experiences to the 8 O-1A visa criteria.

    THE 8 O-1A CRITERIA:
    1. Awards: National or international prizes/awards for excellence
    2. Membership: Membership in associations requiring outstanding achievement
    3. Press: Published material about the applicant in professional/major media
    4. Judging: Evidence of judging the work of others in the field
    5. Contributions: Original scientific, scholarly, or business contributions
    6. Articles: Authorship of scholarly articles in professional publications
    7. Employment: Employment in a critical capacity at distinguished organizations
    8. Remuneration: Evidence of high salary or remuneration

    YOUR TASKS:
    1. Review the structured resume thoroughly
    2. Identify elements relevant to each of the 8 O-1A criteria
    3. Create a mapping between resume elements and criteria
    4. Make reasonable inferences about which experiences might qualify
    5. Ensure no potentially relevant information is overlooked
    6. Provide an initial assessment of evidence strength for each criterion

    EVIDENCE STRENGTH DEFINITIONS:
    - NONE: No evidence found for this criterion
    - WEAK: Limited evidence with minimal significance or recognition
    - MODERATE: Reasonable evidence with some significance or recognition
    - STRONG: Substantial evidence with significant recognition or impact

    Your mapping will be passed to specialized agents for each criterion who will conduct in-depth analysis.
    """

    # Define the nodes in the graph
    def map_experiences(state: MappingAgentState) -> MappingAgentState:
        """Map resume experiences to O-1A criteria."""
        try:
            structured_resume = state["structured_resume"]
            
            # Create prompt for the LLM
            import json
            user_prompt = f"""
            Please analyze this structured resume and map specific elements to each of the 8 O-1A visa criteria.
            
            STRUCTURED RESUME:
            ```
            {json.dumps(structured_resume, indent=2)}
            ```
            
            For each of the 8 O-1A criteria:
            1. Identify all resume elements that potentially satisfy the criterion
            2. Provide relevant context to understand the significance
            3. Assess the potential strength of the evidence (None, Weak, Moderate, Strong)
            
            Format your response as a JSON object with the following structure:
            ```
            {{
              "awards": {{
                "criterion": "Awards",
                "relevantItems": [{{...}}],
                "context": "Explanation of significance...",
                "potentialStrength": "weak|moderate|strong"
              }},
              "membership": {{...}},
              "press": {{...}},
              "judging": {{...}},
              "contributions": {{...}},
              "articles": {{...}},
              "employment": {{...}},
              "remuneration": {{...}}
            }}
            ```
            
            Return ONLY the JSON object without any additional explanations.DO NOT include markdown, code fences, or explanations.
            """
            
            # Get response from LLM
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            response = llm.invoke(messages)
            
            # Extract JSON from response
            response_text = response.content
            if "```" in response_text:
                json_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_text = response_text.split("``````")[0].strip()
            else:
                json_text = response_text.strip()
            
            criteria_mapping = json.loads(json_text)
            
            return {
                "structured_resume": structured_resume,
                "criteria_mapping": criteria_mapping,
                "error": ""
            }
        except Exception as e:
            return {
                "structured_resume": structured_resume,
                "criteria_mapping": {},
                "error": str(e)
            }

    def validate_mapping(state: MappingAgentState) -> MappingAgentState:
        """Validate the criteria mapping and fix any issues."""
        if state["error"]:
            return state
        
        criteria_mapping = state["criteria_mapping"]
        structured_resume = state["structured_resume"]
        
        # Ensure all 8 criteria are present
        required_criteria = [
            "awards", "membership", "press", "judging", 
            "contributions", "articles", "employment", "remuneration"
        ]
        
        for criterion in required_criteria:
            if criterion not in criteria_mapping:
                criteria_mapping[criterion] = {
                    "criterion": criterion.capitalize(),
                    "relevantItems": [],
                    "context": "No relevant evidence found",
                    "potentialStrength": "None"
                }
        
        return {
            "structured_resume": structured_resume,
            "criteria_mapping": criteria_mapping,
            "error": ""
        }

    def enhance_mapping(state: MappingAgentState) -> MappingAgentState:
        """Enhance the mapping with additional insights and connections."""
        if state["error"]:
            return state
        
        structured_resume = state["structured_resume"]
        criteria_mapping = state["criteria_mapping"]
        
        # Create a prompt to enhance the mapping
        import json
        user_prompt = f"""
        I have an initial mapping of resume elements to O-1A criteria, but I need you to enhance it by:
        
        1. Looking for additional connections or evidence that might have been missed
        2. Identifying cross-criterion relevance (e.g., when one achievement supports multiple criteria)
        3. Providing more detailed context about how each element satisfies its criterion
        
        STRUCTURED RESUME:
        ```
        {json.dumps(structured_resume, indent=2)}
        ```
        
        INITIAL MAPPING:
        ```
        {json.dumps(criteria_mapping, indent=2)}
        ```
        
        Analyze the resume again and provide an enhanced version of the mapping with the same structure.
        Focus particularly on:
        - Finding overlooked evidence in the resume
        - Strengthening the context explanations
        - Ensuring consistent evaluation of evidence strength
        
        Return ONLY the enhanced JSON object.
        """
        
        try:
            # Get response from LLM
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            llm = genai.GenerativeModel(model_name="gemini-2.0-flash")
            response = llm.invoke(messages)
            
            # Extract JSON from response
            response_text = response.content
            if "```" in response_text:
                json_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_text = response_text.split("``````")[0].strip()
            else:
                json_text = response_text.strip()
            
            enhanced_mapping = json.loads(json_text)
            
            return {
                "structured_resume": structured_resume,
                "criteria_mapping": enhanced_mapping,
                "error": ""
            }
        except Exception as e:
            # If enhancement fails, keep the original mapping
            return {
                "structured_resume": structured_resume,
                "criteria_mapping": criteria_mapping,
                "error": f"Enhancement failed but using original mapping: {str(e)}"
            }

    # Define the edges of the graph
    def decide_next_step(state: MappingAgentState) -> str:
        """Decide the next step based on the current state."""
        if state["error"]:
            return "handle_error"
        return "validate_mapping"

    def should_enhance(state: MappingAgentState) -> bool:
        """Determine if the mapping should be enhanced."""
        # Count items with moderate or strong evidence
        count = 0
        for criterion, data in state["criteria_mapping"].items():
            strength = data.get("potentialStrength", "").lower()
            if strength in ["moderate", "strong"]:
                count += 1
        
        # If fewer than 3 criteria have moderate/strong evidence, try to enhance
        return count < 3

    def handle_error(state: MappingAgentState) -> MappingAgentState:
        """Handle errors in the mapping process."""
        structured_resume = state["structured_resume"]
        error = state["error"]
        
        # Create a simplified mapping as fallback
        try:
            # Initialize empty criteria mapping
            criteria_mapping = {}
            criteria_names = [
                "awards", "membership", "press", "judging", 
                "contributions", "articles", "employment", "remuneration"
            ]
            
            for criterion in criteria_names:
                criteria_mapping[criterion] = {
                    "criterion": criterion.capitalize(),
                    "relevantItems": [],
                    "context": "Error occurred during mapping",
                    "potentialStrength": "None"
                }
            
            # Try to extract some basic mappings
            if "awards" in structured_resume and structured_resume["awards"]:
                criteria_mapping["awards"]["relevantItems"] = structured_resume["awards"]
                criteria_mapping["awards"]["context"] = "Awards extracted from resume"
                criteria_mapping["awards"]["potentialStrength"] = "Weak"
            
            if "publications" in structured_resume and structured_resume["publications"]:
                criteria_mapping["articles"]["relevantItems"] = structured_resume["publications"]
                criteria_mapping["articles"]["context"] = "Publications extracted from resume"
                criteria_mapping["articles"]["potentialStrength"] = "Weak"
                
            if "memberships" in structured_resume and structured_resume["memberships"]:
                criteria_mapping["membership"]["relevantItems"] = structured_resume["memberships"]
                criteria_mapping["membership"]["context"] = "Memberships extracted from resume"
                criteria_mapping["membership"]["potentialStrength"] = "Weak"
            
            if "workExperience" in structured_resume and structured_resume["workExperience"]:
                criteria_mapping["employment"]["relevantItems"] = structured_resume["workExperience"]
                criteria_mapping["employment"]["context"] = "Work experience extracted from resume"
                criteria_mapping["employment"]["potentialStrength"] = "Weak"
            
            return {
                "structured_resume": structured_resume,
                "criteria_mapping": criteria_mapping,
                "error": ""
            }
        except Exception as e:
            # If everything fails, return an empty mapping with the error
            empty_mapping = {}
            for criterion in ["awards", "membership", "press", "judging", "contributions", "articles", "employment", "remuneration"]:
                empty_mapping[criterion] = {
                    "criterion": criterion.capitalize(),
                    "relevantItems": [],
                    "context": f"Failed to map: {str(e)}",
                    "potentialStrength": "None"
                }
            
            return {
                "structured_resume": structured_resume,
                "criteria_mapping": empty_mapping,
                "error": f"Failed to recover from error: {str(e)}"
            }

    # Create the graph
    workflow = StateGraph(MappingAgentState)
    
    # Add nodes
    workflow.add_node("map_experiences", map_experiences)
    workflow.add_node("validate_mapping", validate_mapping)
    workflow.add_node("enhance_mapping", enhance_mapping)
    workflow.add_node("handle_error", handle_error)
    
    # Add edges
    workflow.add_conditional_edges(
        "map_experiences",
        decide_next_step,
        {
            "validate_mapping": "validate_mapping",
            "handle_error": "handle_error"
        }
    )
    workflow.add_edge("handle_error", "validate_mapping")
    
    workflow.add_conditional_edges(
        "validate_mapping",
        should_enhance,
        {
            True: "enhance_mapping",
            False: END
        }
    )
    
    workflow.add_edge("enhance_mapping", END)
    
    # Set entry point
    workflow.set_entry_point("map_experiences")
    
    # Compile the graph
    mapping_agent = workflow.compile()
    
    return mapping_agent

# Function to map resume experiences to O-1A criteria
def map_resume_to_criteria(structured_resume: Dict[str, Any]) -> Dict[str, Any]:
    """Map a structured resume to O-1A criteria."""
    # Create the agent
    mapping_agent = create_experience_mapping_agent()
    
    # Initialize the state
    initial_state = {
        "structured_resume": structured_resume,
        "criteria_mapping": {},
        "error": ""
    }
    
    # Run the agent
    final_state = mapping_agent.invoke(initial_state)
    
    # Return the criteria mapping
    return final_state["criteria_mapping"]
