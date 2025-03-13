# agents/child_agents/press_agent.py
from typing import Dict, Any
from agents.child_agents.base_agent import create_child_agent_template

def create_press_agent():
    """Create an agent specialized in assessing press coverage criterion"""
    
    system_prompt = """
    You are the Press Coverage Assessment Agent, specializing in evaluating evidence of published material about the candidate in professional or major media.

    SPECIFIC EXPERTISE:
    You evaluate mentions of press coverage, media appearances, and published material about candidates in CVs.

    EVALUATION CRITERIA:
    1. Published material about the candidate
    2. Publication's prestige, circulation, and recognition
    3. Whether the material is specifically about the candidate and their work
    4. Whether the material focuses on achievements in their field
    5. Depth, quality, and prominence of the coverage

    EVIDENCE STRENGTH DEFINITIONS:
    - NONE: No press coverage or only self-published/company newsletter mentions
    - WEAK: Coverage in local media or minor industry publications
    - MODERATE: Coverage in recognized national publications or well-known industry journals
    - STRONG: Significant coverage in major international media or top-tier professional publications
    """
    
    return create_child_agent_template("press", system_prompt)

def evaluate_press(resume_data: Dict[str, Any], criterion_mapping: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate the press coverage criterion for an O-1A visa application"""
    # Create the agent
    press_agent = create_press_agent()
    
    # Initialize the state
    initial_state = {
        "resume_data": resume_data,
        "criterion_mapping": criterion_mapping,
        "assessment": {},
        "error": ""
    }
    
    # Run the agent
    final_state = press_agent.invoke(initial_state)
    
    # Return the assessment
    return final_state["assessment"]
