# agents/child_agents/remuneration_agent.py
from typing import Dict, Any
from agents.child_agents.base_agent import create_child_agent_template

def create_remuneration_agent():
    """Create an agent specialized in assessing remuneration criterion"""
    
    system_prompt = """
    You are the High Remuneration Assessment Agent, specializing in evaluating evidence of command of a high salary or other substantial remuneration relative to others in the field.

    SPECIFIC EXPERTISE:
    You evaluate compensation information in CVs, determining if it demonstrates high remuneration compared to industry standards.

    EVALUATION CRITERIA:
    1. Evidence of high salary or other substantial remuneration
    2. Compensation relative to others in the same field
    3. Salary information through explicit statements or inferences
    4. Additional compensation forms (bonuses, equity, etc.)
    5. Geographic and industry-specific context

    EVIDENCE STRENGTH DEFINITIONS:
    - NONE: No salary information or evidence of only average compensation
    - WEAK: Slightly above-average compensation or insufficient information
    - MODERATE: Clearly above-average compensation relative to field peers
    - STRONG: Exceptional compensation significantly higher than typical for the field
    """
    
    return create_child_agent_template("remuneration", system_prompt)

def evaluate_remuneration(resume_data: Dict[str, Any], criterion_mapping: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate the high remuneration criterion for an O-1A visa application"""
    # Create the agent
    remuneration_agent = create_remuneration_agent()
    
    # Initialize the state
    initial_state = {
        "resume_data": resume_data,
        "criterion_mapping": criterion_mapping,
        "assessment": {},
        "error": ""
    }
    
    # Run the agent
    final_state = remuneration_agent.invoke(initial_state)
    
    # Return the assessment
    return final_state["assessment"]
