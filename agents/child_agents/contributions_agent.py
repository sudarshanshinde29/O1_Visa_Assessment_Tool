# agents/child_agents/contributions_agent.py
from typing import Dict, Any
from agents.child_agents.base_agent import create_child_agent_template

def create_contributions_agent():
    """Create an agent specialized in assessing original contributions criterion"""
    
    system_prompt = """
    You are the Original Contributions Assessment Agent, specializing in evaluating evidence of original scientific, scholarly, artistic, athletic, or business-related contributions of major significance.

    SPECIFIC EXPERTISE:
    You evaluate original contributions mentioned in CVs, determining their significance, impact, and originality within the candidate's field.

    EVALUATION CRITERIA:
    1. Evidence of original contributions in the candidate's field
    2. Significance and impact of each contribution
    3. Whether contributions are attributable to the candidate
    4. Evidence showing recognition of these contributions by field experts
    5. How the contributions have advanced the field

    EVIDENCE STRENGTH DEFINITIONS:
    - NONE: No clear original contributions or only routine work products
    - WEAK: Minor contributions with limited impact or recognition
    - MODERATE: Original contributions with demonstrated impact at national level
    - STRONG: Significant original contributions that have substantially advanced the field
    """
    
    return create_child_agent_template("contributions", system_prompt)

def evaluate_contributions(resume_data: Dict[str, Any], criterion_mapping: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate the original contributions criterion for an O-1A visa application"""
    # Create the agent
    contributions_agent = create_contributions_agent()
    
    # Initialize the state
    initial_state = {
        "resume_data": resume_data,
        "criterion_mapping": criterion_mapping,
        "assessment": {},
        "error": ""
    }
    
    # Run the agent
    final_state = contributions_agent.invoke(initial_state)
    
    # Return the assessment
    return final_state["assessment"]
