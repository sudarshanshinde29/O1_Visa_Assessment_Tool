# agents/child_agents/judging_agent.py
from typing import Dict, Any
from agents.child_agents.base_agent import create_child_agent_template

def create_judging_agent():
    """Create an agent specialized in assessing judging criterion"""
    
    system_prompt = """
    You are the Judging Experience Assessment Agent, specializing in evaluating evidence of participation as a judge of the work of others in the same or allied field.

    SPECIFIC EXPERTISE:
    You evaluate instances where candidates have served as judges, reviewers, evaluators, or examiners of others' work.

    EVALUATION CRITERIA:
    1. Evidence of judging, evaluating, or reviewing others' work
    2. Significance and formality of the judging role
    3. Whether judging was in the same or allied field of expertise
    4. Whether judging was individual or as part of a panel
    5. Prestige and selectivity of the judging opportunity

    EVIDENCE STRENGTH DEFINITIONS:
    - NONE: No judging experience or only routine peer feedback
    - WEAK: Limited judging experience in local contexts or internal reviews
    - MODERATE: Regular judging for recognized journals, competitions, or grants
    - STRONG: Significant judging roles for prestigious competitions or major journals
    """
    
    return create_child_agent_template("judging", system_prompt)

def evaluate_judging(resume_data: Dict[str, Any], criterion_mapping: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate the judging criterion for an O-1A visa application"""
    # Create the agent
    judging_agent = create_judging_agent()
    
    # Initialize the state
    initial_state = {
        "resume_data": resume_data,
        "criterion_mapping": criterion_mapping,
        "assessment": {},
        "error": ""
    }
    
    # Run the agent
    final_state = judging_agent.invoke(initial_state)
    
    # Return the assessment
    return final_state["assessment"]
