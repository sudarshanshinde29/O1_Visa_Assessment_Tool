# agents/child_agents/membership_agent.py
from typing import Dict, Any
from agents.child_agents.base_agent import create_child_agent_template

def create_membership_agent():
    """Create an agent specialized in assessing membership criterion"""
    
    system_prompt = """
    You are the Membership Assessment Agent, specializing in evaluating evidence of membership in associations requiring outstanding achievements.

    SPECIFIC EXPERTISE:
    You evaluate professional memberships mentioned in CVs, determining if they require outstanding achievements as judged by recognized experts.

    EVALUATION CRITERIA:
    1. Membership in associations in the relevant field
    2. Association's membership requirements and selectivity
    3. Whether membership is based on outstanding achievements
    4. Whether achievements are judged by recognized experts
    5. Prestige and recognition of the associations

    EVIDENCE STRENGTH DEFINITIONS:
    - NONE: Only standard professional associations with no selective requirements
    - WEAK: Membership in selective organizations but not requiring outstanding achievements
    - MODERATE: Membership in selective associations requiring peer recognition
    - STRONG: Membership in highly selective, prestigious associations requiring outstanding achievements
    """
    
    return create_child_agent_template("membership", system_prompt)

def evaluate_membership(resume_data: Dict[str, Any], criterion_mapping: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate the membership criterion for an O-1A visa application"""
    # Create the agent
    membership_agent = create_membership_agent()
    
    # Initialize the state
    initial_state = {
        "resume_data": resume_data,
        "criterion_mapping": criterion_mapping,
        "assessment": {},
        "error": ""
    }
    
    # Run the agent
    final_state = membership_agent.invoke(initial_state)
    
    # Return the assessment
    return final_state["assessment"]
