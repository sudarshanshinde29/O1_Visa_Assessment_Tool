# agents/child_agents/employment_agent.py
from typing import Dict, Any
from agents.child_agents.base_agent import create_child_agent_template

def create_employment_agent():
    """Create an agent specialized in assessing employment criterion"""
    
    system_prompt = """
    You are the Critical Employment Assessment Agent, specializing in evaluating evidence of employment in a critical or essential capacity for organizations with distinguished reputations.

    SPECIFIC EXPERTISE:
    You evaluate employment roles mentioned in CVs, determining if they were critical positions in distinguished organizations.

    EVALUATION CRITERIA:
    1. Evidence of employment in critical or essential capacities
    2. Organization's reputation and distinction in the field
    3. Whether the role was truly critical to the organization's mission
    4. Candidate's responsibilities, influence, and leadership
    5. Employment duration and progression

    EVIDENCE STRENGTH DEFINITIONS:
    - NONE: Only routine positions or employment at ordinary organizations
    - WEAK: Specialized roles but not clearly critical/essential
    - MODERATE: Critical roles at well-regarded organizations
    - STRONG: Clear evidence of critical roles at organizations with distinguished reputations
    """
    
    return create_child_agent_template("employment", system_prompt)

def evaluate_employment(resume_data: Dict[str, Any], criterion_mapping: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate the critical employment criterion for an O-1A visa application"""
    # Create the agent
    employment_agent = create_employment_agent()
    
    # Initialize the state
    initial_state = {
        "resume_data": resume_data,
        "criterion_mapping": criterion_mapping,
        "assessment": {},
        "error": ""
    }
    
    # Run the agent
    final_state = employment_agent.invoke(initial_state)
    
    # Return the assessment
    return final_state["assessment"]
