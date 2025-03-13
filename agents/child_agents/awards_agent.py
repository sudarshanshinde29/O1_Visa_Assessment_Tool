# agents/child_agents/awards_agent.py
from typing import Dict, Any
from agents.child_agents.base_agent import create_child_agent_template

def create_awards_agent():
    """Create an agent specialized in assessing awards criterion"""
    
    system_prompt = """
    You are the Awards Assessment Agent, specializing in evaluating evidence of national or international recognition through prizes or awards for excellence.

    SPECIFIC EXPERTISE:
    You evaluate awards and prizes mentioned in CVs, determining their significance, prestige, and relevance to the candidate's field.

    EVALUATION CRITERIA:
    1. Receipt of nationally or internationally recognized prizes or awards
    2. Prestige and recognition level of each award
    3. Relevance to the candidate's field of endeavor
    4. Competitive nature and selectivity of the awards
    5. Whether awards were individual or team-based

    EVIDENCE STRENGTH DEFINITIONS:
    - NONE: No awards or only routine recognition
    - WEAK: Minor awards or recognition at local/regional level with limited selectivity
    - MODERATE: Recognized awards at national level OR competitive grants/fellowships
    - STRONG: Major internationally recognized awards OR national awards of exceptional prestige
    """
    
    return create_child_agent_template("awards", system_prompt)

def evaluate_awards(resume_data: Dict[str, Any], criterion_mapping: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate the awards criterion for an O-1A visa application"""
    # Create the agent
    awards_agent = create_awards_agent()
    
    # Initialize the state
    initial_state = {
        "resume_data": resume_data,
        "criterion_mapping": criterion_mapping,
        "assessment": {},
        "error": ""
    }
    
    # Run the agent
    final_state = awards_agent.invoke(initial_state)
    
    # Return the assessment
    return final_state["assessment"]
