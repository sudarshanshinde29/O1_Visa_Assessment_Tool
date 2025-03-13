# agents/child_agents/articles_agent.py
from typing import Dict, Any
from agents.child_agents.base_agent import create_child_agent_template

def create_articles_agent():
    """Create an agent specialized in assessing scholarly articles criterion"""
    
    system_prompt = """
    You are the Scholarly Articles Assessment Agent, specializing in evaluating evidence of authorship of scholarly articles in professional journals or other major media.

    SPECIFIC EXPERTISE:
    You evaluate scholarly publications mentioned in CVs, determining their significance, impact, and recognition within the candidate's field.

    EVALUATION CRITERIA:
    1. Authorship of scholarly articles, publications, or equivalent media
    2. Prestige and impact factor of each publication venue
    3. Candidate's authorship role (first, corresponding, etc.)
    4. Citation metrics or other impact indicators when available
    5. Publication quantity, quality, and consistency

    EVIDENCE STRENGTH DEFINITIONS:
    - NONE: No scholarly publications or only non-peer-reviewed publications
    - WEAK: Few publications in minor journals or limited citations
    - MODERATE: Regular publications in respected journals with normal citation patterns
    - STRONG: Extensive publication record in prestigious journals with significant citations
    """
    
    return create_child_agent_template("articles", system_prompt)

def evaluate_articles(resume_data: Dict[str, Any], criterion_mapping: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate the scholarly articles criterion for an O-1A visa application"""
    # Create the agent
    articles_agent = create_articles_agent()
    
    # Initialize the state
    initial_state = {
        "resume_data": resume_data,
        "criterion_mapping": criterion_mapping,
        "assessment": {},
        "error": ""
    }
    
    # Run the agent
    final_state = articles_agent.invoke(initial_state)
    
    # Return the assessment
    return final_state["assessment"]
