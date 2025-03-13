# agents/agent_manager.py
from typing import Dict, Any
from agents.child_agents.awards_agent import create_awards_agent
from agents.child_agents.membership_agent import create_membership_agent
from agents.child_agents.press_agent import create_press_agent
from agents.child_agents.judging_agent import create_judging_agent
from agents.child_agents.contributions_agent import create_contributions_agent
from agents.child_agents.articles_agent import create_articles_agent
from agents.child_agents.employment_agent import create_employment_agent
from agents.child_agents.remuneration_agent import create_remuneration_agent
from agents.parent_agent import ParentAgent
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentManager:
    def __init__(self):
        self.agents = {}
        self.parent_agent = ParentAgent()
        self._load_agents()
        
    def _load_agents(self):
        """Load all child agents."""
        self.agents = {
            "awards": create_awards_agent(),
            "membership": create_membership_agent(),
            "press": create_press_agent(),
            "judging": create_judging_agent(),
            "contributions": create_contributions_agent(),
            "articles": create_articles_agent(),
            "employment": create_employment_agent(),
            "remuneration": create_remuneration_agent()
        }

    def process_criterion(self, criterion: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a specific criterion using the appropriate child agent."""
        agent = self.agents.get(criterion)
        if not agent:
            return {"error": f"No agent found for criterion: {criterion}"}
        
        try:
            logger.info(f"Invoking {criterion} agent with input data: {input_data}")
            result = agent.invoke(input_data)
            logger.info(f"Result from {criterion} agent: {result}")
            return result
        except Exception as e:
            logger.error(f"{criterion} agent failed: {str(e)}")
            return {"error": f"{criterion} agent failed: {str(e)}"}

    def coordinate_assessment(self, structured_resume: Dict[str, Any], criteria_mapping: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate the full assessment process."""
        try:
            logger.info("Starting child agent assessments...")
            # Process each criterion with its dedicated agent
            child_assessments = {}
            for criterion in self.agents.keys():
                logger.info(f"Processing {criterion} criterion...")
                input_data = {
                    "resume_data": structured_resume,
                    "criterion_mapping": criteria_mapping.get(criterion, {})
                }
                child_assessments[criterion] = self.process_criterion(criterion, input_data)
            
            logger.info("Child agent assessments complete. Starting parent agent...")
            # Now invoke the parent agent with all child assessments
            parent_input = {
                "structured_resume": structured_resume,
                "criteria_mapping": criteria_mapping,
                "child_assessments": child_assessments
            }
            logger.info(f"Invoking parent agent with input data: {parent_input}")
            parent_result = self.parent_agent.invoke(parent_input)
            logger.info(f"Result from parent agent: {parent_result}")
            
            logger.info("Parent agent assessment complete.")
            # Return the combined results
            return {
                "child_assessments": child_assessments,
                "final_assessment": parent_result.get("final_assessment", {}),
                "error": parent_result.get("error", "")
            }
        except Exception as e:
            logger.error(f"Error in coordination: {str(e)}")
            return {
                "error": f"Error coordinating assessment: {str(e)}",
                "child_assessments": {},
                "final_assessment": {
                    "rating": "LOW",
                    "justification": f"System error occurred: {str(e)}",
                    "criteria_summary": {}
                }
            }

    def get_all_agents_status(self) -> Dict[str, str]:
        """Get the status of all agents."""
        status = {}
        
        # Check child agents
        for criterion, agent in self.agents.items():
            status[criterion] = "loaded" if agent else "not loaded"
        
        # Check parent agent
        status["parent"] = "loaded" if self.parent_agent else "not loaded"
        
        return status