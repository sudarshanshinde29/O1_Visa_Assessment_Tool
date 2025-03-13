# agents/parent_agent.py
from typing import Dict, Any, List, TypedDict, Optional, Tuple
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import langgraph as lg
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
import os
import json

from sentence_transformers import SentenceTransformer


# Define state for the parent agent
class ParentAgentState(TypedDict):
    structured_resume: Dict[str, Any]
    criteria_mapping: Dict[str, Any]
    child_assessments: Dict[str, Dict[str, Any]]
    rag_context: List[str]
    interim_analyses: Dict[str, Any]
    final_assessment: Dict[str, Any]
    error: str
    stage: str

class ParentAgent:
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
        self.system_prompt = self._get_system_prompt()
        self.vectorstore = self._setup_knowledge_base()
        self.workflow = self._create_workflow()
    
    def _get_system_prompt(self) -> str:
        """Define the system prompt for the parent agent."""
        return """
        You are the Critical Assessment Agent, an expert legal analyst specializing in O-1A visa qualification assessment. Your role is to coordinate a team of 8 specialized agents to evaluate a candidate's CV against O-1A visa criteria and determine their qualification likelihood.
        
        PRIMARY RESPONSIBILITIES:
        1. Review the structured resume and criteria mapping from specialized agents
        2. Validate and critically assess the findings of each child agent
        3. Apply USCIS standards and policies to the evidence
        4. Synthesize all findings to determine the final rating
        5. Provide a comprehensive explanation for the rating with specific references to USCIS guidelines
        
        RATING DETERMINATION RULES:
        To qualify for an O-1A visa, a candidate must satisfy at least 3 of the 8 criteria:
        - HIGH: Meets 5+ criteria with at least 3 having strong evidence, OR meets 3-4 criteria with exceptional evidence
        - MEDIUM: Meets 3-4 criteria with moderate to strong evidence
        - LOW: Meets fewer than 3 criteria OR meets exactly 3 with mostly weak evidence
        
        Your analysis must be grounded in both the evidence from the resume and specific USCIS standards for O-1A visas. When analyzing evidence, refer to official USCIS policy guidance and precedent decisions.
        """
    
    def _setup_knowledge_base(self):
        """Set up the RAG knowledge base for O-1A requirements."""
        # Check if we have a persisted vector store
        persist_directory = "./knowledge_base/chroma_db"
        
        try:
            if os.path.exists(persist_directory):
                model = SentenceTransformer('all-MiniLM-L6-v2')
                embeddings = model.encode(["Hello world!", "O-1A visa requirements"], show_progress_bar=True)

                # Load existing vector store
                # embeddings = OpenAIEmbeddings()
                return Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        except:
            pass
            
        # Create new vector store if it doesn't exist
        try:
            # Load the knowledge base document
            knowledge_base_path = "./knowledge_base/o1a_requirements.md"
            if not os.path.exists(knowledge_base_path):
                # Create a minimal knowledge base document if it doesn't exist
                self._create_minimal_knowledge_base(knowledge_base_path)
                
            with open(knowledge_base_path, "r") as f:
                knowledge_base_text = f.read()
            
            # Split the text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n# ", "\n## ", "\n### ", "\n#### ", "\n", " ", ""]
            )
            docs = text_splitter.create_documents([knowledge_base_text])
            
            # Create embeddings and store in vector database
            embeddings = OpenAIEmbeddings()
            os.makedirs(os.path.dirname(persist_directory), exist_ok=True)
            vectorstore = Chroma.from_documents(
                documents=docs,
                embedding=embeddings,
                persist_directory=persist_directory
            )
            
            return vectorstore
        except Exception as e:
            print(f"Error setting up knowledge base: {str(e)}")
            # Return None if setup fails, agent will work without RAG
            return None
    
    def _create_minimal_knowledge_base(self, path: str):
        """Create a minimal knowledge base file if none exists."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        minimal_content = """
        # O-1A Visa Requirements
        
        ## Core O-1A Visa Requirements
        The O-1A nonimmigrant visa is for individuals with extraordinary ability in sciences, education, business, or athletics. To qualify, applicants must demonstrate sustained national or international acclaim by meeting at least 3 of the 8 criteria.
        
        ## The Eight O-1A Criteria
        1. Receipt of nationally or internationally recognized prizes or awards for excellence
        2. Membership in associations requiring outstanding achievements as judged by recognized experts
        3. Published material about the beneficiary in professional or major media
        4. Participation as a judge of the work of others in the same or allied field
        5. Original scientific, scholarly, or business-related contributions of major significance
        6. Authorship of scholarly articles in professional publications or major media
        7. Employment in a critical or essential capacity for organizations with distinguished reputation
        8. High salary or remuneration in relation to others in the field
        
        ## Evidence Evaluation
        USCIS evaluates evidence based on quality and quantity. Strong evidence across multiple criteria increases chances of approval.
        """
        with open(path, "w") as f:
            f.write(minimal_content)
    
    def query_knowledge_base(self, query: str, k: int = 3) -> List[str]:
        """Query the RAG knowledge base for relevant information."""
        if not self.vectorstore:
            return ["Knowledge base unavailable"]
        
        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            return [doc.page_content for doc in docs]
        except Exception as e:
            print(f"Error querying knowledge base: {str(e)}")
            return ["Error retrieving information from knowledge base"]
    
    def _create_workflow(self):
        """Create the workflow for the parent agent."""
        workflow = StateGraph(ParentAgentState)
        
        # Add nodes
        workflow.add_node("initial_analysis", self.initial_analysis)
        workflow.add_node("retrieve_context", self.retrieve_context)
        workflow.add_node("analyze_child_assessments", self.analyze_child_assessments)
        workflow.add_node("cross_reference_criteria", self.cross_reference_criteria)
        workflow.add_node("final_determination", self.final_determination)
        workflow.add_node("generate_recommendations", self.generate_recommendations)
        workflow.add_node("handle_error", self.handle_error)
        
        # Add edges
        workflow.add_edge("initial_analysis", "retrieve_context")
        workflow.add_edge("retrieve_context", "analyze_child_assessments")
        workflow.add_edge("analyze_child_assessments", "cross_reference_criteria")
        workflow.add_edge("cross_reference_criteria", "final_determination")
        workflow.add_edge("final_determination", "generate_recommendations")
        workflow.add_edge("generate_recommendations", END)
        
        # Error handling edges
        workflow.add_edge("handle_error", "final_determination")
        
        # Conditional edges for error handling
        def should_handle_error(state: ParentAgentState) -> bool:
            return bool(state.get("error", ""))
        
        workflow.add_conditional_edges(
            "initial_analysis",
            lambda state: "handle_error" if should_handle_error(state) else "retrieve_context",
            {
                "handle_error": "handle_error",
                "retrieve_context": "retrieve_context"
            }
        )

        
        workflow.add_conditional_edges(
            "retrieve_context",
            lambda state: "handle_error" if should_handle_error(state) else "analyze_child_assessments",
            {
                "handle_error": "handle_error",
                "analyze_child_assessments": "analyze_child_assessments"
            }
        )

        workflow.add_conditional_edges(
            "analyze_child_assessments",
            lambda state: "handle_error" if should_handle_error(state) else "cross_reference_criteria",
            {
                "handle_error": "handle_error",
                "cross_reference_criteria": "cross_reference_criteria"
            }
        )

        
        # Set entry point
        workflow.set_entry_point("initial_analysis")
        
        # Compile the graph
        return workflow.compile()
    
    def initial_analysis(self, state: ParentAgentState) -> ParentAgentState:
        """Initial analysis of the structured resume and criteria mapping."""
        try:
            structured_resume = state.get("structured_resume", {})
            criteria_mapping = state.get("criteria_mapping", {})
            
            # Validate input
            if not structured_resume:
                return {**state, "error": "Structured resume is missing or empty"}
            
            if not criteria_mapping:
                return {**state, "error": "Criteria mapping is missing or empty"}
            
            # Create prompt for the LLM
            user_prompt = f"""
            Please perform an initial analysis of this structured resume for O-1A visa assessment:
            
            ```
            {json.dumps(structured_resume, indent=2)}
            ```
            
            Focus on:
            1. Identifying the applicant's primary field of expertise
            2. Determining the most promising criteria based on the resume
            3. Noting any potential challenges or weaknesses in the application
            
            Provide a concise analysis that will guide our detailed criteria assessment.
            """
            
            # Get response from LLM
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=user_prompt)
            ]
            response = self.llm.invoke(messages)
            
            return {
                **state,
                "interim_analyses": {"initial_analysis": response.content},
                "stage": "initial_analysis",
                "error": ""
            }
            
        except Exception as e:
            return {**state, "error": f"Error in initial analysis: {str(e)}"}
    
    def retrieve_context(self, state: ParentAgentState) -> ParentAgentState:
        """Retrieve relevant context from the knowledge base."""
        try:
            # Extract the field of expertise from the initial analysis
            initial_analysis = state.get("interim_analyses", {}).get("initial_analysis", "")
            
            # Generate queries based on the initial analysis
            queries = [
                "O-1A visa requirements and standards",
                "Evidence evaluation for O-1A visa applications",
                "USCIS policy on extraordinary ability"
            ]
            
            # Add more specific queries based on the initial analysis
            if "science" in initial_analysis.lower() or "research" in initial_analysis.lower():
                queries.append("O-1A requirements for scientists and researchers")
            
            if "business" in initial_analysis.lower() or "entrepreneur" in initial_analysis.lower():
                queries.append("O-1A requirements for business professionals and entrepreneurs")
            
            if "tech" in initial_analysis.lower() or "software" in initial_analysis.lower():
                queries.append("O-1A requirements for technology professionals")
            
            # Retrieve context for each query
            rag_context = []
            for query in queries:
                contexts = self.query_knowledge_base(query)
                rag_context.extend(contexts)
            
            # Deduplicate context
            rag_context = list(set(rag_context))
            
            return {
                **state,
                "rag_context": rag_context,
                "stage": "retrieve_context",
                "error": ""
            }
            
        except Exception as e:
            return {**state, "error": f"Error retrieving context: {str(e)}"}
    
    def analyze_child_assessments(self, state: ParentAgentState) -> ParentAgentState:
        """Analyze the assessments from child agents."""
        try:
            child_assessments = state.get("child_assessments", {})
            rag_context = state.get("rag_context", [])
            
            # Combine relevant RAG context
            combined_context = "\n\n".join(rag_context[:3])  # Limit to most relevant chunks
            
            # Create prompt for the LLM
            user_prompt = f"""
            Analyze the following assessments from specialized child agents for each O-1A criterion:
            
            ```
            {json.dumps(child_assessments, indent=2)}
            ```
            
            Based on the following O-1A visa requirements:
            
            {combined_context}
            
            For each criterion, provide:
            1. A critical evaluation of the child agent's assessment
            2. An assessment of whether the evidence meets USCIS standards
            3. Any concerns about the quality or sufficiency of evidence
            
            Provide a detailed analysis for each criterion.
            """
            
            # Get response from LLM
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=user_prompt)
            ]
            response = self.llm.invoke(messages)
            
            return {
                **state,
                "interim_analyses": {
                    **(state.get("interim_analyses", {})),
                    "child_assessment_analysis": response.content
                },
                "stage": "analyze_child_assessments",
                "error": ""
            }
            
        except Exception as e:
            return {**state, "error": f"Error analyzing child assessments: {str(e)}"}
    
    def cross_reference_criteria(self, state: ParentAgentState) -> ParentAgentState:
        """Cross-reference evidence across different criteria."""
        try:
            child_assessments = state.get("child_assessments", {})
            interim_analyses = state.get("interim_analyses", {})
            structured_resume = state.get("structured_resume", {})
            
            # Create prompt for the LLM
            user_prompt = f"""
            Perform a cross-referencing analysis across the 8 O-1A criteria to identify:
            
            1. Evidence that supports multiple criteria
            2. Internal consistency of evidence across criteria
            3. Potentially overlooked evidence from the resume
            
            Resume:
            ```
            {json.dumps(structured_resume, indent=2)}
            ```
            
            Child Assessments:
            ```
            {json.dumps(child_assessments, indent=2)}
            ```
            
            Provide a comprehensive cross-reference analysis focusing on strengthening the O-1A case.
            """
            
            # Get response from LLM
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=user_prompt)
            ]
            response = self.llm.invoke(messages)
            
            return {
                **state,
                "interim_analyses": {
                    **interim_analyses,
                    "cross_reference_analysis": response.content
                },
                "stage": "cross_reference_criteria",
                "error": ""
            }
            
        except Exception as e:
            return {**state, "error": f"Error cross-referencing criteria: {str(e)}"}
    
    def final_determination(self, state: ParentAgentState) -> ParentAgentState:
        """Make the final determination about O-1A qualification."""
        try:
            child_assessments = state.get("child_assessments", {})
            interim_analyses = state.get("interim_analyses", {})
            rag_context = state.get("rag_context", [])
            
            # Extract strengths from child assessments
            criteria_strengths = {}
            for criterion, assessment in child_assessments.items():
                if isinstance(assessment, dict) and "assessment" in assessment:
                    strength = assessment["assessment"].get("strength", "None")
                    criteria_strengths[criterion] = strength
                else:
                    criteria_strengths[criterion] = "None"
            
            # Combine relevant RAG context for final determination
            combined_context = "\n\n".join(rag_context[:5])  # Include more context for final determination
            
            # Create prompt for the LLM
            user_prompt = f"""
            Make a final determination about this applicant's qualification for an O-1A visa.
            
            Criteria Strengths:
            ```
            {json.dumps(criteria_strengths, indent=2)}
            ```
            
            Previous Analyses:
            
            Initial Analysis:
            {interim_analyses.get("initial_analysis", "Not available")}
            
            Child Assessment Analysis:
            {interim_analyses.get("child_assessment_analysis", "Not available")}
            
            Cross-Reference Analysis:
            {interim_analyses.get("cross_reference_analysis", "Not available")}
            
            O-1A Requirements and Standards:
            {combined_context}
            
            Based on all available information, determine:
            1. Overall Rating (HIGH, MEDIUM, or LOW)
            2. Detailed justification with specific references to USCIS standards
            3. Summary of evidence for each criterion
            4. Overall strength of the application
            
            Your determination must be well-reasoned and supported by specific evidence and USCIS standards.
            """
            
            # Get response from LLM
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=user_prompt)
            ]
            response = self.llm.invoke(messages)
            
            # Extract rating from response
            rating = self._extract_rating(response.content)
            
            # Prepare final assessment
            final_assessment = {
                "rating": rating,
                "justification": response.content,
                "criteria_summary": criteria_strengths
            }
            
            return {
                **state,
                "final_assessment": final_assessment,
                "stage": "final_determination",
                "error": ""
            }
            
        except Exception as e:
            # Create a fallback rating based on child assessments if final determination fails
            try:
                child_assessments = state.get("child_assessments", {})
                criteria_strengths = {}
                
                for criterion, assessment in child_assessments.items():
                    if isinstance(assessment, dict) and "assessment" in assessment:
                        strength = assessment["assessment"].get("strength", "None")
                        criteria_strengths[criterion] = strength
                    else:
                        criteria_strengths[criterion] = "None"
                
                # Simple calculation
                strong_count = sum(1 for s in criteria_strengths.values() if s == "Strong")
                moderate_count = sum(1 for s in criteria_strengths.values() if s == "Moderate")
                
                if strong_count >= 3 or (strong_count + moderate_count) >= 5:
                    rating = "HIGH"
                elif (strong_count + moderate_count) >= 3:
                    rating = "MEDIUM"
                else:
                    rating = "LOW"
                
                return {
                    **state,
                    "final_assessment": {
                        "rating": rating,
                        "justification": f"Rating based on {strong_count} strong and {moderate_count} moderate criteria. Error occurred: {str(e)}",
                        "criteria_summary": criteria_strengths
                    },
                    "stage": "final_determination",
                    "error": f"Error in final determination: {str(e)}"
                }
            except:
                return {**state, "error": f"Error in final determination: {str(e)}"}
    
    def generate_recommendations(self, state: ParentAgentState) -> ParentAgentState:
        """Generate recommendations for improving the application."""
        try:
            final_assessment = state.get("final_assessment", {})
            child_assessments = state.get("child_assessments", {})
            criteria_strengths = final_assessment.get("criteria_summary", {})
            rating = final_assessment.get("rating", "LOW")
            
            # Create prompt for the LLM
            user_prompt = f"""
            Based on the final assessment (Rating: {rating}), generate specific recommendations for strengthening this O-1A visa application.
            
            Criteria Strengths:
            ```
            {json.dumps(criteria_strengths, indent=2)}
            ```
            
            Focus on:
            1. Specific improvements for weak criteria
            2. Additional evidence needed for borderline criteria
            3. Strategic advice for presenting the strongest case
            4. Alternative visa categories if O-1A is not recommended
            
            Provide actionable, specific recommendations.
            """
            
            # Get response from LLM
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=user_prompt)
            ]
            response = self.llm.invoke(messages)
            
            # Update final assessment with recommendations
            final_assessment["recommendations"] = response.content
            
            return {
                **state,
                "final_assessment": final_assessment,
                "stage": "generate_recommendations",
                "error": ""
            }
            
        except Exception as e:
            # Add basic recommendations if generation fails
            final_assessment = state.get("final_assessment", {})
            criteria_strengths = final_assessment.get("criteria_summary", {})
            
            weak_criteria = [c for c, s in criteria_strengths.items() if s in ["Weak", "None"]]
            basic_recommendations = f"Consider strengthening evidence for: {', '.join(weak_criteria)}"
            
            final_assessment["recommendations"] = basic_recommendations
            
            return {
                **state,
                "final_assessment": final_assessment,
                "stage": "generate_recommendations",
                "error": f"Error generating recommendations: {str(e)}"
            }
    
    def handle_error(self, state: ParentAgentState) -> ParentAgentState:
        """Handle errors in the parent agent process."""
        error = state.get("error", "Unknown error")
        stage = state.get("stage", "unknown")
        
        print(f"Error in parent agent at stage {stage}: {error}")
        
        # Create a fallback assessment
        child_assessments = state.get("child_assessments", {})
        criteria_strengths = {}
        
        try:
            for criterion, assessment in child_assessments.items():
                if isinstance(assessment, dict) and "assessment" in assessment:
                    strength = assessment["assessment"].get("strength", "None")
                    criteria_strengths[criterion] = strength
                else:
                    criteria_strengths[criterion] = "None"
            
            # Simple calculation
            strong_count = sum(1 for s in criteria_strengths.values() if s == "Strong")
            moderate_count = sum(1 for s in criteria_strengths.values() if s == "Moderate")
            
            if strong_count >= 3 or (strong_count + moderate_count) >= 5:
                rating = "HIGH"
            elif (strong_count + moderate_count) >= 3:
                rating = "MEDIUM"
            else:
                rating = "LOW"
            
            return {
                **state,
                "final_assessment": {
                    "rating": rating,
                    "justification": f"Rating based on {strong_count} strong and {moderate_count} moderate criteria. Error occurred during {stage}: {error}",
                    "criteria_summary": criteria_strengths,
                    "error_occurred": True,
                    "recommendations": "Unable to generate recommendations due to error."
                },
                "error": error  # Preserve the error for debugging
            }
        except Exception as e:
            # If everything fails, return a minimal assessment
            return {
                **state,
                "final_assessment": {
                    "rating": "LOW",
                    "justification": f"Unable to properly assess due to errors: {error}, {str(e)}",
                    "criteria_summary": {},
                    "error_occurred": True,
                    "recommendations": "Unable to evaluate application due to system errors."
                },
                "error": f"{error}; Additional error in error handling: {str(e)}"
            }
    
    import re

    def _extract_rating(self, text: str) -> str:
        # Check for lines like "Overall Rating: HIGH" using regex
        match = re.search(r"OVERALL RATING:\s*(HIGH|MEDIUM|LOW)", text.upper())
        if match:
            return match.group(1)
        return "LOW"  # fallback if nothing matches

    
    def invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke the parent agent workflow."""
        # Validate input
        if "structured_resume" not in input_data:
            return {"error": "Structured resume is required"}
        
        if "criteria_mapping" not in input_data:
            return {"error": "Criteria mapping is required"}
        
        if "child_assessments" not in input_data:
            return {"error": "Child assessments are required"}
        
        # Initialize state
        state = {
            "structured_resume": input_data["structured_resume"],
            "criteria_mapping": input_data["criteria_mapping"],
            "child_assessments": input_data["child_assessments"],
            "rag_context": [],
            "interim_analyses": {},
            "final_assessment": {},
            "error": "",
            "stage": ""
        }
        
        # Run the workflow
        final_state = self.workflow.invoke(state)
        
        # Return the final assessment
        return {
            "final_assessment": final_state["final_assessment"],
            "error": final_state.get("error", "")
        }

# Function to assess O-1A qualification
def assess_o1a_qualification(structured_resume: Dict[str, Any], criteria_mapping: Dict[str, Any], child_assessments: Dict[str, Any]) -> Dict[str, Any]:
    """Assess a candidate's qualification for an O-1A visa."""
    # Create the parent agent
    parent_agent = ParentAgent()
    
    # Run the parent agent
    return parent_agent.invoke({
        "structured_resume": structured_resume,
        "criteria_mapping": criteria_mapping,
        "child_assessments": child_assessments
    })
