# models/assessment.py
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

class CriterionEvidence(BaseModel):
    criterion: str
    relevantItems: List[Dict[str, Any]] = Field(default_factory=list)
    context: str
    potentialStrength: str

class CriteriaMapping(BaseModel):
    awards: CriterionEvidence
    membership: CriterionEvidence
    press: CriterionEvidence
    judging: CriterionEvidence
    contributions: CriterionEvidence
    articles: CriterionEvidence
    employment: CriterionEvidence
    remuneration: CriterionEvidence