from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

class EducationItem(BaseModel):
    institution: str
    degree: str
    field: str
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    description: Optional[str] = None

class WorkExperienceItem(BaseModel):
    company: str
    title: str
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    description: Optional[str] = None
    achievements: List[str] = []

class PublicationItem(BaseModel):
    title: str
    venue: str
    date: Optional[str] = None
    authors: Optional[List[str]] = None
    description: Optional[str] = None
    citations: Optional[int] = None

class AwardItem(BaseModel):
    name: str
    issuer: str
    date: Optional[str] = None
    description: Optional[str] = None

class MembershipItem(BaseModel):
    organization: str
    role: Optional[str] = None
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    description: Optional[str] = None

class PressItem(BaseModel):
    publication: str
    title: str
    date: Optional[str] = None
    description: Optional[str] = None

class JudgingItem(BaseModel):
    role: str
    organization: str
    date: Optional[str] = None
    description: Optional[str] = None

class ContributionItem(BaseModel):
    title: str
    description: str
    impact: Optional[str] = None
    date: Optional[str] = None

class StructuredResume(BaseModel):
    personalInfo: Dict[str, Any] = Field(default_factory=dict)
    education: List[Dict[str, Any]] = Field(default_factory=list)
    workExperience: List[Dict[str, Any]] = Field(default_factory=list)
    publications: List[Dict[str, Any]] = Field(default_factory=list)
    awards: List[Dict[str, Any]] = Field(default_factory=list)
    memberships: List[Dict[str, Any]] = Field(default_factory=list)
    pressAndMedia: List[Dict[str, Any]] = Field(default_factory=list)
    judgingExperience: List[Dict[str, Any]] = Field(default_factory=list)
    contributions: List[Dict[str, Any]] = Field(default_factory=list)
    skills: List[str] = Field(default_factory=list)
    additionalInfo: Dict[str, Any] = Field(default_factory=dict)