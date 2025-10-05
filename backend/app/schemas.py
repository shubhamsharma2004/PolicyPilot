from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class AskRequest(BaseModel):
    question: str
    filters: Optional[Dict[str, Any]] = None
    top_k: int = 5
    follow_up_context: Optional[str] = None

class Citation(BaseModel):
    doc_id: str
    section: Optional[str] = None
    page: Optional[int] = None
    snippet: Optional[str] = None

class AskResponse(BaseModel):
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    policy_matches: List[str] = Field(default_factory=list)
    confidence: str = "medium"
    disclaimer: Optional[str] = "If your contract specifies otherwise, your contract prevails."
    follow_up_suggestions: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class FeedbackIn(BaseModel):
    answer_id: str
    rating: int = 5
    comment: Optional[str] = None
