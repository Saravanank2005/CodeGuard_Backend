# Pydantic models for request/response validation
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime

# User Models
class UserRegister(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6)
    full_name: str = Field(..., min_length=2)
    institution: Optional[str] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: str
    email: str
    full_name: str
    institution: Optional[str]
    created_at: datetime
    total_submissions: int = 0
    total_reports: int = 0

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse

# Submission Models
class SubmissionCreate(BaseModel):
    assignment_name: str
    files: List[str]  # File names
    
class SubmissionResponse(BaseModel):
    id: str
    user_id: str
    assignment_name: str
    files: List[dict]
    created_at: datetime
    status: str  # "pending", "processing", "completed", "failed"

# Report Models
class ReportResponse(BaseModel):
    id: str
    user_id: str
    submission_id: str
    assignment_name: str
    total_comparisons: int
    high_risk_pairs: List[dict]
    similarity_details: List[dict]
    created_at: datetime
    
# Statistics Models
class UserStatistics(BaseModel):
    total_submissions: int
    total_reports: int
    total_comparisons: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    recent_submissions: List[dict]
