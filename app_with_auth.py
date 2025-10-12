# FastAPI Backend with Authentication and MongoDB Integration
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from datetime import datetime, timedelta
from typing import List
import os
from bson import ObjectId

# Import our modules
from database import (
    connect_to_mongo, 
    close_mongo_connection,
    get_users_collection,
    get_submissions_collection,
    get_reports_collection,
    get_statistics_collection
)
from auth import (
    get_password_hash,
    verify_password,
    create_access_token,
    get_current_user
)
from models import (
    UserRegister,
    UserLogin,
    UserResponse,
    Token,
    SubmissionResponse,
    ReportResponse,
    UserStatistics
)

# Initialize FastAPI
app = FastAPI(title="CodeGuard API", version="2.0")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://*.vercel.app",
        "https://codeguard-frontend.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup and Shutdown Events
@app.on_event("startup")
async def startup_event():
    """Initialize database connection on startup"""
    await connect_to_mongo()
    
    # Create indexes for better performance
    users_collection = get_users_collection()
    if users_collection:
        await users_collection.create_index("email", unique=True)
        await users_collection.create_index("created_at")
    
    submissions_collection = get_submissions_collection()
    if submissions_collection:
        await submissions_collection.create_index("user_id")
        await submissions_collection.create_index("created_at")
    
    reports_collection = get_reports_collection()
    if reports_collection:
        await reports_collection.create_index("user_id")
        await reports_collection.create_index("submission_id")

@app.on_event("shutdown")
async def shutdown_event():
    """Close database connection on shutdown"""
    await close_mongo_connection()

# ============= AUTHENTICATION ROUTES =============

@app.post("/api/auth/register", response_model=Token, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserRegister):
    """Register a new user"""
    users_collection = get_users_collection()
    if users_collection is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    # Check if user already exists
    existing_user = await users_collection.find_one({"email": user_data.email})
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create new user
    hashed_password = get_password_hash(user_data.password)
    new_user = {
        "email": user_data.email,
        "password": hashed_password,
        "full_name": user_data.full_name,
        "institution": user_data.institution,
        "created_at": datetime.utcnow(),
        "total_submissions": 0,
        "total_reports": 0
    }
    
    result = await users_collection.insert_one(new_user)
    
    # Create access token
    access_token = create_access_token(data={"sub": user_data.email})
    
    # Prepare user response
    user_response = UserResponse(
        id=str(result.inserted_id),
        email=new_user["email"],
        full_name=new_user["full_name"],
        institution=new_user["institution"],
        created_at=new_user["created_at"],
        total_submissions=0,
        total_reports=0
    )
    
    return Token(access_token=access_token, user=user_response)

@app.post("/api/auth/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login user"""
    users_collection = get_users_collection()
    if users_collection is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    # Find user
    user = await users_collection.find_one({"email": form_data.username})
    if not user or not verify_password(form_data.password, user["password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token = create_access_token(data={"sub": user["email"]})
    
    # Prepare user response
    user_response = UserResponse(
        id=str(user["_id"]),
        email=user["email"],
        full_name=user["full_name"],
        institution=user.get("institution"),
        created_at=user["created_at"],
        total_submissions=user.get("total_submissions", 0),
        total_reports=user.get("total_reports", 0)
    )
    
    return Token(access_token=access_token, user=user_response)

@app.get("/api/auth/me", response_model=UserResponse)
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """Get current user information"""
    return UserResponse(
        id=current_user["_id"],
        email=current_user["email"],
        full_name=current_user["full_name"],
        institution=current_user.get("institution"),
        created_at=current_user["created_at"],
        total_submissions=current_user.get("total_submissions", 0),
        total_reports=current_user.get("total_reports", 0)
    )

# ============= SUBMISSION ROUTES (Protected) =============

@app.post("/api/submissions/upload")
async def upload_submission(
    assignment_name: str,
    files: List[UploadFile] = File(...),
    current_user: dict = Depends(get_current_user)
):
    """Upload files for plagiarism detection (requires authentication)"""
    submissions_collection = get_submissions_collection()
    if submissions_collection is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    # Save files and create submission record
    uploaded_files = []
    for file in files:
        file_content = await file.read()
        file_data = {
            "filename": file.filename,
            "content": file_content.decode('utf-8'),
            "size": len(file_content),
            "uploaded_at": datetime.utcnow()
        }
        uploaded_files.append(file_data)
    
    # Create submission
    submission = {
        "user_id": current_user["_id"],
        "user_email": current_user["email"],
        "assignment_name": assignment_name,
        "files": uploaded_files,
        "status": "completed",
        "created_at": datetime.utcnow()
    }
    
    result = await submissions_collection.insert_one(submission)
    
    # Update user submission count
    users_collection = get_users_collection()
    await users_collection.update_one(
        {"_id": ObjectId(current_user["_id"])},
        {"$inc": {"total_submissions": 1}}
    )
    
    return {
        "message": "Files uploaded successfully",
        "submission_id": str(result.inserted_id),
        "files_count": len(uploaded_files)
    }

@app.get("/api/submissions")
async def get_user_submissions(current_user: dict = Depends(get_current_user)):
    """Get all submissions for current user"""
    submissions_collection = get_submissions_collection()
    if submissions_collection is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    cursor = submissions_collection.find(
        {"user_id": current_user["_id"]}
    ).sort("created_at", -1)
    
    submissions = []
    async for submission in cursor:
        submissions.append({
            "id": str(submission["_id"]),
            "assignment_name": submission["assignment_name"],
            "files_count": len(submission["files"]),
            "status": submission["status"],
            "created_at": submission["created_at"].isoformat()
        })
    
    return {"submissions": submissions}

@app.delete("/api/submissions/{submission_id}")
async def delete_submission(
    submission_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete a submission"""
    submissions_collection = get_submissions_collection()
    if submissions_collection is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    result = await submissions_collection.delete_one({
        "_id": ObjectId(submission_id),
        "user_id": current_user["_id"]
    })
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Submission not found")
    
    return {"message": "Submission deleted successfully"}

# ============= STATISTICS ROUTES (Protected) =============

@app.get("/api/statistics")
async def get_user_statistics(current_user: dict = Depends(get_current_user)):
    """Get statistics for current user"""
    submissions_collection = get_submissions_collection()
    reports_collection = get_reports_collection()
    
    if not submissions_collection or not reports_collection:
        raise HTTPException(status_code=503, detail="Database not available")
    
    # Get submission count
    total_submissions = await submissions_collection.count_documents(
        {"user_id": current_user["_id"]}
    )
    
    # Get report count
    total_reports = await reports_collection.count_documents(
        {"user_id": current_user["_id"]}
    )
    
    # Get recent submissions
    recent_cursor = submissions_collection.find(
        {"user_id": current_user["_id"]}
    ).sort("created_at", -1).limit(5)
    
    recent_submissions = []
    async for sub in recent_cursor:
        recent_submissions.append({
            "assignment_name": sub["assignment_name"],
            "files_count": len(sub["files"]),
            "created_at": sub["created_at"].isoformat()
        })
    
    return {
        "total_submissions": total_submissions,
        "total_reports": total_reports,
        "total_comparisons": 0,  # Calculate from reports
        "high_risk_count": 0,
        "medium_risk_count": 0,
        "low_risk_count": 0,
        "recent_submissions": recent_submissions
    }

# ============= HEALTH CHECK =============

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    users_collection = get_users_collection()
    db_status = users_collection is not None
    
    return {
        "ok": True,
        "database_connected": db_status,
        "version": "2.0"
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "CodeGuard API v2.0",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
