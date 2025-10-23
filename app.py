# FastAPI Backend with Authentication and MongoDB Integration
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from datetime import datetime, timedelta
from typing import List
import os
from bson import ObjectId

# ML/Similarity imports
import Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib

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

# ============= SIMILARITY CALCULATION FUNCTIONS =============

def calculate_similarity(code1: str, code2: str) -> dict:
    """
    Calculate similarity between two code snippets using multiple methods.
    Returns a dict with various similarity metrics.
    """
    try:
        # Remove extra whitespace for better comparison
        code1_clean = ' '.join(code1.split())
        code2_clean = ' '.join(code2.split())
        
        # Method 1: Levenshtein Distance (character-level similarity)
        lev_distance = Levenshtein.distance(code1_clean, code2_clean)
        max_len = max(len(code1_clean), len(code2_clean))
        if max_len > 0:
            sim_lex = 1 - (lev_distance / max_len)
        else:
            sim_lex = 1.0
        
        # Method 2: Sequence Matcher (ratio of matching subsequences)
        seqmatch = difflib.SequenceMatcher(None, code1_clean, code2_clean).ratio()
        
        # Method 3: TF-IDF + Cosine Similarity (semantic similarity)
        try:
            vectorizer = TfidfVectorizer(
                token_pattern=r'\b\w+\b',
                ngram_range=(1, 2),
                min_df=1
            )
            tfidf_matrix = vectorizer.fit_transform([code1_clean, code2_clean])
            cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except:
            cosine_sim = 0.0
        
        # Method 4: Jaccard Similarity (set-based comparison)
        words1 = set(code1_clean.split())
        words2 = set(code2_clean.split())
        if len(words1.union(words2)) > 0:
            jaccard = len(words1.intersection(words2)) / len(words1.union(words2))
        else:
            jaccard = 0.0
        
        # Calculate overall similarity (weighted average)
        # Give more weight to sequence matching and TF-IDF
        overall_similarity = (
            sim_lex * 0.2 +      # Lexical: 20%
            seqmatch * 0.3 +     # Sequence: 30%
            cosine_sim * 0.3 +   # TF-IDF: 30%
            jaccard * 0.2        # Jaccard: 20%
        )
        
        return {
            "similarity": overall_similarity,
            "sim_lex": sim_lex,
            "sim_ast": seqmatch,  # Using sequence match as AST proxy
            "jaccard": jaccard,
            "seqmatch": seqmatch,
            "cosine": cosine_sim
        }
    except Exception as e:
        print(f"Error calculating similarity: {e}")
        # Return low similarity on error
        return {
            "similarity": 0.0,
            "sim_lex": 0.0,
            "sim_ast": 0.0,
            "jaccard": 0.0,
            "seqmatch": 0.0,
            "cosine": 0.0
        }

def get_risk_level(similarity: float) -> str:
    """Determine risk level based on similarity score"""
    if similarity >= 0.8:
        return "high"
    elif similarity >= 0.5:
        return "medium"
    else:
        return "low"

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "https://code-guard-frontend-nu.vercel.app"
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
    if users_collection is not None:
        await users_collection.create_index("email", unique=True)
        await users_collection.create_index("created_at")
    
    submissions_collection = get_submissions_collection()
    if submissions_collection is not None:
        await submissions_collection.create_index("user_id")
        await submissions_collection.create_index("created_at")
    
    reports_collection = get_reports_collection()
    if reports_collection is not None:
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
    assignment_name: str = Form(...),
    student_ids: str = Form(...),  # Comma-separated student IDs
    files: List[UploadFile] = File(...),
    current_user: dict = Depends(get_current_user)
):
    """Upload student files for plagiarism detection (requires authentication)"""
    submissions_collection = get_submissions_collection()
    reports_collection = get_reports_collection()
    
    if submissions_collection is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    # Parse student IDs
    student_id_list = [sid.strip() for sid in student_ids.split(',')]
    
    # Validate: number of files should match number of student IDs
    if len(files) != len(student_id_list):
        raise HTTPException(
            status_code=400, 
            detail=f"Number of files ({len(files)}) must match number of student IDs ({len(student_id_list)})"
        )
    
    # Save files with student IDs
    uploaded_files = []
    for idx, file in enumerate(files):
        file_content = await file.read()
        student_id = student_id_list[idx]
        
        file_data = {
            "student_id": student_id,
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
        "total_students": len(uploaded_files),
        "status": "pending_analysis",
        "created_at": datetime.utcnow()
    }
    
    result = await submissions_collection.insert_one(submission)
    submission_id = str(result.inserted_id)
    
    # Run plagiarism detection using ML
    comparisons = []
    plagiarism_detected = False
    high_risk_count = 0
    medium_risk_count = 0
    low_risk_count = 0
    
    # Compare each pair of students
    for i in range(len(uploaded_files)):
        for j in range(i + 1, len(uploaded_files)):
            student1 = uploaded_files[i]
            student2 = uploaded_files[j]
            
            # Calculate actual similarity using ML algorithms
            similarity_data = calculate_similarity(
                student1["content"],
                student2["content"]
            )
            
            similarity = similarity_data["similarity"]
            risk_level = get_risk_level(similarity)
            
            # Count risk levels
            if risk_level == "high":
                high_risk_count += 1
                plagiarism_detected = True
            elif risk_level == "medium":
                medium_risk_count += 1
            else:
                low_risk_count += 1
            
            comparison = {
                "student1_id": student1["student_id"],
                "student1_file": student1["filename"],
                "student2_id": student2["student_id"],
                "student2_file": student2["filename"],
                "similarity_score": round(similarity, 4),
                "risk_level": risk_level,
                # Detailed metrics for statistics page
                "sim_lex": round(similarity_data["sim_lex"], 4),
                "sim_ast": round(similarity_data["sim_ast"], 4),
                "jaccard": round(similarity_data["jaccard"], 4),
                "seqmatch": round(similarity_data["seqmatch"], 4)
            }
            comparisons.append(comparison)
    
    # Create plagiarism report with actual results
    report = {
        "user_id": current_user["_id"],
        "submission_id": submission_id,
        "assignment_name": assignment_name,
        "total_students": len(uploaded_files),
        "total_comparisons": len(comparisons),
        "comparisons": comparisons,
        "plagiarism_detected": plagiarism_detected,
        "high_risk_count": high_risk_count,
        "medium_risk_count": medium_risk_count,
        "low_risk_count": low_risk_count,
        "created_at": datetime.utcnow()
    }
    
    await reports_collection.insert_one(report)
    
    # Update submission status
    await submissions_collection.update_one(
        {"_id": ObjectId(submission_id)},
        {"$set": {"status": "completed"}}
    )
    
    return {
        "message": "Files uploaded and analyzed successfully",
        "submission_id": submission_id,
        "total_students": len(uploaded_files),
        "total_comparisons": len(comparisons),
        "plagiarism_detected": plagiarism_detected
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
            "files": submission["files"],  # Include files array
            "status": submission["status"],
            "created_at": submission["created_at"].isoformat(),
            "timestamp": submission["created_at"].strftime("%Y-%m-%d %H:%M:%S")
        })
    
    return submissions  # Return array directly

@app.delete("/api/submissions/{submission_id}")
async def delete_submission(
    submission_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete a submission and its associated reports (cascade delete)"""
    submissions_collection = get_submissions_collection()
    reports_collection = get_reports_collection()
    
    if submissions_collection is None or reports_collection is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    # First, check if submission exists and belongs to user
    submission = await submissions_collection.find_one({
        "_id": ObjectId(submission_id),
        "user_id": current_user["_id"]
    })
    
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")
    
    # Delete the submission
    result = await submissions_collection.delete_one({
        "_id": ObjectId(submission_id),
        "user_id": current_user["_id"]
    })
    
    # Cascade delete: Delete all associated reports
    reports_result = await reports_collection.delete_many({
        "submission_id": submission_id,
        "user_id": current_user["_id"]
    })
    
    return {
        "message": "Submission and associated reports deleted successfully",
        "submission_deleted": result.deleted_count,
        "reports_deleted": reports_result.deleted_count
    }

# ============= REPORTS ROUTES (Protected) =============

@app.get("/api/reports")
async def get_user_reports(current_user: dict = Depends(get_current_user)):
    """Get all plagiarism reports for current user"""
    reports_collection = get_reports_collection()
    if reports_collection is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    cursor = reports_collection.find(
        {"user_id": current_user["_id"]}
    ).sort("created_at", -1)
    
    reports = []
    async for report in cursor:
        reports.append({
            "id": str(report["_id"]),
            "submission_id": report["submission_id"],
            "assignment_name": report["assignment_name"],
            "total_students": report["total_students"],
            "total_comparisons": report["total_comparisons"],
            "plagiarism_detected": report["plagiarism_detected"],
            "high_risk_count": report["high_risk_count"],
            "medium_risk_count": report["medium_risk_count"],
            "low_risk_count": report["low_risk_count"],
            "created_at": report["created_at"].isoformat(),
            "timestamp": report["created_at"].strftime("%Y-%m-%d %H:%M:%S")
        })
    
    return reports

@app.get("/api/reports/{report_id}")
async def get_report_details(
    report_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get detailed plagiarism report with all comparisons"""
    reports_collection = get_reports_collection()
    if reports_collection is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    report = await reports_collection.find_one({
        "_id": ObjectId(report_id),
        "user_id": current_user["_id"]
    })
    
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    
    return {
        "id": str(report["_id"]),
        "assignment_name": report["assignment_name"],
        "total_students": report["total_students"],
        "total_comparisons": report["total_comparisons"],
        "plagiarism_detected": report["plagiarism_detected"],
        "high_risk_count": report["high_risk_count"],
        "medium_risk_count": report["medium_risk_count"],
        "low_risk_count": report["low_risk_count"],
        "comparisons": report["comparisons"],  # Full comparison details
        "created_at": report["created_at"].isoformat()
    }

# ============= STATISTICS ROUTES (Protected) =============

@app.get("/api/statistics")
async def get_user_statistics(current_user: dict = Depends(get_current_user)):
    """Get statistics for current user with detailed similarity data"""
    submissions_collection = get_submissions_collection()
    reports_collection = get_reports_collection()
    
    if submissions_collection is None or reports_collection is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    # Get submission count
    total_submissions = await submissions_collection.count_documents(
        {"user_id": current_user["_id"]}
    )
    
    # Get report count
    total_reports = await reports_collection.count_documents(
        {"user_id": current_user["_id"]}
    )
    
    # Calculate totals and collect all comparisons
    total_comparisons = 0
    high_risk_count = 0
    medium_risk_count = 0
    low_risk_count = 0
    total_students_set = set()
    
    # Collect all similarities for frontend
    similarities = []
    student_stats = {}  # Track stats per student
    assignment_stats = {}  # Track stats per assignment
    high_risk_pairs = []
    
    reports_cursor = reports_collection.find({"user_id": current_user["_id"]})
    async for report in reports_cursor:
        assignment_name = report.get("assignment_name", "Unknown")
        total_comparisons += report.get("total_comparisons", 0)
        high_risk_count += report.get("high_risk_count", 0)
        medium_risk_count += report.get("medium_risk_count", 0)
        low_risk_count += report.get("low_risk_count", 0)
        
        # Initialize assignment stats
        if assignment_name not in assignment_stats:
            assignment_stats[assignment_name] = {
                "assignment": assignment_name,
                "total_pairs": 0,
                "high_risk": 0,
                "medium_risk": 0,
                "low_risk": 0,
                "similarities": []
            }
        
        # Process each comparison
        for comp in report.get("comparisons", []):
            student1 = comp.get("student1_id", "")
            student2 = comp.get("student2_id", "")
            similarity = comp.get("similarity_score", 0.0) * 100  # Convert to percentage
            risk_level = comp.get("risk_level", "low")
            
            # Add to students set
            total_students_set.add(student1)
            total_students_set.add(student2)
            
            # Track student stats
            for student in [student1, student2]:
                if student not in student_stats:
                    student_stats[student] = {
                        "student_id": student,
                        "high_risk_count": 0,
                        "medium_risk_count": 0,
                        "total_submissions": 0,
                        "similarities": []
                    }
            
            # Add similarity data for frontend
            similarity_entry = {
                "student1": student1,
                "student2": student2,
                "assignment": assignment_name,
                "similarity": similarity,
                "sim_lex": comp.get("sim_lex", 0.0) * 100,
                "sim_ast": comp.get("sim_ast", 0.0) * 100,
                "jaccard": comp.get("jaccard", 0.0) * 100,
                "seqmatch": comp.get("seqmatch", 0.0) * 100,
                "risk_level": risk_level
            }
            similarities.append(similarity_entry)
            
            # Update student stats
            if risk_level == "high":
                student_stats[student1]["high_risk_count"] += 1
                student_stats[student2]["high_risk_count"] += 1
                high_risk_pairs.append(similarity_entry)
            elif risk_level == "medium":
                student_stats[student1]["medium_risk_count"] += 1
                student_stats[student2]["medium_risk_count"] += 1
            
            for student in [student1, student2]:
                student_stats[student]["similarities"].append(similarity)
            
            # Update assignment stats
            assignment_stats[assignment_name]["total_pairs"] += 1
            assignment_stats[assignment_name]["similarities"].append(similarity)
            if risk_level == "high":
                assignment_stats[assignment_name]["high_risk"] += 1
            elif risk_level == "medium":
                assignment_stats[assignment_name]["medium_risk"] += 1
            else:
                assignment_stats[assignment_name]["low_risk"] += 1
    
    # Calculate top suspects
    top_suspects = []
    for student_id, stats in student_stats.items():
        if stats["similarities"]:
            avg_similarity = sum(stats["similarities"]) / len(stats["similarities"])
            top_suspects.append({
                "student_id": student_id,
                "high_risk_count": stats["high_risk_count"],
                "medium_risk_count": stats["medium_risk_count"],
                "total_submissions": len(stats["similarities"]),
                "avg_similarity": avg_similarity
            })
    
    # Sort top suspects by high risk count, then avg similarity
    top_suspects.sort(key=lambda x: (x["high_risk_count"], x["avg_similarity"]), reverse=True)
    top_suspects = top_suspects[:10]  # Top 10
    
    # Format assignment stats
    assignment_stats_list = []
    for assignment_name, stats in assignment_stats.items():
        avg_sim = sum(stats["similarities"]) / len(stats["similarities"]) if stats["similarities"] else 0
        assignment_stats_list.append({
            "assignment": assignment_name,
            "total_pairs": stats["total_pairs"],
            "high_risk": stats["high_risk"],
            "medium_risk": stats["medium_risk"],
            "low_risk": stats["low_risk"],
            "avg_similarity": avg_sim
        })
    
    # Get recent submissions
    recent_cursor = submissions_collection.find(
        {"user_id": current_user["_id"]}
    ).sort("created_at", -1).limit(5)
    
    recent_submissions = []
    async for sub in recent_cursor:
        recent_submissions.append({
            "assignment_name": sub["assignment_name"],
            "total_students": sub.get("total_students", 0),
            "status": sub.get("status", "unknown"),
            "created_at": sub["created_at"].isoformat()
        })
    
    return {
        "total_submissions": total_submissions,
        "total_reports": total_reports,
        "total_comparisons": total_comparisons,
        "total_students": len(total_students_set),
        "high_risk_count": high_risk_count,
        "medium_risk_count": medium_risk_count,
        "low_risk_count": low_risk_count,
        "high_risk_pairs": high_risk_pairs,
        "similarities": similarities,  # All comparisons with detailed metrics
        "top_suspects": top_suspects,  # Ranked list of students
        "assignment_stats": assignment_stats_list,  # Per-assignment breakdown
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
