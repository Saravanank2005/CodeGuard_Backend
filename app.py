# backend/app.py
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import joblib, io, time, os
import numpy as np
import pandas as pd
from difflib import SequenceMatcher
import tokenize as py_tokenize
import keyword, re, ast

# --- Paths ---
APP_ROOT = Path(__file__).parent
SUB_DIR = APP_ROOT / "submissions"
SUB_DIR.mkdir(exist_ok=True)
FRONTEND_DIR = APP_ROOT.parent / "frontend"

# --- Load model bundle (with graceful fallback) ---
MODEL_OK = False
lex_vectorizer = None
ast_vectorizer = None
clf = None
feature_cols = None
model_path = str(APP_ROOT / "../models/plagiarism_model_bundle.joblib")
try:
    bundle = joblib.load(model_path)
    lex_vectorizer = bundle.get("lex_vectorizer")
    ast_vectorizer = bundle.get("ast_vectorizer")
    clf = bundle.get("model")
    feature_cols = bundle.get("feature_cols")
    if clf is not None:
        MODEL_OK = True
    else:
        print(f"Model file loaded but 'model' key missing: {model_path}")
except EOFError:
    print(f"Model file appears to be empty or corrupted: {model_path}")
except Exception as e:
    print(f"Failed to load model bundle from {model_path}: {e}")


# --- FastAPI app ---
app = FastAPI(title="Plagiarism Checker MVP")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helper functions ---
def tokenize_normalized(source: str):
    toks = []
    reader = io.BytesIO(source.encode('utf-8')).readline
    try:
        for tok in py_tokenize.tokenize(reader):
            ttype = tok.type
            tstr = tok.string
            if ttype == py_tokenize.NAME:
                if keyword.iskeyword(tstr):
                    toks.append(f"KW_{tstr}")
                else:
                    toks.append("VAR")
            elif ttype == py_tokenize.NUMBER:
                toks.append("NUM")
            elif ttype == py_tokenize.STRING:
                toks.append("STR")
            elif ttype == py_tokenize.OP:
                toks.append(tstr)
    except Exception:
        for part in re.findall(r"[A-Za-z_]+|\d+|\S", source):
            if keyword.iskeyword(part):
                toks.append(f"KW_{part}")
            elif part.isdigit():
                toks.append("NUM")
            elif re.match(r"[A-Za-z_]", part):
                toks.append("VAR")
            else:
                toks.append(part)
    return toks

def ast_node_sequence(source: str):
    try:
        tree = ast.parse(source)
        return [type(n).__name__ for n in ast.walk(tree)]
    except Exception:
        return ["ParseError"]

def cosine_sim_from_vectors(vi, vj):
    num = vi.multiply(vj).sum()
    den = (np.sqrt(vi.multiply(vi).sum()) * np.sqrt(vj.multiply(vj).sum()))
    return float(num / den) if den else 0.0

def jaccard_set(a, b):
    u = a | b
    return 0.0 if not u else len(a & b) / len(u)

def compare_against_all(new_code: str, existing_records: list, top_k=10):
    results = []
    toks_new = tokenize_normalized(new_code)
    lex_new = " ".join(toks_new)
    ast_new = " ".join(ast_node_sequence(new_code))
    v_lex_new = lex_vectorizer.transform([lex_new])
    v_ast_new = ast_vectorizer.transform([ast_new])
    set_new = set(toks_new)

    for r in existing_records:
        code_j = r["code"]
        toks_j = tokenize_normalized(code_j)
        lex_j = " ".join(toks_j)
        ast_j = " ".join(ast_node_sequence(code_j))
        v_lex_j = lex_vectorizer.transform([lex_j])
        v_ast_j = ast_vectorizer.transform([ast_j])

        sim_lex = cosine_sim_from_vectors(v_lex_new, v_lex_j)
        sim_ast = cosine_sim_from_vectors(v_ast_new, v_ast_j)
        jac = jaccard_set(set_new, set(toks_j))
        sm = SequenceMatcher(None, new_code, code_j).ratio()
        len_norm = abs(len(toks_new) - len(toks_j)) / max(len(toks_new), len(toks_j), 1)

        feat = np.array([[sim_lex, sim_ast, jac, sm, len_norm]])
        prob = float(clf.predict_proba(feat)[:,1][0])
        results.append({
            "filename": r["filename"],
            "sim_lex": sim_lex,
            "sim_ast": sim_ast,
            "jaccard": jac,
            "seqmatch": sm,
            "len_diff_norm": len_norm,
            "probability": prob
        })
    results = sorted(results, key=lambda x: x["probability"], reverse=True)
    return results[:top_k]

def load_all_submissions():
    records = []
    for p in sorted(SUB_DIR.glob("*.py")):
        records.append({"filename": p.name, "code": p.read_text(encoding="utf-8")})
    return records

# --- API endpoints ---
@app.get("/", response_class=HTMLResponse)
def home():
    """Serve the modern frontend interface"""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    else:
        # Fallback if frontend not found
        html_content = """
        <html>
            <head>
                <title>Python Plagiarism Checker</title>
            </head>
            <body>
                <h1>Welcome to the Python Plagiarism Checker!</h1>
                <p>Frontend files not found. Please ensure the 'frontend' folder exists.</p>
                <p>Visit <a href="/docs">Swagger Docs</a> for API testing.</p>
            </body>
        </html>
        """
        return HTMLResponse(content=html_content, status_code=200)

@app.get("/style.css")
def get_css():
    """Serve the CSS file"""
    css_path = FRONTEND_DIR / "style.css"
    if css_path.exists():
        return FileResponse(str(css_path), media_type="text/css")
    raise HTTPException(status_code=404, detail="CSS file not found")

@app.get("/script.js")
def get_js():
    """Serve the JavaScript file"""
    js_path = FRONTEND_DIR / "script.js"
    if js_path.exists():
        return FileResponse(str(js_path), media_type="application/javascript")
    raise HTTPException(status_code=404, detail="JavaScript file not found")

@app.post("/upload")
async def upload_code(student_id: str = Form(...), file: UploadFile = File(...)):
    if not MODEL_OK:
        raise HTTPException(status_code=503, detail="Model not available. Please restore the model bundle before using upload/analyze features.")

    if not file.filename.endswith(".py"):
        raise HTTPException(status_code=400, detail="Only .py files accepted")
    
    # Include student_id in the filename for better tracking
    # Format: timestamp_assignment_studentID.py
    # Remove .py extension from original filename to avoid double extension
    base_filename = file.filename[:-3] if file.filename.endswith('.py') else file.filename
    fname = f"{int(time.time())}_{base_filename}_{student_id}.py"
    
    path = SUB_DIR / fname
    contents = await file.read()
    path.write_bytes(contents)

    existing = load_all_submissions()
    existing = [r for r in existing if r["filename"] != path.name]

    new_code = contents.decode("utf-8", errors="ignore")
    top = compare_against_all(new_code, existing, top_k=10)

    # Save a small report file next to submission (optional)
    rep_path = path.with_suffix(".report.json")
    import json
    rep_path.write_text(json.dumps(top, indent=2))

    return JSONResponse({"filename": path.name, "top_matches": top})

@app.delete("/submissions/{filename}")
async def delete_submission(filename: str):
    """Delete a submission file and its report"""
    try:
        # Delete the Python file
        file_path = SUB_DIR / filename
        if file_path.exists() and file_path.suffix == ".py":
            file_path.unlink()
            
            # Also delete the report file if it exists
            report_path = file_path.with_suffix(".report.json")
            if report_path.exists():
                report_path.unlink()
            
            return JSONResponse({
                "success": True, 
                "message": f"Deleted {filename}",
                "filename": filename
            })
        else:
            raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")

@app.get("/submissions")
async def list_submissions():
    """Get list of all uploaded submissions"""
    try:
        submissions = []
        for p in sorted(SUB_DIR.glob("*.py"), reverse=True):  # Most recent first
            file_stat = p.stat()
            submissions.append({
                "filename": p.name,
                "size": file_stat.st_size,
                "uploaded": int(file_stat.st_mtime),
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(file_stat.st_mtime))
            })
        return JSONResponse({"submissions": submissions, "count": len(submissions)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing files: {str(e)}")


@app.get("/health")
def health_check():
    return JSONResponse({"ok": True, "model_loaded": MODEL_OK})


@app.get("/mock_statistics")
def mock_statistics():
    """Return mock statistics for frontend testing when model is missing"""
    # Simple mock data matching the expected response shape
    mock = {
        "total_submissions": 2,
        "total_students": 2,
        "total_comparisons": 1,
        "high_risk_pairs": [],
        "similarities": [
            {
                "student1": "23it045",
                "student2": "23it044",
                "file1": "1759817233_D_factorial_student4_23it045.py",
                "file2": "1759817251_D_factorial_student3_23it044.py",
                "assignment": "D_factorial",
                "similarity": 15.4,
                "sim_lex": 47.7,
                "sim_ast": 24.3,
                "jaccard": 36.8,
                "seqmatch": 32.4,
                "risk_level": "low"
            }
        ],
        "top_suspects": [],
        "student_stats": [],
        "assignment_stats": []
    }
    return JSONResponse(mock)

@app.get("/statistics")
async def get_statistics():
    """Get comprehensive statistics about plagiarism patterns"""
    if not MODEL_OK:
        raise HTTPException(status_code=503, detail="Model not available. Statistics require the trained model bundle.")

    try:
        import json
        
        # Load all submissions
        all_files = list(SUB_DIR.glob("*.py"))
        
        # Analyze all pairs
        similarities = []
        student_stats = {}
        high_risk_pairs = []
        
        for i, file1 in enumerate(all_files):
            code1 = file1.read_text(encoding="utf-8")
            
            # Extract student ID and assignment from filename
            # Formats supported:
            # 1. timestamp_assignment_oldname_studentID.py (e.g., 123_D_factorial_student4_23IT045.py)
            # 2. timestamp_assignment_studentID.py (e.g., 123_D_factorial_23IT045.py)
            # 3. timestamp_filename.py (old format)
            parts1 = file1.stem.split('_')
            
            if len(parts1) >= 3:
                # Student ID is always the last part
                student1 = parts1[-1]
                
                # Assignment extraction: skip timestamp (first) and studentID (last)
                # Also skip the second-to-last if it looks like "student1", "student2" etc
                assignment_parts = parts1[1:-1]  # Everything between timestamp and student ID
                
                # Remove "studentN" pattern if it exists at the end
                if assignment_parts and assignment_parts[-1].startswith('student'):
                    assignment_parts = assignment_parts[:-1]
                
                assignment1 = '_'.join(assignment_parts) if assignment_parts else "unknown"
            elif len(parts1) == 2:
                # Format: timestamp_filename.py (old format)
                student1 = parts1[-1]
                assignment1 = parts1[-1]
            else:
                student1 = file1.stem
                assignment1 = "unknown"
            
            # Initialize student stats
            if student1 not in student_stats:
                student_stats[student1] = {
                    "student_id": student1,
                    "total_submissions": 0,
                    "high_risk_count": 0,
                    "medium_risk_count": 0,
                    "avg_similarity": 0.0,
                    "submissions": []
                }
            
            student_stats[student1]["total_submissions"] += 1
            student_stats[student1]["submissions"].append({
                "filename": file1.name,
                "assignment": assignment1
            })
            
            # Compare with all other files
            for j, file2 in enumerate(all_files):
                if i >= j:  # Skip self and already compared pairs
                    continue
                    
                code2 = file2.read_text(encoding="utf-8")
                parts2 = file2.stem.split('_')
                
                # Extract student ID from file2 using same logic
                if len(parts2) >= 3:
                    student2 = parts2[-1]
                    
                    # Assignment extraction with "studentN" removal
                    assignment_parts2 = parts2[1:-1]
                    if assignment_parts2 and assignment_parts2[-1].startswith('student'):
                        assignment_parts2 = assignment_parts2[:-1]
                    assignment2 = '_'.join(assignment_parts2) if assignment_parts2 else "unknown"
                elif len(parts2) == 2:
                    student2 = parts2[-1]
                    assignment2 = parts2[-1]
                else:
                    student2 = file2.stem
                    assignment2 = "unknown"
                
                # Only compare same assignments
                if assignment1 != assignment2:
                    continue
                
                # Calculate similarity
                toks1 = tokenize_normalized(code1)
                toks2 = tokenize_normalized(code2)
                lex1 = " ".join(toks1)
                lex2 = " ".join(toks2)
                ast1 = " ".join(ast_node_sequence(code1))
                ast2 = " ".join(ast_node_sequence(code2))
                
                v_lex1 = lex_vectorizer.transform([lex1])
                v_lex2 = lex_vectorizer.transform([lex2])
                v_ast1 = ast_vectorizer.transform([ast1])
                v_ast2 = ast_vectorizer.transform([ast2])
                
                sim_lex = cosine_sim_from_vectors(v_lex1, v_lex2)
                sim_ast = cosine_sim_from_vectors(v_ast1, v_ast2)
                jac = jaccard_set(set(toks1), set(toks2))
                sm = SequenceMatcher(None, code1, code2).ratio()
                len_norm = abs(len(toks1) - len(toks2)) / max(len(toks1), len(toks2), 1)
                
                feat = np.array([[sim_lex, sim_ast, jac, sm, len_norm]])
                prob = float(clf.predict_proba(feat)[:,1][0])
                
                similarity_data = {
                    "student1": student1,
                    "student2": student2,
                    "file1": file1.name,
                    "file2": file2.name,
                    "assignment": assignment1,
                    "similarity": prob * 100,
                    "sim_lex": sim_lex * 100,
                    "sim_ast": sim_ast * 100,
                    "jaccard": jac * 100,
                    "seqmatch": sm * 100,
                    "risk_level": "high" if prob > 0.8 else "medium" if prob > 0.5 else "low"
                }
                
                similarities.append(similarity_data)
                
                # Track high-risk pairs
                if prob > 0.8:
                    high_risk_pairs.append(similarity_data)
                    student_stats[student1]["high_risk_count"] += 1
                    if student2 in student_stats:
                        student_stats[student2]["high_risk_count"] += 1
                elif prob > 0.5:
                    student_stats[student1]["medium_risk_count"] += 1
                    if student2 in student_stats:
                        student_stats[student2]["medium_risk_count"] += 1
        
        # Calculate average similarities for each student
        for student_id, stats in student_stats.items():
            student_sims = [s["similarity"] for s in similarities 
                          if s["student1"] == student_id or s["student2"] == student_id]
            stats["avg_similarity"] = sum(student_sims) / len(student_sims) if student_sims else 0.0
        
        # Sort similarities by risk
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Get top plagiarism suspects
        suspects = sorted(student_stats.values(), 
                         key=lambda x: (x["high_risk_count"], x["avg_similarity"]), 
                         reverse=True)[:10]
        
        # Assignment-wise statistics
        assignment_stats = {}
        for sim in similarities:
            assign = sim["assignment"]
            if assign not in assignment_stats:
                assignment_stats[assign] = {
                    "assignment": assign,
                    "total_pairs": 0,
                    "high_risk": 0,
                    "medium_risk": 0,
                    "low_risk": 0,
                    "avg_similarity": 0.0
                }
            assignment_stats[assign]["total_pairs"] += 1
            if sim["risk_level"] == "high":
                assignment_stats[assign]["high_risk"] += 1
            elif sim["risk_level"] == "medium":
                assignment_stats[assign]["medium_risk"] += 1
            else:
                assignment_stats[assign]["low_risk"] += 1
        
        for assign, stats in assignment_stats.items():
            assign_sims = [s["similarity"] for s in similarities if s["assignment"] == assign]
            stats["avg_similarity"] = sum(assign_sims) / len(assign_sims) if assign_sims else 0.0
        
        return JSONResponse({
            "total_submissions": len(all_files),
            "total_students": len(student_stats),
            "total_comparisons": len(similarities),
            "high_risk_pairs": len(high_risk_pairs),
            "similarities": similarities[:50],  # Top 50 most similar pairs
            "high_risk_pairs": high_risk_pairs,
            "top_suspects": suspects,
            "student_stats": list(student_stats.values()),
            "assignment_stats": list(assignment_stats.values())
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating statistics: {str(e)}")

# --- Run app ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
