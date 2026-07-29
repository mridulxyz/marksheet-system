import os
import re
import shutil
import secrets
import json
import base64
import fitz  # PyMuPDF
import openai
from PIL import Image

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Boolean, func, text
from sqlalchemy.orm import sessionmaker, Session, relationship, declarative_base

# --- SECURITY (ADMIN LOGIN) ---
security = HTTPBasic()
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "cuadmin123")

def authenticate_admin(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, ADMIN_USERNAME)
    correct_password = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

# --- OPENAI CLIENT SETUP ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ai_client = openai.OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# --- DATABASE SETUP ---
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./calcutta_university.db")
if DATABASE_URL.startswith("postgres://"): 
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(
    DATABASE_URL, 
    pool_pre_ping=True,      # Auto-reconnects dropped SSL handles
    pool_recycle=300,        # Recycles connections every 5 mins
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- DATABASE MODELS ---
class Student(Base):
    __tablename__ = "students"
    registration_no = Column(String, primary_key=True, index=True)
    roll_no = Column(String)
    name = Column(String)
    admission_year = Column(String)
    course = Column(String, default="Unknown Course")
    overall_cgpa = Column(String) 
    overall_grade = Column(String)
    
    marksheet_received = Column(Boolean, default=False)
    certificate_received = Column(Boolean, default=False)
    post_grad_status = Column(String, default="Unemployed")
    post_grad_details = Column(String, default="") 
    proof_document_path = Column(String, default="") 
    
    semesters = relationship("SemesterRecord", back_populates="student", cascade="all, delete-orphan")

class SemesterRecord(Base):
    __tablename__ = "semester_records"
    id = Column(Integer, primary_key=True, index=True)
    registration_no = Column(String, ForeignKey("students.registration_no"))
    semester = Column(String)
    year = Column(String)
    full_marks = Column(String, default="400")
    marks_obtained = Column(String)
    credit = Column(String, default="20")
    sgpa = Column(String)
    
    student = relationship("Student", back_populates="semesters")

os.makedirs("uploads", exist_ok=True)

# --- OPENAI GPT-4o-MINI VISION PARSER ---

def parse_marksheet_with_openai_vision(page):
    pix = page.get_pixmap(dpi=130)
    img_bytes = pix.tobytes("jpeg")
    base64_image = base64.b64encode(img_bytes).decode('utf-8')

    prompt = """
    You are an expert transcript parser for Calcutta University Grade Sheets. Analyze the provided grade sheet image and return ONLY a valid JSON object in this exact schema:

    {
      "registration_no": "424-1211-0240-19",
      "roll_no": "192424-11-0044",
      "name": "SWAGATA PURKAIT",
      "course": "B.A. (Honours) Examination (Under CBCS)",
      "overall_cgpa": "6.819",
      "overall_grade": "B+",
      "semesters": [
        {"semester": "I", "year": "2019", "full_marks": "400", "marks": "248", "credit": "20", "sgpa": "5.624"},
        {"semester": "II", "year": "2020", "full_marks": "400", "marks": "310", "credit": "20", "sgpa": "7.705"},
        {"semester": "III", "year": "2022", "full_marks": "500", "marks": "330", "credit": "26", "sgpa": "6.899"},
        {"semester": "IV", "year": "2023", "full_marks": "500", "marks": "372", "credit": "26", "sgpa": "7.367"},
        {"semester": "V", "year": "2023", "full_marks": "400", "marks": "263", "credit": "24", "sgpa": "6.527"},
        {"semester": "VI", "year": "2024", "full_marks": "400", "marks": "270", "credit": "24", "sgpa": "6.686"}
      ]
    }

    STRICT CRITICAL RULES:
    1. OVERALL GRADE: Look ONLY at row 'VI' in the bottom summary table under column 'Letter Grade' (e.g. B+, A+, A, B, C+, C, D, O). Do NOT read status 'P' (Passed) or subject grades from top subject tables.
    2. OVERALL CGPA: Look ONLY at row 'VI' in the bottom summary table under column 'CGPA' (e.g. 6.819).
    3. SEMESTERS: You MUST extract ALL cleared semester rows (I, II, III, IV, V, VI) from the bottom summary table. Each item in 'semesters' MUST contain: 'semester' (I, II, III, IV, V, VI), 'year', 'full_marks', 'marks', 'credit', 'sgpa'.
    4. IF SEMESTER NOT CLEARED: If 'Semester not cleared' is printed in Remarks, set 'overall_cgpa': 'N.A.' and 'overall_grade': 'Fail / Semester Not Cleared'.
    5. COURSE TITLE: Omit 'Semester - VI' from course title. Use 'B.A. (Honours) Examination (Under CBCS)'.
    """

    response = ai_client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ]
    )

    result_json = response.choices[0].message.content
    return json.loads(result_json)

# --- FASTAPI APP SETUP ---
app = FastAPI(title="Shyampur Siddheswari Mahavidyalaya Marksheet Portal")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- SAFE STARTUP DB CREATION & MIGRATION ---
@app.on_event("startup")
def startup_db_setup():
    try:
        Base.metadata.create_all(bind=engine)
        with engine.connect() as conn:
            if "postgresql" in DATABASE_URL:
                conn.execute(text("ALTER TABLE students ADD COLUMN IF NOT EXISTS course VARCHAR DEFAULT 'Unknown Course';"))
                conn.execute(text("ALTER TABLE students ADD COLUMN IF NOT EXISTS marksheet_received BOOLEAN DEFAULT FALSE;"))
                conn.execute(text("ALTER TABLE students ADD COLUMN IF NOT EXISTS certificate_received BOOLEAN DEFAULT FALSE;"))
                conn.execute(text("ALTER TABLE students ADD COLUMN IF NOT EXISTS post_grad_status VARCHAR DEFAULT 'Unemployed';"))
                conn.execute(text("ALTER TABLE students ADD COLUMN IF NOT EXISTS post_grad_details VARCHAR DEFAULT '';"))
                conn.execute(text("ALTER TABLE students ADD COLUMN IF NOT EXISTS proof_document_path VARCHAR DEFAULT '';"))
                conn.execute(text("ALTER TABLE semester_records ADD COLUMN IF NOT EXISTS full_marks VARCHAR DEFAULT '400';"))
                conn.execute(text("ALTER TABLE semester_records ADD COLUMN IF NOT EXISTS credit VARCHAR DEFAULT '20';"))
                conn.commit()
            elif "sqlite" in DATABASE_URL:
                columns = [row[1] for row in conn.execute(text("PRAGMA table_info(students);")).fetchall()]
                if columns:
                    if "course" not in columns: conn.execute(text("ALTER TABLE students ADD COLUMN course VARCHAR DEFAULT 'Unknown Course';"))
                    if "marksheet_received" not in columns: conn.execute(text("ALTER TABLE students ADD COLUMN marksheet_received BOOLEAN DEFAULT 0;"))
                    if "certificate_received" not in columns: conn.execute(text("ALTER TABLE students ADD COLUMN certificate_received BOOLEAN DEFAULT 0;"))
                    if "post_grad_status" not in columns: conn.execute(text("ALTER TABLE students ADD COLUMN post_grad_status VARCHAR DEFAULT 'Unemployed';"))
                    if "post_grad_details" not in columns: conn.execute(text("ALTER TABLE students ADD COLUMN post_grad_details VARCHAR;"))
                    if "proof_document_path" not in columns: conn.execute(text("ALTER TABLE students ADD COLUMN proof_document_path VARCHAR;"))
                
                sem_columns = [row[1] for row in conn.execute(text("PRAGMA table_info(semester_records);")).fetchall()]
                if sem_columns:
                    if "full_marks" not in sem_columns: conn.execute(text("ALTER TABLE semester_records ADD COLUMN full_marks VARCHAR DEFAULT '400';"))
                    if "credit" not in sem_columns: conn.execute(text("ALTER TABLE semester_records ADD COLUMN credit VARCHAR DEFAULT '20';"))
                conn.commit()
    except Exception as e:
        print(f"Startup Migration Note: {e}")

# --- FRONTEND ROUTE ---
@app.get("/")
def serve_frontend(username: str = Depends(authenticate_admin)):
    return FileResponse("index.html")

# --- API ENDPOINTS ---

@app.post("/api/admin/upload-marksheet")
async def upload_marksheet(
    file: UploadFile = File(...), 
    db: Session = Depends(get_db), 
    user: str = Depends(authenticate_admin)
):
    if not ai_client:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY environment variable is missing on Render.")

    temp_pdf_path = f"temp_{file.filename}"
    with open(temp_pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    extracted_count = 0
    debug_logs = []
    
    try:
        doc = fitz.open(temp_pdf_path)
        
        for page_num in range(len(doc)):
            try:
                page = doc[page_num]
                
                # Parse page using OpenAI Vision
                data = parse_marksheet_with_openai_vision(page)

                reg_no = data.get("registration_no")
                if not reg_no or reg_no == "null":
                    debug_logs.append(f"Page {page_num+1}: Could not find Registration Number.")
                    continue

                roll_no = data.get("roll_no", "Unknown")
                name = data.get("name", "Unknown Student")
                course = data.get("course", "B.A. (Honours) Examination (Under CBCS)")
                overall_cgpa = data.get("overall_cgpa", "N.A.")
                overall_grade = data.get("overall_grade", "Fail / Semester Not Cleared")
                
                # Dynamic Key Normalization for Semester Data
                raw_semesters = data.get("semesters", [])
                normalized_semesters = []
                for sem in raw_semesters:
                    if not isinstance(sem, dict): continue
                    
                    raw_s = str(sem.get("semester") or sem.get("sem") or "").strip().upper()
                    raw_s = re.sub(r'SEMESTER\s*', '', raw_s).strip()
                    if not raw_s: continue

                    normalized_semesters.append({
                        "semester": raw_s,
                        "year": str(sem.get("year") or sem.get("exam_year") or "").strip(),
                        "full_marks": str(sem.get("full_marks") or sem.get("total_marks") or "400").strip(),
                        "marks": str(sem.get("marks") or sem.get("marks_obtained") or sem.get("obtained") or "-").strip(),
                        "credit": str(sem.get("credit") or sem.get("semester_credit") or "20").strip(),
                        "sgpa": str(sem.get("sgpa") or sem.get("gpa") or "N.A.").strip()
                    })

                # Database Upsert
                existing_student = db.query(Student).filter(Student.registration_no == reg_no).first()
                if existing_student:
                    existing_student.name = name
                    existing_student.roll_no = roll_no
                    existing_student.course = course
                    existing_student.overall_cgpa = overall_cgpa
                    existing_student.overall_grade = overall_grade
                    
                    db.query(SemesterRecord).filter(SemesterRecord.registration_no == reg_no).delete()
                    for sem in normalized_semesters:
                        db.add(SemesterRecord(
                            registration_no=reg_no, 
                            semester=sem["semester"], 
                            year=sem["year"], 
                            full_marks=sem["full_marks"],
                            marks_obtained=sem["marks"], 
                            credit=sem["credit"],
                            sgpa=sem["sgpa"]
                        ))
                else:
                    admission_year = "20" + reg_no.split("-")[-1] if "-" in reg_no else "Unknown"
                    student = Student(
                        registration_no=reg_no, 
                        roll_no=roll_no, 
                        name=name, 
                        admission_year=admission_year, 
                        course=course, 
                        overall_cgpa=overall_cgpa, 
                        overall_grade=overall_grade
                    )
                    db.add(student)
                    for sem in normalized_semesters:
                        db.add(SemesterRecord(
                            registration_no=reg_no, 
                            semester=sem["semester"], 
                            year=sem["year"], 
                            full_marks=sem["full_marks"],
                            marks_obtained=sem["marks"], 
                            credit=sem["credit"],
                            sgpa=sem["sgpa"]
                        ))
                        
                extracted_count += 1
                debug_logs.append(f"Page {page_num+1} AI Success: Reg={reg_no}, Name={name}, Roll={roll_no}, CGPA={overall_cgpa}, Grade={overall_grade}, Sems={len(normalized_semesters)}")
            except Exception as page_err:
                db.rollback()
                debug_logs.append(f"Page {page_num+1} Exception: {page_err}")
                continue

        doc.close()
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"PDF extraction error: {str(e)}")
    finally:
        if os.path.exists(temp_pdf_path): 
            os.remove(temp_pdf_path)

    return {
        "message": f"Successfully extracted {extracted_count} student record(s)!",
        "debug_logs": debug_logs
    }

@app.get("/api/student/{reg_no}")
def get_student(reg_no: str, db: Session = Depends(get_db), user: str = Depends(authenticate_admin)):
    student = db.query(Student).filter(Student.registration_no == reg_no).first()
    if not student: 
        raise HTTPException(status_code=404, detail="Student record not found.")
    
    sems = db.query(SemesterRecord).filter(SemesterRecord.registration_no == reg_no).all()
    
    # Calculate Passout Year: Year of Semester VI
    passout_year = "Unknown"
    for s in sems:
        if s.semester in ["VI", "6", "VI."] and s.year:
            passout_year = s.year

    return {
        "student": {
            "name": student.name, 
            "reg_no": student.registration_no, 
            "roll_no": student.roll_no,
            "admission_year": student.admission_year, 
            "passout_year": passout_year, 
            "course": student.course,
            "cgpa": student.overall_cgpa, 
            "grade": student.overall_grade,
            "marksheet_received": student.marksheet_received, 
            "certificate_received": student.certificate_received,
            "status": student.post_grad_status, 
            "details": student.post_grad_details, 
            "proof": student.proof_document_path
        },
        "semesters": [
            {
                "semester": s.semester, 
                "year": s.year, 
                "full_marks": s.full_marks or "400",
                "marks": s.marks_obtained, 
                "credit": s.credit or "20",
                "sgpa": s.sgpa
            } 
            for s in sems
        ]
    }

@app.post("/api/admin/update-status/{reg_no}")
async def update_student_status(
    reg_no: str,
    course: str = Form(...),
    marksheet_received: bool = Form(...),
    certificate_received: bool = Form(...),
    status: str = Form(...),
    details: str = Form(""),
    proof_file: UploadFile = File(None),
    db: Session = Depends(get_db),
    user: str = Depends(authenticate_admin)
):
    student = db.query(Student).filter(Student.registration_no == reg_no).first()
    if not student: 
        raise HTTPException(status_code=404, detail="Student record not found.")

    student.course = course
    student.marksheet_received = marksheet_received
    student.certificate_received = certificate_received
    student.post_grad_status = status
    student.post_grad_details = details

    if proof_file:
        safe_filename = f"{reg_no}_proof.{proof_file.filename.split('.')[-1]}"
        file_path = os.path.join("uploads", safe_filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(proof_file.file, buffer)
        student.proof_document_path = file_path

    db.commit()
    return {"message": "Record updated successfully!"}

@app.get("/api/admin/grade-stats")
def get_grade_stats(db: Session = Depends(get_db), user: str = Depends(authenticate_admin)):
    stats = db.query(Student.course, Student.overall_grade, func.count(Student.registration_no)) \
              .group_by(Student.course, Student.overall_grade).all()
    
    result = {}
    for course, grade, count in stats:
        c = course if course else "Unknown Course"
        g = grade if grade else "Fail/NA"
        if c not in result:
            result[c] = {}
        result[c][g] = count
        
    return result

@app.get("/api/admin/all-students")
def get_all_students(db: Session = Depends(get_db), user: str = Depends(authenticate_admin)):
    return db.query(Student).all()
