from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.orm import sessionmaker, Session, relationship, declarative_base
import os
import pdfplumber
import re
import shutil
import pytesseract
from PIL import Image
import secrets
from fastapi.security import HTTPBasic, HTTPBasicCredentials

# --- SECURITY (ADMIN LOGIN) ---
security = HTTPBasic()
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "cuadmin123")

def authenticate_admin(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, ADMIN_USERNAME)
    correct_password = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=401, detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

# --- DATABASE SETUP (Cloud PostgreSQL) ---
# It reads from the cloud environment, or falls back to local SQLite for testing
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./calcutta_university.db")
if DATABASE_URL.startswith("postgres://"): 
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Student(Base):
    __tablename__ = "students"
    registration_no = Column(String, primary_key=True, index=True)
    roll_no = Column(String)
    name = Column(String)
    admission_year = Column(String)
    overall_cgpa = Column(String) 
    overall_grade = Column(String)
    
    # NEW FIELDS: Marksheet Collection & Post-Graduation
    marksheet_collected_date = Column(String, default="Not Collected") 
    post_grad_status = Column(String, default="Pending") # Job, Higher Study
    post_grad_details = Column(String, default="") # Company, College
    proof_document_path = Column(String, default="") # Path to uploaded proof
    
    semesters = relationship("SemesterRecord", back_populates="student")

class SemesterRecord(Base):
    __tablename__ = "semester_records"
    id = Column(Integer, primary_key=True, index=True)
    registration_no = Column(String, ForeignKey("students.registration_no"))
    semester = Column(String)
    year = Column(String)
    marks_obtained = Column(String)
    sgpa = Column(String)
    student = relationship("Student", back_populates="semesters")

Base.metadata.create_all(bind=engine)
os.makedirs("uploads", exist_ok=True)

# --- FASTAPI APP ---
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

# --- WEB UI ROUTE (Secured with Login) ---
@app.get("/")
def serve_frontend(username: str = Depends(authenticate_admin)):
    return FileResponse("index.html")

# --- API ENDPOINTS ---
@app.post("/api/admin/upload-marksheet")
async def upload_marksheet(file: UploadFile = File(...), db: Session = Depends(get_db), user: str = Depends(authenticate_admin)):
    temp_pdf_path = f"temp_{file.filename}"
    with open(temp_pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    extracted_count = 0
    try:
        with pdfplumber.open(temp_pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                if len(text.strip()) < 50:
                    img = page.to_image(resolution=300).original
                    text = pytesseract.image_to_string(img)

                name_match = re.search(r'Name[\s\.\:]+([A-Za-z\s]+)(?=\n|Registration|Roll)', text, re.IGNORECASE)
                reg_match = re.search(r'Registration\s*No[\.\:\s]*([A-Za-z0-9\-]+)', text, re.IGNORECASE)
                roll_match = re.search(r'Roll[\s\&]*No[\.\:\s]*([A-Za-z0-9\-]+)', text, re.IGNORECASE)

                if not reg_match: continue 
                    
                reg_no = reg_match.group(1).strip()
                name = name_match.group(1).strip() if name_match else "Unknown"
                roll_no = roll_match.group(1).strip() if roll_match else "Unknown"
                try: admission_year = "20" + reg_no.split("-")[-1]
                except: admission_year = "Unknown"

                # Save to Database (Basic info)
                existing_student = db.query(Student).filter(Student.registration_no == reg_no).first()
                if not existing_student:
                    student = Student(
                        registration_no=reg_no, roll_no=roll_no, name=name, 
                        admission_year=admission_year, overall_cgpa="Check System", overall_grade="-"
                    )
                    db.add(student)
                    extracted_count += 1
        db.commit()
    finally:
        if os.path.exists(temp_pdf_path): os.remove(temp_pdf_path)

    return {"message": f"Successfully extracted {extracted_count} student records via Cloud OCR!"}

@app.post("/api/admin/update-status/{reg_no}")
async def update_student_status(
    reg_no: str,
    collection_date: str = Form(...),
    status: str = Form(...),
    details: str = Form(...),
    proof_file: UploadFile = File(None),
    db: Session = Depends(get_db),
    user: str = Depends(authenticate_admin)
):
    student = db.query(Student).filter(Student.registration_no == reg_no).first()
    if not student: raise HTTPException(status_code=404, detail="Student not found")

    student.marksheet_collected_date = collection_date
    student.post_grad_status = status
    student.post_grad_details = details

    if proof_file:
        safe_filename = f"{reg_no}_proof.{proof_file.filename.split('.')[-1]}"
        file_path = os.path.join("uploads", safe_filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(proof_file.file, buffer)
        student.proof_document_path = file_path

    db.commit()
    return {"message": "Marksheet Collection & Proof updated successfully!"}

@app.get("/api/student/{reg_no}")
def get_student(reg_no: str, db: Session = Depends(get_db), user: str = Depends(authenticate_admin)):
    student = db.query(Student).filter(Student.registration_no == reg_no).first()
    if not student: raise HTTPException(status_code=404, detail="Student not found")
    
    return {
        "name": student.name, "reg_no": student.registration_no, "roll_no": student.roll_no,
        "collection_date": student.marksheet_collected_date,
        "status": student.post_grad_status, "details": student.post_grad_details,
        "proof": student.proof_document_path
    }