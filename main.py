import os
import re
import shutil
import secrets
import json
import base64
import time
import gc
import io
from PIL import Image

# Safe Imports
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Boolean, func, text
from sqlalchemy.orm import sessionmaker, Session, relationship, declarative_base, joinedload

# --- SECURITY (MULTI-USER AUTHENTICATION) ---
security = HTTPBasic()

ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "cuadmin123")

USER1_USERNAME = os.getenv("USER1_USERNAME", "staff1")
USER1_PASSWORD = os.getenv("USER1_PASSWORD", "staff123")

USER2_USERNAME = os.getenv("USER2_USERNAME", "staff2")
USER2_PASSWORD = os.getenv("USER2_PASSWORD", "staff123")

def get_current_user(credentials: HTTPBasicCredentials = Depends(security)):
    username = credentials.username
    password = credentials.password

    if secrets.compare_digest(username, ADMIN_USERNAME) and secrets.compare_digest(password, ADMIN_PASSWORD):
        return {"username": username, "role": "admin"}

    if secrets.compare_digest(username, USER1_USERNAME) and secrets.compare_digest(password, USER1_PASSWORD):
        return {"username": username, "role": "user"}

    if secrets.compare_digest(username, USER2_USERNAME) and secrets.compare_digest(password, USER2_PASSWORD):
        return {"username": username, "role": "user"}

    raise HTTPException(
        status_code=401,
        detail="Incorrect username or password",
        headers={"WWW-Authenticate": "Basic"},
    )

def require_admin(user: dict = Depends(get_current_user)):
    if user["role"] != "admin":
        raise HTTPException(
            status_code=403,
            detail="Admin permissions required to perform this operation."
        )
    return user

# --- GEMINI CLIENT SETUP (Using Latest Google GenAI SDK) ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ai_client = genai.Client(api_key=GEMINI_API_KEY) if (genai and GEMINI_API_KEY) else None

# --- DATABASE SETUP ---
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./calcutta_university.db")
if DATABASE_URL.startswith("postgres://"): 
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(
    DATABASE_URL, 
    pool_pre_ping=True,      
    pool_recycle=300,        
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
    passout_year = Column(String, default="Nil")
    course = Column(String, default="Unknown Course")
    subject = Column(String, default="BNGA")
    overall_cgpa = Column(String) 
    overall_grade = Column(String)
    remarks = Column(String, default="Qualified")
    
    marksheet_received = Column(Boolean, default=False)
    certificate_received = Column(Boolean, default=False)
    marksheet_issue_date = Column(String, default="")
    certificate_issue_date = Column(String, default="")
    issued_by = Column(String, default="")
    
    post_grad_status = Column(String, default="Unknown") 
    post_grad_details = Column(String, default="") 
    proof_document_path = Column(String, default="") 
    pdf_document_path = Column(String, default="") 
    
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
os.makedirs("uploads/pdf_repository", exist_ok=True)

# --- GLOBAL BACKGROUND PROGRESS TRACKER ---
bg_upload_status = {
    "is_processing": False,
    "total_pages": 0,
    "processed_pages": 0,
    "extracted_count": 0,
    "status_message": "Idle",
    "filename": ""
}

# --- HELPER: AUTO-REPAIR PASSOUT YEAR ---
def auto_repair_passout_year(student_obj):
    """Dynamically fixes missing Passout Years from existing DB records to correct 'NIL' bugs."""
    py = str(student_obj.passout_year).strip()
    is_fail = ("fail" in str(student_obj.overall_grade).lower() or "not cleared" in str(student_obj.remarks).lower() or student_obj.overall_grade in ["", "None", "N.A."])
    
    if is_fail:
        if py.lower() != "nil":
            student_obj.passout_year = "Nil"
            return True
    else:
        if py.lower() in ["nil", "none", "unknown", "", "n.a.", "null"]:
            max_yr = None
            max_val = 0
            for sem in student_obj.semesters:
                if sem.year:
                    m = re.search(r'(20[1-3]\d)', str(sem.year))
                    if m:
                        val = int(m.group(1))
                        if sem.semester in ["VI", "6", "VI.", "6th"]:
                            max_yr = str(val)
                            break
                        if val > max_val:
                            max_val = val
                            max_yr = str(val)
            if max_yr:
                student_obj.passout_year = max_yr
                return True
    return False

# --- GEMINI 3.6 FLASH VISION PARSER ---
def parse_marksheet_with_gemini_vision(page):
    pix = page.get_pixmap(dpi=200)
    img_bytes = pix.tobytes("jpeg")
    
    # Load image for Gemini
    image = Image.open(io.BytesIO(img_bytes))

    prompt = """
    You are an expert transcript parser. Carefully inspect this Calcutta University Grade Sheet image with 100% precision and return ONLY a valid JSON object matching this exact schema:

    {
      "registration_no": "424-1211-0240-19",
      "roll_no": "192424-11-0044",
      "name": "SWAGATA PURKAIT",
      "course": "B.A. (Honours) Examination (Under CBCS)",
      "subject": "BNGA",
      "passout_year": "2024",
      "overall_cgpa": "6.819",
      "overall_grade": "B+",
      "remarks": "Qualified with Honours",
      "semesters": [
        {"semester": "I", "year": "2019", "full_marks": "400", "marks": "248", "credit": "20", "sgpa": "5.624"},
        {"semester": "II", "year": "2020", "full_marks": "400", "marks": "310", "credit": "20", "sgpa": "7.705"},
        {"semester": "III", "year": "2022", "full_marks": "500", "marks": "330", "credit": "26", "sgpa": "6.899"},
        {"semester": "IV", "year": "2023", "full_marks": "500", "marks": "372", "credit": "26", "sgpa": "7.367"},
        {"semester": "V", "year": "2023", "full_marks": "400", "marks": "263", "credit": "24", "sgpa": "6.527"},
        {"semester": "VI", "year": "2024", "full_marks": "400", "marks": "270", "credit": "24", "sgpa": "6.686"}
      ]
    }

    VERIFICATION RULES FOR 100% ACCURACY:
    1. SUMMARY TABLE LOCATION: Look at the table near the bottom labeled 'Semester', 'Year', 'Full Marks', 'Marks Obtained', 'Semester Credit', 'SGPA', 'Cumulative Credit', 'CGPA', 'Letter Grade', 'Remarks'.
    2. DO NOT SKIP ANY SEMESTER ROWS: Carefully read every row from Semester I to Semester VI. Do NOT omit Semester IV or any other row if present in the summary table!
    3. DO NOT READ TOP SUBJECT TABLES: Do NOT extract numbers from the course component tables above (e.g. BNGA-CC13, CC14, DSE-A4, DSE-B4). Read ONLY from the summary table at the bottom.
    4. ONLY PRESENT SEMESTERS: If a semester is empty/blank in the table (e.g. Sem IV, V, VI blank for a failed student), do NOT include it in the 'semesters' array!
    5. SUBJECT CODE: Extract subject code prefix from course code (e.g. 'BNGA' from BNGA-CC13 or 'ENGG' from ENGG-CC13). Default to 'BNGA' if not clearly stated. DO NOT return full words like 'chemistry' or 'physics'.
    6. OVERALL GRADE: Read 'overall_grade' ONLY from column 'Letter Grade' in row 'VI' of the bottom summary table (e.g. B+, A+, A, B, C+, C, D, O). Never read words like 'Good' or status 'P'.
    7. OVERALL CGPA: Read 'overall_cgpa' ONLY from column 'CGPA' in row 'VI' of the bottom summary table (e.g. 6.819).
    8. IF SEMESTER NOT CLEARED: If 'Semester not cleared' is printed in Remarks, set 'overall_cgpa': 'N.A.' and 'overall_grade': 'Fail / Semester Not Cleared'.
    9. REMARKS: Read the exact Remarks line right below the summary table (e.g. 'Qualified with Honours' or 'Semester not cleared').
    10. COURSE TITLE: Omit 'Semester - VI' from course title. Use 'B.A. (Honours) Examination (Under CBCS)'.
    11. PASSOUT YEAR: Extract the examination year from the main title at the top of the marksheet (e.g. 'Examination - 2024' -> '2024'). If the student failed or overall grade is blank, set passout_year to "Nil".
    12. FAILED/NOT CLEARED OVERRIDE: If the student failed or overall grade is blank, set overall_grade to "Fail / Semester Not Cleared", overall_cgpa to "N.A.", passout_year to "Nil", and return an empty array [] for "semesters".
    """

    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Using Gemini 3.6 Flash (The newest, most robust multimodal model)
            response = ai_client.models.generate_content(
                model='gemini-3.6-flash',
                contents=[prompt, image],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                ),
            )
            result_json = response.text
            return json.loads(result_json)
        except Exception as e:
            err_msg = str(e).lower()
            if ("429" in err_msg or "rate" in err_msg or "quota" in err_msg) and attempt < max_retries - 1:
                time.sleep((attempt + 1) * 3)
            else:
                raise e

# --- LOCAL FALLBACK PARSER ---

def extract_text_rows_from_rect(page, rect_box):
    try:
        words = page.get_text("words", clip=rect_box)
        if not words: return ""
        sorted_words = sorted(words, key=lambda w: (round(w[1] / 5.0), w[0]))

        lines = []
        current_line = []
        last_y = None

        for w in sorted_words:
            x0, y0, x1, y1, word = w[0], w[1], w[2], w[3], w[4]
            if last_y is None or abs(y0 - last_y) <= 5:
                current_line.append(word)
                last_y = y0
            else:
                lines.append(" ".join(current_line))
                current_line = [word]
                last_y = y0

        if current_line: lines.append(" ".join(current_line))
        return "\n".join(lines)
    except Exception:
        return page.get_text("text", clip=rect_box) or ""

def extract_reg_no_bulletproof(text: str):
    if not text: return None, "Text empty"
    t = re.sub(r'[—–_~]', '-', text)
    t_fixed = t.replace('O', '0').replace('o', '0').replace('Q', '0').replace('I', '1').replace('l', '1').replace('S', '5')

    match = re.search(r'(?:Registration|Regn|Reg|Registra)[^\d]{0,15}([0-9\-\s\.\/]{13,22})', t_fixed, re.IGNORECASE)
    if match:
        digits = re.sub(r'\D', '', match.group(1))
        if len(digits) >= 13:
            d = digits[:13]
            return f"{d[:3]}-{d[3:7]}-{d[7:11]}-{d[11:]}", "Reg Label"

    all_digits_clusters = re.findall(r'\b\d{13}\b', re.sub(r'\D', ' ', t_fixed))
    if all_digits_clusters:
        d = all_digits_clusters[0]
        return f"{d[:3]}-{d[3:7]}-{d[7:11]}-{d[11:]}", "13-Digit Cluster"

    return None, "No 13-digit pattern found"

def extract_roll_no_bulletproof(text: str):
    if not text: return "Unknown"
    t = re.sub(r'[—–_~]', '-', text)
    t_fixed = t.replace('O', '0').replace('o', '0').replace('Q', '0').replace('I', '1').replace('l', '1').replace('S', '5')

    all_digits_clusters = re.findall(r'\b\d{12}\b', re.sub(r'\D', ' ', t_fixed))
    if all_digits_clusters:
        d = all_digits_clusters[0]
        return f"{d[:6]}-{d[6:8]}-{d[8:]}"

    return "Unknown"

def extract_name_bulletproof(text: str):
    if not text: return "Unknown Student"
    match = re.search(r'Name\s*[\:\.]*\s*([A-Za-z\s\.]+?)(?=Registration|Regn|Roll|\d{3}\-|\n|$)', text, re.IGNORECASE)
    if match:
        raw_name = re.sub(r'[^A-Za-z\s\.]', '', match.group(1)).strip()
        if len(raw_name) > 2: return raw_name
    return "Unknown Student"

def extract_course(text: str):
    if not text: return "B.A. (Honours) Examination"
    match = re.search(r'(B\.?A\.?|B\.?Sc\.?|B\.?Com\.?)[^\n\r,]*', text, re.IGNORECASE)
    if match:
        raw_course = match.group(0).strip()
        cleaned_course = re.sub(r'Semester\s*[\-\–\_]?\s*(VI|V|IV|III|II|I|\d+)', '', raw_course, flags=re.IGNORECASE)
        return re.sub(r'\s+', ' ', cleaned_course).strip()
    return "B.A. (Honours) Examination"

def extract_passout_year(text: str):
    if not text: return "Nil"
    match = re.search(r'Examination[^\n]*?\b(20[1-3]\d)\b', text, re.IGNORECASE)
    if match:
        return match.group(1)
    
    lines = text.split('\n')[:20]
    for line in lines:
        m = re.search(r'\b(20[1-3]\d)\b', line)
        if m: return m.group(1)
            
    return "Nil"

def parse_summary_table_local(table_text: str):
    sems = []
    cgpa = "N.A."
    grade = "Fail / Semester Not Cleared"
    remarks = "Qualified"
    if not table_text: return sems, cgpa, grade, remarks

    text_clean = re.sub(r'(\d)\s*[\,\.]\s*(\d)', r'\1.\2', table_text)
    is_not_cleared = bool(re.search(r'(?:Semester\s*not\s*cleared|not\s*cleared)', text_clean, re.IGNORECASE))

    rem_match = re.search(r'Remarks\s*[\:\.]*\s*([^\n\r]+)', text_clean, re.IGNORECASE)
    if rem_match:
        remarks = rem_match.group(1).strip()

    grade_m = re.search(r'\b(A\+|B\+|C\+|A|B|C|D|O)(?!\w)', text_clean)
    if grade_m: 
        grade = grade_m.group(1)
    else:
        is_not_cleared = True

    if is_not_cleared:
        return [], "N.A.", "Fail / Semester Not Cleared", remarks

    lines = [l.strip() for l in text_clean.splitlines() if l.strip()]
    sem_keys = ['I', 'II', 'III', 'IV', 'V', 'VI']

    for line in lines:
        for sem in sem_keys:
            if re.search(rf'\b{sem}\b', line):
                years = re.findall(r'\b(20\d\d)\b', line)
                integers = [i for i in re.findall(r'\b(\d{2,3})\b', line) if i not in years]
                floats = re.findall(r'\b([1-9]\.\d{2,3})\b', line)

                if years and integers and floats:
                    s_yr = years[0]
                    s_fm = integers[0] if len(integers) >= 1 else "400"
                    s_marks = integers[1] if len(integers) >= 2 else (integers[0] if integers else "-")
                    s_cred = integers[2] if len(integers) >= 3 else "20"
                    s_sgpa = floats[0]

                    if not any(item["semester"] == sem for item in sems):
                        sems.append({
                            "semester": sem,
                            "year": s_yr,
                            "full_marks": s_fm,
                            "marks": s_marks,
                            "credit": s_cred,
                            "sgpa": s_sgpa
                        })

    sem_order = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6}
    sems = sorted(sems, key=lambda x: sem_order.get(x["semester"], 99))

    all_floats = re.findall(r'\b([1-9]\.\d{2,3})\b', text_clean)
    valid_gpas = [f for f in all_floats if 1.0 <= float(f) <= 10.0]
    if valid_gpas: cgpa = valid_gpas[-1]

    return sems, cgpa, grade, remarks

# --- BACKGROUND WORKER FOR PDFs ---

def process_large_pdf_in_background(temp_pdf_path: str, selected_course: str):
    global bg_upload_status
    db = SessionLocal()
    
    try:
        doc = fitz.open(temp_pdf_path) if fitz else []
        total_pages = len(doc)
        
        bg_upload_status["is_processing"] = True
        bg_upload_status["total_pages"] = total_pages
        bg_upload_status["processed_pages"] = 0
        bg_upload_status["extracted_count"] = 0
        bg_upload_status["status_message"] = f"Processing page 1 of {total_pages}..."
        
        ai_quota_exceeded = False
        extracted_count = 0
        
        for page_num in range(total_pages):
            try:
                page = doc[page_num]
                reg_no = None
                normalized_semesters = []
                remarks = "Qualified"
                subject = "BNGA"
                passout_year = "Nil"
                
                # --- STRATEGY 1: GEMINI VISION ---
                if ai_client and not ai_quota_exceeded:
                    try:
                        if page_num > 0: time.sleep(0.3)
                        data = parse_marksheet_with_gemini_vision(page)
                        reg_no = data.get("registration_no")
                        if not reg_no or reg_no == "null":
                            bg_upload_status["processed_pages"] = page_num + 1
                            continue

                        roll_no = data.get("roll_no", "Unknown")
                        name = data.get("name", "Unknown Student")
                        course = selected_course if (selected_course and selected_course != "AUTO") else data.get("course", "B.A. (Honours) Examination (Under CBCS)")
                        subject = data.get("subject", "BNGA")
                        passout_year = str(data.get("passout_year", "Nil")).strip()
                        
                        remarks = data.get("remarks", "Qualified with Honours")
                        overall_cgpa = data.get("overall_cgpa", "N.A.")
                        overall_grade = data.get("overall_grade", "Fail / Semester Not Cleared")

                        if "not cleared" in str(remarks).lower() or overall_grade.lower() in ["none", "null", "n.a.", "", "fail / semester not cleared"]:
                            overall_cgpa = "N.A."
                            overall_grade = "Fail / Semester Not Cleared"
                            normalized_semesters = []
                            passout_year = "Nil"
                        else:
                            raw_semesters = data.get("semesters", [])
                            for sem in raw_semesters:
                                if not isinstance(sem, dict): continue
                                raw_s = str(sem.get("semester") or sem.get("sem") or "").strip().upper()
                                raw_s = re.sub(r'SEMESTER\s*', '', raw_s).strip()
                                if not raw_s: continue

                                yr_val = str(sem.get("year") or sem.get("exam_year") or "").strip()
                                sgpa_val = str(sem.get("sgpa") or sem.get("gpa") or "").strip()
                                marks_val = str(sem.get("marks") or sem.get("marks_obtained") or "").strip()

                                if (not yr_val or yr_val in ["-", "N.A.", "None", ""]) and (not sgpa_val or sgpa_val in ["-", "N.A.", "None", ""]):
                                    continue

                                normalized_semesters.append({
                                    "semester": raw_s,
                                    "year": yr_val if yr_val not in ["-", "N.A."] else "-",
                                    "full_marks": str(sem.get("full_marks") or sem.get("total_marks") or "400").strip(),
                                    "marks": marks_val if marks_val not in ["-", "N.A."] else "-",
                                    "credit": str(sem.get("credit") or sem.get("semester_credit") or "20").strip(),
                                    "sgpa": sgpa_val if sgpa_val not in ["-", "N.A."] else "N.A."
                                })
                    except Exception as ai_err:
                        if "429" in str(ai_err) or "rate" in str(ai_err) or "quota" in str(ai_err):
                            ai_quota_exceeded = True

                # --- STRATEGY 2: LOCAL FALLBACK ---
                if not ai_client or ai_quota_exceeded or not reg_no:
                    full_text = page.get_text("text") or ""
                    reg_no, match_method = extract_reg_no_bulletproof(full_text)
                    if not reg_no:
                        bg_upload_status["processed_pages"] = page_num + 1
                        continue

                    roll_no = extract_roll_no_bulletproof(full_text)
                    name = extract_name_bulletproof(full_text)
                    course = selected_course if (selected_course and selected_course != "AUTO") else extract_course(full_text)
                    
                    subj_match = re.search(r'\b([A-Z]{3,4})\-[A-Z0-9]+\b', full_text)
                    subject = subj_match.group(1).upper() if subj_match else "BNGA"
                    
                    passout_year = extract_passout_year(full_text)

                    rect = page.rect
                    table_rect = fitz.Rect(0, rect.height * 0.48, rect.width, rect.height * 0.88)
                    table_text = extract_text_rows_from_rect(page, table_rect)

                    normalized_semesters, overall_cgpa, overall_grade, remarks = parse_summary_table_local(table_text)
                
                # --- STRICT ENFORCEMENT & BULLETPROOF FALLBACK ---
                if "fail" in overall_grade.lower() or "not cleared" in str(remarks).lower() or overall_grade in ["", "None", "N.A.", "Fail / Semester Not Cleared"]:
                    passout_year = "Nil"
                    overall_cgpa = "N.A."
                    overall_grade = "Fail / Semester Not Cleared"
                    normalized_semesters = []
                else:
                    passout_year = str(passout_year).strip()
                    if passout_year.lower() in ["nil", "unknown", "", "none", "null", "n.a."]:
                        max_yr = None
                        max_val = 0
                        for sem in normalized_semesters:
                            m = re.search(r'(20[1-3]\d)', str(sem["year"]))
                            if m:
                                val = int(m.group(1))
                                if sem["semester"] in ["VI", "6", "VI.", "6th"]:
                                    max_yr = str(val)
                                    break
                                if val > max_val:
                                    max_val = val
                                    max_yr = str(val)
                        if max_yr:
                            passout_year = max_yr
                        else:
                            passout_year = "Nil"

                # Save PDF copy
                pdf_repo_path = f"uploads/pdf_repository/{reg_no}.pdf"
                try:
                    new_pdf = fitz.open()
                    new_pdf.insert_pdf(doc, from_page=page_num, to_page=page_num)
                    new_pdf.save(pdf_repo_path)
                    new_pdf.close()
                except Exception:
                    pdf_repo_path = ""

                # Database Upsert
                existing_student = db.query(Student).filter(Student.registration_no == reg_no).first()
                if existing_student:
                    existing_student.name = name
                    existing_student.roll_no = roll_no
                    existing_student.course = course
                    existing_student.subject = subject
                    existing_student.passout_year = passout_year
                    existing_student.overall_cgpa = overall_cgpa
                    existing_student.overall_grade = overall_grade
                    existing_student.remarks = remarks
                    if pdf_repo_path: existing_student.pdf_document_path = pdf_repo_path
                    
                    db.query(SemesterRecord).filter(SemesterRecord.registration_no == reg_no).delete()
                    for sem in normalized_semesters:
                        db.add(SemesterRecord(
                            registration_no=reg_no, semester=sem["semester"], year=sem["year"], 
                            full_marks=sem["full_marks"], marks_obtained=sem["marks"], credit=sem["credit"], sgpa=sem["sgpa"]
                        ))
                else:
                    admission_year = "20" + reg_no.split("-")[-1] if "-" in reg_no else "Unknown"
                    student = Student(
                        registration_no=reg_no, roll_no=roll_no, name=name, admission_year=admission_year, 
                        passout_year=passout_year, course=course, subject=subject, overall_cgpa=overall_cgpa, 
                        overall_grade=overall_grade, remarks=remarks, pdf_document_path=pdf_repo_path
                    )
                    db.add(student)
                    for sem in normalized_semesters:
                        db.add(SemesterRecord(
                            registration_no=reg_no, semester=sem["semester"], year=sem["year"], 
                            full_marks=sem["full_marks"], marks_obtained=sem["marks"], credit=sem["credit"], sgpa=sem["sgpa"]
                        ))

                extracted_count += 1
                bg_upload_status["processed_pages"] = page_num + 1
                bg_upload_status["extracted_count"] = extracted_count
                bg_upload_status["status_message"] = f"Processing page {page_num+1} of {total_pages} ({extracted_count} records saved so far)..."

                if page_num % 5 == 0:
                    db.commit()
                    gc.collect()

            except Exception as page_err:
                db.rollback()
                bg_upload_status["processed_pages"] = page_num + 1
                continue

        if doc: doc.close()
        db.commit()
        
        bg_upload_status["is_processing"] = False
        bg_upload_status["status_message"] = f"🎉 ✅ EXTRACTION DONE! Successfully extracted {extracted_count} student record(s) from {total_pages} pages into the database."
    except Exception as e:
        db.rollback()
        bg_upload_status["is_processing"] = False
        bg_upload_status["status_message"] = f"❌ Error during background extraction: {str(e)}"
    finally:
        db.close()
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)

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
    try: yield db
    finally: db.close()

# --- SAFE STARTUP DB CREATION & MIGRATION ---
@app.on_event("startup")
def startup_db_setup():
    try:
        Base.metadata.create_all(bind=engine)
        with engine.begin() as conn:
            if "postgresql" in DATABASE_URL:
                conn.execute(text("ALTER TABLE students ADD COLUMN IF NOT EXISTS course VARCHAR DEFAULT 'Unknown Course';"))
                conn.execute(text("ALTER TABLE students ADD COLUMN IF NOT EXISTS subject VARCHAR DEFAULT 'BNGA';"))
                conn.execute(text("ALTER TABLE students ADD COLUMN IF NOT EXISTS passout_year VARCHAR DEFAULT 'Nil';"))
                conn.execute(text("ALTER TABLE students ADD COLUMN IF NOT EXISTS remarks VARCHAR DEFAULT 'Qualified';"))
                conn.execute(text("ALTER TABLE students ADD COLUMN IF NOT EXISTS marksheet_received BOOLEAN DEFAULT FALSE;"))
                conn.execute(text("ALTER TABLE students ADD COLUMN IF NOT EXISTS certificate_received BOOLEAN DEFAULT FALSE;"))
                conn.execute(text("ALTER TABLE students ADD COLUMN IF NOT EXISTS marksheet_issue_date VARCHAR DEFAULT '';"))
                conn.execute(text("ALTER TABLE students ADD COLUMN IF NOT EXISTS certificate_issue_date VARCHAR DEFAULT '';"))
                conn.execute(text("ALTER TABLE students ADD COLUMN IF NOT EXISTS issued_by VARCHAR DEFAULT '';"))
                conn.execute(text("ALTER TABLE students ADD COLUMN IF NOT EXISTS post_grad_status VARCHAR DEFAULT 'Unknown';"))
                conn.execute(text("ALTER TABLE students ADD COLUMN IF NOT EXISTS post_grad_details VARCHAR DEFAULT '';"))
                conn.execute(text("ALTER TABLE students ADD COLUMN IF NOT EXISTS proof_document_path VARCHAR DEFAULT '';"))
                conn.execute(text("ALTER TABLE students ADD COLUMN IF NOT EXISTS pdf_document_path VARCHAR DEFAULT '';"))
                conn.execute(text("ALTER TABLE semester_records ADD COLUMN IF NOT EXISTS full_marks VARCHAR DEFAULT '400';"))
                conn.execute(text("ALTER TABLE semester_records ADD COLUMN IF NOT EXISTS credit VARCHAR DEFAULT '20';"))
            elif "sqlite" in DATABASE_URL:
                columns = [row[1] for row in conn.execute(text("PRAGMA table_info(students);")).fetchall()]
                if columns:
                    if "course" not in columns: conn.execute(text("ALTER TABLE students ADD COLUMN course VARCHAR DEFAULT 'Unknown Course';"))
                    if "subject" not in columns: conn.execute(text("ALTER TABLE students ADD COLUMN subject VARCHAR DEFAULT 'BNGA';"))
                    if "passout_year" not in columns: conn.execute(text("ALTER TABLE students ADD COLUMN passout_year VARCHAR DEFAULT 'Nil';"))
                    if "remarks" not in columns: conn.execute(text("ALTER TABLE students ADD COLUMN remarks VARCHAR DEFAULT 'Qualified';"))
                    if "marksheet_received" not in columns: conn.execute(text("ALTER TABLE students ADD COLUMN marksheet_received BOOLEAN DEFAULT 0;"))
                    if "certificate_received" not in columns: conn.execute(text("ALTER TABLE students ADD COLUMN certificate_received BOOLEAN DEFAULT 0;"))
                    if "marksheet_issue_date" not in columns: conn.execute(text("ALTER TABLE students ADD COLUMN marksheet_issue_date VARCHAR;"))
                    if "certificate_issue_date" not in columns: conn.execute(text("ALTER TABLE students ADD COLUMN certificate_issue_date VARCHAR;"))
                    if "issued_by" not in columns: conn.execute(text("ALTER TABLE students ADD COLUMN issued_by VARCHAR;"))
                    if "post_grad_status" not in columns: conn.execute(text("ALTER TABLE students ADD COLUMN post_grad_status VARCHAR DEFAULT 'Unknown';"))
                    if "post_grad_details" not in columns: conn.execute(text("ALTER TABLE students ADD COLUMN post_grad_details VARCHAR;"))
                    if "proof_document_path" not in columns: conn.execute(text("ALTER TABLE students ADD COLUMN proof_document_path VARCHAR;"))
                    if "pdf_document_path" not in columns: conn.execute(text("ALTER TABLE students ADD COLUMN pdf_document_path VARCHAR;"))
                
                sem_columns = [row[1] for row in conn.execute(text("PRAGMA table_info(semester_records);")).fetchall()]
                if sem_columns:
                    if "full_marks" not in sem_columns: conn.execute(text("ALTER TABLE semester_records ADD COLUMN full_marks VARCHAR DEFAULT '400';"))
                    if "credit" not in sem_columns: conn.execute(text("ALTER TABLE semester_records ADD COLUMN credit VARCHAR DEFAULT '20';"))
    except Exception as e:
        print(f"Startup Migration Note: {e}")

# --- FRONTEND ROUTE ---
@app.get("/")
def serve_frontend(user: dict = Depends(get_current_user)):
    return FileResponse("index.html")

# --- AUTH INFO ENDPOINT ---
@app.get("/api/auth/me")
def get_auth_me(user: dict = Depends(get_current_user)):
    return {"username": user["username"], "role": user["role"]}

# --- API ENDPOINTS ---

@app.get("/api/admin/upload-status")
def get_upload_status(user: dict = Depends(get_current_user)):
    return bg_upload_status

@app.post("/api/admin/upload-marksheet")
async def upload_marksheet(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...), 
    selected_course: str = Form("AUTO"),
    user: dict = Depends(require_admin)
):
    temp_pdf_path = f"temp_{secrets.token_hex(4)}_{file.filename}"
    with open(temp_pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    doc = fitz.open(temp_pdf_path) if fitz else []
    total_pages = len(doc)
    doc.close()

    background_tasks.add_task(process_large_pdf_in_background, temp_pdf_path, selected_course)

    return {
        "message": f"🚀 Successfully started background processing for {total_pages} page(s)! Refresh Tab 6 (Student Directory) or Tab 7 (PDF Repository) to see extracted records in real-time."
    }

@app.get("/api/student/{reg_no}")
def get_student(reg_no: str, db: Session = Depends(get_db), user: dict = Depends(get_current_user)):
    student = db.query(Student).options(joinedload(Student.semesters)).filter(Student.registration_no == reg_no).first()
    if not student: 
        raise HTTPException(status_code=404, detail="Student record not found.")
    
    if auto_repair_passout_year(student):
        db.commit()

    return {
        "student": {
            "name": student.name, 
            "reg_no": student.registration_no, 
            "roll_no": student.roll_no,
            "admission_year": student.admission_year, 
            "passout_year": student.passout_year, 
            "course": student.course,
            "subject": student.subject or "BNGA",
            "cgpa": student.overall_cgpa, 
            "grade": student.overall_grade,
            "remarks": student.remarks or "Qualified",
            "marksheet_received": student.marksheet_received, 
            "certificate_received": student.certificate_received,
            "marksheet_issue_date": student.marksheet_issue_date or "",
            "certificate_issue_date": student.certificate_issue_date or "",
            "issued_by": student.issued_by or "",
            "status": student.post_grad_status or "Unknown", 
            "details": student.post_grad_details, 
            "proof": student.proof_document_path,
            "pdf_path": student.pdf_document_path
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
            for s in student.semesters
        ]
    }

@app.post("/api/admin/update-profile-full/{reg_no}")
async def update_student_profile_full(
    reg_no: str,
    payload: dict,
    db: Session = Depends(get_db),
    user: dict = Depends(get_current_user)
):
    student = db.query(Student).filter(Student.registration_no == reg_no).first()
    if not student: 
        raise HTTPException(status_code=404, detail="Student not found")

    student.name = payload.get("name", student.name)
    student.roll_no = payload.get("roll_no", student.roll_no)
    student.course = payload.get("course", student.course)
    student.subject = payload.get("subject", student.subject)
    student.passout_year = payload.get("passout_year", student.passout_year)
    student.overall_cgpa = payload.get("cgpa", student.overall_cgpa)
    student.overall_grade = payload.get("grade", student.overall_grade)
    student.remarks = payload.get("remarks", student.remarks)

    if "fail" in student.overall_grade.lower() or "not cleared" in student.remarks.lower() or student.overall_grade in ["", "None", "N.A."]:
        student.passout_year = "Nil"
        student.overall_cgpa = "N.A."
        student.overall_grade = "Fail / Semester Not Cleared"
        payload["semesters"] = []

    new_semesters = payload.get("semesters", [])
    db.query(SemesterRecord).filter(SemesterRecord.registration_no == reg_no).delete()
    for sem in new_semesters:
        db.add(SemesterRecord(
            registration_no=reg_no,
            semester=sem.get("semester"),
            year=sem.get("year"),
            full_marks=sem.get("full_marks", "400"),
            marks_obtained=sem.get("marks"),
            credit=sem.get("credit", "20"),
            sgpa=sem.get("sgpa")
        ))

    db.commit()
    return {"message": "Student profile and marks updated successfully!"}

@app.post("/api/admin/update-issuance-detailed/{reg_no}")
async def update_issuance_detailed(
    reg_no: str,
    marksheet_received: bool = Form(...),
    certificate_received: bool = Form(...),
    marksheet_issue_date: str = Form(""),
    certificate_issue_date: str = Form(""),
    issued_by: str = Form(""),
    db: Session = Depends(get_db),
    user: dict = Depends(get_current_user)
):
    student = db.query(Student).filter(Student.registration_no == reg_no).first()
    if not student: 
        raise HTTPException(status_code=404, detail="Student not found")

    student.marksheet_received = marksheet_received
    student.certificate_received = certificate_received
    student.marksheet_issue_date = marksheet_issue_date
    student.certificate_issue_date = certificate_issue_date
    student.issued_by = issued_by

    db.commit()
    return {"message": "Issuance details updated successfully!"}

@app.delete("/api/admin/student/{reg_no}")
def delete_student(reg_no: str, db: Session = Depends(get_db), user: dict = Depends(require_admin)):
    student = db.query(Student).filter(Student.registration_no == reg_no).first()
    if not student: 
        raise HTTPException(status_code=404, detail="Student not found")
    
    if student.pdf_document_path and os.path.exists(student.pdf_document_path):
        try: os.remove(student.pdf_document_path)
        except: pass

    db.query(SemesterRecord).filter(SemesterRecord.registration_no == reg_no).delete()
    db.delete(student)
    db.commit()
    return {"message": f"Student {reg_no} deleted successfully"}

@app.post("/api/admin/clear-all-students")
def clear_all_students(db: Session = Depends(get_db), user: dict = Depends(require_admin)):
    db.query(SemesterRecord).delete()
    db.query(Student).delete()
    db.commit()
    
    try:
        shutil.rmtree("uploads/pdf_repository")
        os.makedirs("uploads/pdf_repository", exist_ok=True)
    except: pass

    return {"message": "All student records and stored PDF marksheets cleared successfully"}

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
    user: dict = Depends(get_current_user)
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

@app.get("/api/admin/all-students")
def get_all_students(db: Session = Depends(get_db), user: dict = Depends(get_current_user)):
    students = db.query(Student).options(joinedload(Student.semesters)).all()
    needs_commit = False
    
    for s in students:
        if auto_repair_passout_year(s):
            needs_commit = True
            
    if needs_commit:
        db.commit()
        
    return students

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
