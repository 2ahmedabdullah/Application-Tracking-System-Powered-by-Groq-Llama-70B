#utils.py

import os
import json
import re
import ollama
import numpy as np
from pypdf import PdfReader
from docx import Document
from models import ParsedResume, SegmentedResume, WorkExperienceMetadata, WorkExperience, sanitize_date_string, CandidateRemediationPlan
from datetime import datetime
from pydantic import ValidationError
from typing import List
from groq import Groq
from dotenv import load_dotenv

# --- INITIALIZATION ---
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

OLLAMA_CLIENT = ollama.Client(host=config.get("ollama_base_url", "http://localhost:11434"))
EXTRACTION_MODEL = config.get("extraction_model", "llama3.1:70b")
EMBEDDING_MODEL = config.get("embedding_model", "nomic-embed-text") # Fallback configuration safeguard

# --- SIMILARITY MATH ---
def calculate_cosine_similarity(v1, v2):
    dot_prod = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return float(dot_prod / (norm_v1 * norm_v2)) if norm_v1 and norm_v2 else 0.0



def normalize_date_string(date_str: str) -> str:
    """Converts 'August 2024' or 'July 2026' to 'YYYY-MM' deterministically."""
    cleaned = date_str.strip().lower()
    if cleaned in ["present", "current", "now"]:
        return "2026-06"  # Locked to current pipeline date
        
    # Replace common separators
    cleaned = re.sub(r'[^a-z0-9\s]', ' ', cleaned)
    
    try:
        # Try parsing standard formats like "August 2024" or "Aug 2024"
        for fmt in ("%B %Y", "%b %Y", "%m %Y", "%Y"):
            try:
                dt = datetime.strptime(cleaned, fmt)
                return dt.strftime("%Y-%m")
            except ValueError:
                continue
        return "2026-01" # Safe baseline fallback
    except Exception:
        return "2026-01"
        
# --- FILE EXTRACTION INGESTION LAYERS ---
def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text: 
            full_text += text + "\n"
    return full_text

def extract_text_from_docx(docx_path: str) -> str:
    doc = Document(docx_path)
    full_text = []
    for paragraph in doc.paragraphs:
        if paragraph.text.strip(): 
            full_text.append(paragraph.text)
    return "\n".join(full_text)

def extract_raw_text(file_path: str) -> str:
    _, ext = os.path.splitext(file_path.lower())
    if ext == ".pdf": 
        return extract_text_from_pdf(file_path)
    elif ext == ".docx": 
        return extract_text_from_docx(file_path)
    else: 
        raise ValueError(f"Unsupported file format: '{ext}'")

# --- PRODUCTION-GRADE DETERMINISTIC EXTRACTION ENGINE ---
def extract_skills_from_block(text_block: str, technical_dictionary: set) -> list:
    """
    Scans text blocks using regex boundaries. Fully generic, protecting 
    special tech characters (/, -, +, #) and resolving multi-word phrases.
    """
    if not text_block:
        return []
        
    found_skills = set()
    # Lowercase text but do NOT completely destroy internal hyphenation/slashes yet
    normalized_block = text_block.lower()
    
    # Clean up outer line noise while keeping critical technical punctuation characters intact
    normalized_block = re.sub(r'[()\[\]{}.,;:*•"\'\?]', ' ', normalized_block)
    # Standardize whitespace formatting
    normalized_block = " ".join(normalized_block.split())

    for skill in technical_dictionary:
        skill_lower = skill.lower().strip()
        if not skill_lower:
            continue
            
        # Standardize the skill text variants (e.g. handling variations in spaces/ampersands)
        escaped_skill = re.escape(skill_lower)
        
        # Word boundary pattern matching: handles standalone text, start/end of lines, or space clusters
        # Natively handles complex tags like 'C++', 'CI/CD', 'Scikit-learn', and 'SQL'
        pattern = rf'(?:^|\s|\b){escaped_skill}(?:\s|\b|$)'
        
        if re.search(pattern, normalized_block):
            found_skills.add(skill)
            
    return sorted(list(found_skills))

# --- LAYOUT-AGNOSTIC INGESTION PIPELINE ---
def parse_resume_to_json_generic(file_path: str, dynamic_jd_skills: list) -> ParsedResume:
    """
    A layout-agnostic, zero-hardcoded resume ingestion parser.
    Utilizes Groq Cloud (llama-3.3-70b-versatile) for sub-second parsing.
    """
    # 1. Initialize and verify the Groq client
    load_dotenv()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        raise ValueError("❌ Missing Environment Variable: 'GROQ_API_KEY' could not be found.")
    
    groq_client = Groq(api_key=GROQ_API_KEY)
    
    # 2. Extract and sanitize raw incoming document strings
    raw_resume_text = extract_raw_text(file_path)
    if not raw_resume_text.strip():
        raise ValueError(f"Extraction failed or file is empty: {file_path}")
        
    runtime_taxonomy = {str(skill).strip() for skill in dynamic_jd_skills if skill}

    # 3. Streamlined One-Pass Comprehensive Parsing Prompt
    parsing_prompt = """
        You are an advanced enterprise talent intelligence parsing system. 
        Your task is to analyze the provided resume text and map it into a strict, structured JSON object format.
        
        CRITICAL PROCESSING INSTRUCTIONS:
        1. Extract the candidate's full legal name into 'candidate_name'.
        2. Identify all technical tools, frameworks, programming languages, and competencies explicitly found in the text into 'global_skills'.
        3. Break down the employment history into chronological 'experience_history' blocks.
        4. For every individual work block, extract:
           - 'company': Name of the organization.
           - 'job_title': Professional designation.
           - 'start_date' / 'end_date': Format explicitly as 'YYYY-MM'. Use null or 'Present' if ongoing.
           - 'inferred_seniority': Tier mapping (e.g., Intern, Junior_IC, Mid_IC, Senior_IC, Lead_IC, Manager, Director).
           - 'skills_used': Isolate a dedicated list of all technical tools, databases, and ML methodologies used *specifically* within that role's boundaries.

        OUTPUT FORMAT:
        You must return a valid JSON object matching this exact structural schema setup:
        {
            "candidate_name": "string",
            "global_skills": ["string"],
            "experience_history": [
                {
                    "company": "string",
                    "job_title": "string",
                    "start_date": "string",
                    "end_date": "string",
                    "inferred_seniority": "string",
                    "skills_used": ["string"]
                }
            ]
        }
    """

    print("⚡ [Groq Engine] Executing complete layout extraction and metadata serialization...")
    
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": parsing_prompt},
                {"role": "user", "content": f"Analyze and partition this candidate resume text:\n\n{raw_resume_text}"}
            ],
            temperature=0.0,
            max_tokens=3000,
            response_format={"type": "json_object"}
        )
        
        raw_content = response.choices[0].message.content.strip()
        payload_dict = json.loads(raw_content)
        
        # 4. Perform deterministic verification fallbacks 
        # Ensures no key skills were lost during structural assignment
        validated_global_skills = extract_skills_from_block(raw_resume_text, runtime_taxonomy)
        if not payload_dict.get("global_skills"):
            payload_dict["global_skills"] = validated_global_skills
            
        # Ensure structural placeholders are present to satisfy Pydantic compliance bounds
        payload_dict["education_history"] = []
        payload_dict["raw_achievements"] = []
        
        # 5. Sanitize date strings inline across the returned structural layers
        for job in payload_dict.get("experience_history", []):
            job["start_date"] = sanitize_date_string(job.get("start_date", ""))
            job["end_date"] = sanitize_date_string(job.get("end_date", ""))

        # Validate unified dictionary structure against your baseline Pydantic Object layout
        return ParsedResume.model_validate(payload_dict)

    except Exception as e:
        print(f"[CRITICAL FAILURE] Cloud parsing execution layer failed: {str(e)}")
        raise e



# --- VECTOR SEMANTIC GAP ANALYSIS ---
def identify_missing_skills(candidate_skills, required_skills, client, model_name, threshold=0.70):
    if not required_skills: 
        return []
    if not candidate_skills: 
        return required_skills
        
    cand_vectors = np.array([client.embeddings(model=model_name, prompt=s)['embedding'] for s in candidate_skills])
    req_vectors = np.array([client.embeddings(model=model_name, prompt=s)['embedding'] for s in required_skills])
    
    cand_norms = np.linalg.norm(cand_vectors, axis=1, keepdims=True)
    req_norms = np.linalg.norm(req_vectors, axis=1, keepdims=True)
    
    # Avoid zero-division exceptions smoothly
    cand_norms[cand_norms == 0] = 1.0
    req_norms[req_norms == 0] = 1.0
    
    cand_vectors = cand_vectors / cand_norms
    req_vectors = req_vectors / req_norms
    
    similarity_matrix = np.dot(req_vectors, cand_vectors.T)
    best_matches = np.max(similarity_matrix, axis=1)
    
    return [required_skills[i] for i, score in enumerate(best_matches) if score < threshold]



def generate_remediation_plan(raw_jd: str, employment_history: List, missing_skills: List[str]) -> CandidateRemediationPlan:
    """
    Stage 2: Strategic Additions of missing JD requirements utilizing Groq Cloud API
    for sub-second structured inference execution.
    """  
    # 1. Initialize and verify the Groq client
    # Explicitly load the local .env file variables into system environment memory
    load_dotenv() 

    # Now os.getenv will successfully pull your key right out of your .env file
    GROQ_API_KEY = os.getenv("GROQ_API_KEY") 
    if not GROQ_API_KEY:
        raise ValueError(
            "❌ Missing Environment Variable: 'GROQ_API_KEY' could not be found. "
            "Please ensure your .env file exists and contains: GROQ_API_KEY=your_key"
        )
        
    groq_client = Groq(api_key=GROQ_API_KEY)

    # 2. Safely serialize custom Pydantic models/dicts into a text block
    serializable_history = []
    for job in employment_history:
        if hasattr(job, "model_dump"):
            serializable_history.append(job.model_dump())
        elif hasattr(job, "dict"):
            serializable_history.append(job.dict())
        else:
            serializable_history.append(job)
            
    cv_context_string = json.dumps(serializable_history, indent=2)
    runtime_taxonomy = [str(skill).strip() for skill in missing_skills if skill]
    
    # 3. Construct System Prompt (Updated to pass Groq's pre-flight JSON string check)
    remediation_prompt = """
        You are an expert resume optimizer and technical career strategist.
        Your task is to strategically inject missing technical skills into the candidate's CV either as a brand new bullet point or by paraphrasing an existing one.
        You must map these gaps naturally to avoid detection by string indexers.

        CRITICAL RULES:
        1. Identify which of the provided 'Missing Skills' match the domain context of each company work block.
        2. For every injected skill, write 2 comprehensive, high-impact bullet points starting with an action verb.
        3. Blend the missing skill seamlessly into the existing contextual responsibilities of that specific company.
        4. Provide a clear, engineering-focused rationale explaining why the skill fits naturally into that specific stack.
        5. The field 'type' must strictly be either "New" or "Paraphrased" based on your modifications.
        6. Rationale definition -> Not why this skill is important, But why this skill was alloted to this target_company.
        7. Rationale -> Do Not include definitions or advantage of a skill in rationale as a reasoning.
              
        OUTPUT FORMAT:
        You must return a valid JSON object matching this exact structural schema configuration:
        {
            "remediation_plan": [
                {
                    "missing_skill": "string",
                    "target_company": "string",
                    "rationale": "string",
                    "precise_injection_bullet": "string",
                    "type": "New"
                }
            ]
        }
    """
    
    user_payload = (
        f"Target Missing Skills to Incorporate:\n{runtime_taxonomy}\n\n"
        f"Raw Candidate CV Workspace Context:\n{cv_context_string}\n\n"
        f"Raw Job Description Context:\n{raw_jd}"
    )

    print("⚡ Routing payload to Groq Cloud Architecture (llama-3.3-70b-versatile)...")
    
    try:
        # 4. Request structured output directly from Groq using json_object mode
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": remediation_prompt},
                {"role": "user", "content": user_payload}
            ],
            temperature=0.1,
            max_tokens=2000,
            # Groq natively supports JSON mode to guarantee a structurally clean response
            response_format={"type": "json_object"} 
        )
        
        raw_content = response.choices[0].message.content.strip()

        # 5. Intercept JSON dictionary, backfill metadata, and validate via Pydantic
        payload_dict = json.loads(raw_content)
        payload_dict["raw_cv_text"] = cv_context_string
        payload_dict["raw_jd_text"] = raw_jd
        
        remediation_plan = CandidateRemediationPlan.model_validate(payload_dict)
        return remediation_plan
        
    except json.JSONDecodeError as jde:
        print(f"Critical error: Groq did not return valid JSON formatting: {str(jde)}")
        raise ValueError("Failed to parse response payload as valid JSON structure.") from jde
    except ValidationError as ve:
        print(f"Schema verification failed for compiled validation payload: {ve}")
        raise ValueError("Groq's payload broke formatting compliance with CandidateRemediationPlan.") from ve


# def generate_remediation_plan(raw_jd: str, employment_history: dict, missing_skills: List[str]) -> CandidateRemediationPlan:
#     """
#     Stage 2: Strategic Additions of missing JD requirements, 
#     constructs the gap mitigation prompt, and returns a structured remediation schema.
#     """  
#     # Clean and isolate the target skills to inject safely
#     runtime_taxonomy = [str(skill).strip() for skill in missing_skills if skill]
#     serializable_history = []
#     for job in employment_history:
#         if hasattr(job, "model_dump"):  # Pydantic v2
#             serializable_history.append(job.model_dump())
#         elif hasattr(job, "dict"):      # Pydantic v1 fallback
#             serializable_history.append(job.dict())
#         else:
#             serializable_history.append(job) # Raw dict fallback
            
#     cv_context_string = json.dumps(serializable_history, indent=2)
    
#     # Clean, literal system instructions for precise injection (No messy nested quotes)
#     remediation_prompt = """
#         You are an expert resume optimizer and technical career strategist.
#         Your task is to strategically inject missing technical skills into the candidate's CV either as a brand new bullet point or by paraphrasing an existing one.
#         You must map these gaps naturally to avoid detection by string indexers.

#         CRITICAL RULES:
#         1. Identify which of the provided 'Missing Skills' match the domain context of each company work block.
#         2. For every injected skill, write 2 comprehensive, high-impact bullet points starting with an action verb.
#         3. Blend the missing skill seamlessly into the existing contextual responsibilities of that specific company.
#         4. Provide a clear, engineering-focused rationale explaining why the skill fits naturally into that specific stack.
#         5. The field 'type' must strictly be either "New" or "Paraphrased" based on your modifications.
#         6. Rationale definition -> Not why this skill is important, But why this skill was alloted to this target_company.
#         7. Rationale -> Do Not include definitions or advantage of a skill in rationale as a reasoning.
              
#         OUTPUT FORMAT:
#         {
#             "remediation_plan": [
#                 {
#                     "missing_skill": "string",
#                     "target_company": "string",
#                     "rationale": "string",
#                     "precise_injection_bullet": "string",
#                     "type": "New"
#                 }
#             ]
#         }
#     """
    
#     # Construct the user payload using correct function parameters
#     user_payload = (
#         f"Target Missing Skills to Incorporate:\n{runtime_taxonomy}\n\n"
#         f"Raw Candidate CV Workspace Context:\n{cv_context_string}\n\n"
#         f"Raw Job Description Context:\n{raw_jd}"
#     )

#     print(f"Generating precise remediation plan using {EXTRACTION_MODEL}...")
    
#     # Execute structured output chat call via Ollama
#     remediation_response = OLLAMA_CLIENT.chat(
#         model=EXTRACTION_MODEL,
#         messages=[
#             {"role": "system", "content": remediation_prompt},
#             {"role": "user", "content": user_payload}
#         ],
#         # Force the output structural framework via Pydantic model JSON schema
#         format=CandidateRemediationPlan.model_json_schema(),
#         options={"temperature": 0.1, "seed": 42}
#     )
    
#     # Extract, clean, and defend raw content bounds
#     raw_content = remediation_response['message']['content'].strip()

#     # --- COMPLETION: Intercept JSON, backfill source strings, and validate ---
#     try:
#         # 1. Parse LLM response to a dictionary first
#         payload_dict = json.loads(raw_content)
        
#         # 2. Backfill the required top-level context fields directly
#         payload_dict["raw_cv_text"] = cv_context_string
#         payload_dict["raw_jd_text"] = raw_jd
        
#         # 3. Hydrate and validate the complete data payload object via Pydantic v2
#         remediation_plan = CandidateRemediationPlan.model_validate(payload_dict)
#         return remediation_plan
        
#     except json.JSONDecodeError as jde:
#         print(f"Critical error: LLM did not return standard JSON formatting. Raw text: {raw_content[:200]}")
#         raise ValueError("Failed to parse response payload as valid JSON structure.") from jde
#     except ValidationError as ve:
#         print(f"Schema verification failed for compiled validation payload: {ve}")
#         raise ValueError("Ollama's response broke formatting compliance with CandidateRemediationPlan.") from ve
    
