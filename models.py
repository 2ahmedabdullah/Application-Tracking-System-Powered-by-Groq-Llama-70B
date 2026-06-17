# models.py

from typing import List, Optional, Any, Annotated, Literal
from pydantic import BaseModel, Field, BeforeValidator
import re

# ---------------------------------------------------------
# DEFENSIVE LLM SANITIZATION UTILITIES
# ---------------------------------------------------------
def sanitize_skill_string(v: Any) -> str:
    if not v or not isinstance(v, str):
        return ""
    text = v.strip()
    match = re.search(r'\((.*?)\)', text)
    if match:
        clean_text = match.group(1)
    else:
        clean_text = text

    clean_text = re.sub(r'^(ml frameworks|modern data platforms|frameworks|tools|platforms|languages)\s+for\s+|^layers\s+of\s+', '', clean_text, flags=re.IGNORECASE)
    if ',' in clean_text:
        return clean_text.split(',')[0].strip()
    return clean_text.strip()

def sanitize_date_string(v: Any) -> str:
    if not v or not isinstance(v, str):
        return "1970-01"
        
    text = v.strip().lower()
    
    # 1. Handle Active/Current Roles Boundlessly
    if text in ["present", "current", "now", "ongoing", "till date"]:
        return "2026-06"  # Locked to current 2026 runtime pipeline context
        
    # 2. Match Standard ISO or Slash Formats (e.g., 2024-08 or 08/2024)
    iso_match = re.search(r'(\d{4})[-/](\d{1,2})', text)
    if iso_match:
        return f"{iso_match.group(1)}-{int(iso_match.group(2)):02d}"
        
    us_match = re.search(r'(\d{1,2})[-/](\d{4})', text)
    if us_match:
        return f"{us_match.group(2)}-{int(us_match.group(1)):02d}"
        
    # 3. Handle English Text Months (e.g., "August 2024", "Aug 2024")
    months_map = {
        "jan": "01", "feb": "02", "mar": "03", "apr": "04", "may": "05", "jun": "06",
        "jul": "07", "aug": "08", "sep": "09", "oct": "10", "nov": "11", "dec": "12",
        "january": "01", "february": "02", "march": "03", "april": "04", "june": "06",
        "july": "07", "august": "08", "september": "09", "october": "10", "november": "11", "december": "12"
    }
    
    # Extract the year
    year_match = re.search(r'\b(\d{4})\b', text)
    if year_match:
        year = year_match.group(1)
        # Look for a named month token
        for month_name, month_num in months_map.items():
            if month_name in text:
                return f"{year}-{month_num}"
        # Fallback to January if a year exists but no month name can be resolved
        return f"{year}-01"
        
    return "1970-01"

def sanitize_seniority_tier(v: Any) -> str:
    text = str(v).strip().lower()
    if "junior" in text or "jr" in text: return "Junior_IC"
    if "lead" in text or "principal" in text: return "Lead_IC"
    if "senior" in text or "sr" in text: return "Senior_IC"
    if "manager" in text or "head" in text: return "Manager"
    if "exec" in text or "vp" in text or "director" in text: return "Executive"
    return "Mid_IC"

SafeDate = Annotated[str, BeforeValidator(sanitize_date_string)]
SafeTier = Annotated[str, BeforeValidator(sanitize_seniority_tier)]
SafeSkill = Annotated[str, BeforeValidator(sanitize_skill_string)]

# ---------------------------------------------------------
# STRUCTURED DATA LAYERS
# ---------------------------------------------------------
class SegmentedResume(BaseModel):
    candidate_name: str = Field(description="Full name of the candidate")
    global_skills_section: str = Field(description="The unparsed raw text block containing the skills section")
    employment_blocks: List[str] = Field(description="A list of raw text segments per work experience entry.")

class WorkExperienceMetadata(BaseModel):
    company: str = Field(description="Name of the company or organization")
    job_title: str = Field(description="The formal job title held by the candidate")
    start_date: str = Field(description="The raw start date text exactly as written in the resume (e.g., 'August 2024' or '08/2024')")
    end_date: Optional[str] = Field(None, description="The raw end date text exactly as written in the resume (e.g., 'July 2026', 'Present', or 'Current')")
    inferred_seniority: SafeTier = Field(description="Normalized tier mapping")
    skills_extracted: List[str] = Field(default_factory=list)
    

class WorkExperience(BaseModel):
    company: str = Field(description="Name of the company or organization")
    job_title: str = Field(description="The formal job title held by the candidate")
    start_date: SafeDate = Field(description="Start date sanitized to YYYY-MM")
    end_date: Optional[SafeDate] = Field(None, description="End date sanitized to YYYY-MM or None")
    inferred_seniority: SafeTier = Field(description="Normalized tier mapping")
    skills_used: List[str] = Field(description="List of deterministic matched technical terms")

class EducationElement(BaseModel):
    institution: str = Field(description="Name of the university or school")
    degree: str = Field(description="Degree attained")
    field_of_study: str = Field(description="Major or concentration")
    graduation_date: SafeDate = Field(description="Graduation date sanitized to YYYY-MM")

class ParsedResume(BaseModel):
    candidate_name: str = Field(description="Full name of the candidate")
    global_skills: List[SafeSkill] = Field(description="The master list of skills extracted")
    experience_history: List[WorkExperience] = Field(description="Chronological history of employment")
    education_history: List[EducationElement] = Field(description="Academic history")
    raw_achievements: List[str] = Field(description="3-5 extracted impact/metric bullet points")

class RemediationElement(BaseModel):
    missing_skill: str = Field(description="The specific technical skill or tool missing from the original CV.")
    target_company: str = Field(description="The company name from the candidate's work history where this skill contextually fits.")
    rationale: str = Field(description="Engineering-focused reasoning for why this skill naturally integrates into this specific company's stack.")
    precise_injection_bullet: str = Field(description="The high-impact, action-verb-led resume bullet point containing the injected skill.")
    type: Literal["New", "Paraphrased"] = Field(description="Indicates whether this bullet point is brand new or a modification of an existing one.")

class CandidateRemediationPlan(BaseModel):
    remediation_plan: List[RemediationElement] = Field(description="The collection of chronological skill remediation updates")
    raw_cv_text: str = Field(description="The raw, unformatted CV text that was used for the workspace context.")
    raw_jd_text: str = Field(description="The raw, unformatted JD text that was used for the workspace context.")
