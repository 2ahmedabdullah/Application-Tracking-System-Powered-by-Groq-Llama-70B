#scoring_engine.py

import numpy as np
import math
from datetime import datetime
from typing import Dict, Any, List
from scoring_models import JobRequirements
import os, json, ollama
# ---------------------------------------------------------
# ADVANCED MATHEMATICAL UTILITIES & MATRIX OPERATORS
# ---------------------------------------------------------

# Load your config file
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

OLLAMA_CLIENT = ollama.Client(host=config.get("ollama_base_url", "http://localhost:11434"))
EXTRACTION_MODEL = config.get("extraction_model", "llama3.1:70b")


def calculate_adaptive_decay(years_since_end: float, skill_name: str) -> float:
    """
    Applies an exponential half-life decay function depending on 
    the systemic volatility of the skill segment.
    """
    name_clean = skill_name.lower().strip()
    
    # Categorization map for architectural stability
    volatile_stacks = ["langchain", "langgraph", "llama", "nextjs", "tailwindcss", "crewai"]
    framework_stacks = ["react", "vue", "fastapi", "springboot", "django", "nodejs", "express"]
    
    if any(tech in name_clean for tech in volatile_stacks):
        half_life_years = 1.5   # Volatile modern tech stacks decay rapidly
    elif any(tech in name_clean for tech in framework_stacks):
        half_life_years = 3.5   # Frameworks evolve every few years
    else:
        half_life_years = 10.0  # Foundational skills (Python, SQL, Linux, C++) are durable

    return math.pow(0.5, (years_since_end / half_life_years))

# ---------------------------------------------------------
# THE REFACTORED CORE VECTOR SCORING LOGICS
# ---------------------------------------------------------
def score_skills_vector(candidate_skills: List[str], required_skills: List[str], req_vectors: np.ndarray, client, model_name: str, threshold: float = 0.72) -> float:
    """
    Computes true mathematical matrix alignment checking each required 
    skill element individually against the candidate's vector field space.
    """
    if not required_skills:
        return 100.0
    if not candidate_skills or req_vectors.size == 0:
        return 0.0

    # 1. Batch extract embeddings ONLY for the candidate (Required skills are already passed in)
    cand_vectors = np.array([client.embeddings(model=model_name, prompt=s)['embedding'] for s in candidate_skills])

    # 2. Vector Norm calculation for strict unit vector transformations
    req_norms = np.linalg.norm(req_vectors, axis=1, keepdims=True)
    cand_norms = np.linalg.norm(cand_vectors, axis=1, keepdims=True)
    
    req_norms[req_norms == 0] = 1.0
    cand_norms[cand_norms == 0] = 1.0

    req_vectors = req_vectors / req_norms
    cand_vectors = cand_vectors / cand_norms

    # 3. Calculate similarity matrix via dot product: Shape [len(required), len(candidate)]
    similarity_matrix = np.dot(req_vectors, cand_vectors.T)

    # 4. Filter for maximum alignment score per required skill row
    best_matches = np.max(similarity_matrix, axis=1)

    # 5. Measure coverage ratio meeting or exceeding semantic constraint limit
    matched_skills_count = np.sum(best_matches >= threshold)
    return float((matched_skills_count / len(required_skills)) * 100)


def score_experience_vector(experience_history: List[dict], required_skills: List[str], target_months: int) -> float:
    """
    REAL ENTERPRISE ATS SIMULATION:
    1. Flattens job text into a single string to prevent array element mismatches.
    2. Measures total chronological career alignment without double-penalizing missing skills.
    """
    if target_months == 0:
        return 100.0
        
    current_date = datetime.now()
    total_aligned_months = 0.0

    for job in experience_history:
        # 1. Calculate raw duration
        try:
            start = datetime.strptime(job['start_date'], "%Y-%m")
            if job.get('end_date') and job.get('end_date') != "1970-01":
                parsed_end = datetime.strptime(job['end_date'], "%Y-%m")
                end = min(parsed_end, current_date)
            else:
                end = current_date
        except Exception:
            continue 
                
        duration_months = (end.year - start.year) * 12 + (end.month - start.month)
        if duration_months <= 0:
            duration_months = 1

        # 2. CONVERT LIST TO RAW TEXT: Fixes the element-matching bug
        # Merges all skills into a single lowercase text block for broad substring searching
        job_text_block = " ".join([str(s).lower().strip() for s in job.get('skills_used', [])])
        job_title_lower = job.get('job_title', '').lower()
        
        # 3. VERIFY ROLE RELEVANCE: Did this job match the target field?
        # If the candidate has at least ONE core skill keyword OR a relevant title token,
        # enterprise systems credit the entire duration of that job to their career clock.
        has_skill_overlap = any(skill.lower() in job_text_block for skill in required_skills)
        is_relevant_title = "data" in job_title_lower or "engineer" in job_title_lower

        if has_skill_overlap or is_relevant_title:
            total_aligned_months += duration_months

    # 4. Final aggregate score represents total relevant career runway
    experience_ratio = total_aligned_months / target_months
    return min(100.0, round(experience_ratio * 100, 2))



def score_seniority_vector(experience_history: List[dict], target_seniority: str) -> float:
    """Maps historical hierarchy seniority rankings strictly against target requirements."""
    tier_weights = {"Junior_IC": 1, "Mid_IC": 2, "Senior_IC": 3, "Lead_IC": 4, "Manager": 5, "Executive": 6}
    
    highest_tier_found = 0
    target_weight = tier_weights.get(target_seniority, 3)
    
    for job in experience_history:
        job_tier = job.get('inferred_seniority', 'Mid_IC')
        highest_tier_found = max(highest_tier_found, tier_weights.get(job_tier, 2))
        
    if highest_tier_found >= target_weight:
        return 100.0  
    elif highest_tier_found == (target_weight - 1):
        return 70.0   # One rank lower tier penalty
    return 30.0


def score_company_bonus_vector(experience_history: List[dict], target_companies: List[str], client, model_name: str) -> float:
    """Runs a semantic vector proximity check to prevent string mismatches on company variants."""
    if not target_companies:
        return 100.0
        
    target_vectors = np.array([client.embeddings(model=model_name, prompt=c)['embedding'] for c in target_companies])
    target_norms = np.linalg.norm(target_vectors, axis=1, keepdims=True)
    target_norms[target_norms == 0] = 1.0
    target_vectors = target_vectors / target_norms

    for job in experience_history:
        company_name = job.get('company', '').strip()
        if not company_name:
            continue
            
        comp_vector = np.array(client.embeddings(model=model_name, prompt=company_name)['embedding'])
        comp_norm = np.linalg.norm(comp_vector)
        if comp_norm == 0: continue
        comp_vector = comp_vector / comp_norm

        similarities = np.dot(target_vectors, comp_vector)
        if np.max(similarities) >= 0.86: # Safe vector threshold for company variant names
            return 100.0
            
    return 0.0

# ---------------------------------------------------------
# MAIN PIPELINE ASSESSMENT COUPLING ENTRYPOINT
# ---------------------------------------------------------

def evaluate_candidate(candidate_data: dict, jd_requirements: JobRequirements, client, model_name: str) -> dict:
    """Aggregates foundational scoring matrices using your exact runtime configuration."""
    history = candidate_data.get('experience_history', [])
    skills = candidate_data.get('global_skills', [])

    # 1. Pre-compute Job Description vectors ONCE to eliminate redundant downstream API loops
    jd_skills_raw = jd_requirements.required_skills
    if jd_skills_raw:
        req_vectors = np.array([client.embeddings(model=model_name, prompt=s)['embedding'] for s in jd_skills_raw])
    else:
        req_vectors = np.array([])

    # 2. Compute clean decoupled mathematical components passing the cached matrix down
    s_skills = score_skills_vector(
        candidate_skills=skills, 
        required_skills=jd_skills_raw, 
        req_vectors=req_vectors, 
        client=client, 
        model_name=model_name
    )
    
    s_experience = score_experience_vector(
        experience_history=history, 
        required_skills=jd_skills_raw, 
        # req_vectors=req_vectors, 
        # client=client, 
        # model_name=model_name, 
        target_months=jd_requirements.target_experience_months
    )
    
    s_seniority = score_seniority_vector(history, jd_requirements.target_seniority)
    s_company = score_company_bonus_vector(history, jd_requirements.target_companies, client, model_name)
    
    # 3. Handle total weight normalization safely
    raw_total_weight = (
        jd_requirements.weight_skills +
        jd_requirements.weight_experience +
        jd_requirements.weight_seniority +
        jd_requirements.weight_company_bonus
    )
    
    if raw_total_weight <= 0:
        raise ValueError("Total allocated configuration weights must equal greater than zero.")

    final_score = (
          (s_skills * jd_requirements.weight_skills)
        + (s_experience * jd_requirements.weight_experience)
        + (s_seniority * jd_requirements.weight_seniority)
        + (s_company * jd_requirements.weight_company_bonus)) / raw_total_weight
    
    return {
        "final_aggregated_score": round(final_score, 2),
        "vector_breakdown": {
            "skills_match_score": round(s_skills, 2),
            "experience_duration_score": round(s_experience, 2),
            "seniority_rank_score": round(s_seniority, 2),
            "competitor_bonus_score": round(s_company, 2)
        }
    }