#scoring_models.py

import os
import json
from typing import List, Optional
from pydantic import BaseModel, Field
from models import SafeSkill 

# Load configuration for default weight assignments
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

weights_cfg = config.get("default_weights", {})

class JobRequirements(BaseModel):
    required_skills: List[SafeSkill]  
    target_experience_months: int  
    target_seniority: str  
    target_companies: List[str]
    
    # Weights default directly to your JSON configurations if omitted
    weight_skills: float = Field(default=weights_cfg.get("weight_skills", 0.40))
    weight_experience: float = Field(default=weights_cfg.get("weight_experience", 0.30))
    weight_seniority: float = Field(default=weights_cfg.get("weight_seniority", 0.20))
    weight_company_bonus: float = Field(default=weights_cfg.get("weight_company_bonus", 0.10))