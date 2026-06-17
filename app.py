#app.py

import os
import json
from concurrent.futures import ThreadPoolExecutor
from utils import parse_resume_to_json_generic, extract_raw_text, identify_missing_skills, generate_remediation_plan
from jd_parser import parse_jd_to_requirements
from scoring_engine import evaluate_candidate
import ollama
import time

# ---------------------------------------------------------
# 3. EXECUTION BLOCK
# ---------------------------------------------------------
if __name__ == "__main__":
    # 1. Configuration of Source Files (Supports .pdf or .docx dynamically)
    sample_cv = "Candidate_CV_Campus.docx"  
    sample_jd_txt = "jd.txt"
    # Load configuration for the Ollama connection
    CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    
    OLLAMA_CLIENT = ollama.Client(host=config.get("ollama_base_url", "http://localhost:11434"))
    EMBED_MODEL = config.get("embedding_model", "nomic-embed-text")

    # 2. File Pre-flight Checks
    if not os.path.exists(sample_cv):
        print(f"Error: Please place a valid CV file named '{sample_cv}' in this directory.")
        exit(1)
        
    if not os.path.exists(sample_jd_txt):
        print(f"Error: Please place a valid text JD file named '{sample_jd_txt}' in this directory.")
        exit(1)

    print("====================================================")
    print("STARTING LOCAL OLLAMA ATS MATCHING PIPELINE")
    print("====================================================\n")

    raw_word_text = extract_raw_text(sample_cv)

    try:
        start = time.time()

        # --- NEW STEP 1 & 2: CONCURRENT PIPELINE PARSING ---
        print("--- RUNNING CONCURRENT CV & JD PARSING LAYER ---")
        
        # 1. Open a thread pool worker context
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Kick off the JD parsing task thread
            jd_future = executor.submit(parse_jd_to_requirements, sample_jd_txt)
            
            # Note: Since your original generic resume parser relies on dynamic_jd_skills,
            # we run it here independently. If it strictly needs the results of the JD inventory first,
            # pass an empty list or omit it, as the dynamic adjustments can happen down-pipeline.
            cv_future = executor.submit(parse_resume_to_json_generic, sample_cv, dynamic_jd_skills=[])

            # 2. Collect threads synchronously as they complete processing loops
            structured_jd = jd_future.result()
            parsed_cv_data = cv_future.result()

        # Extract requirements inventory out of the completed objects
        jd_skills_inventory = structured_jd.required_skills

        print(f"Candidate Name:       {parsed_cv_data.candidate_name}")
        print("\n=== PARSED OUTPUT OBJECT ===")
        print(f"Candidate Name:       {parsed_cv_data.candidate_name}")
        print(f"Global Skills Count:  {len(parsed_cv_data.global_skills)}")
        print(f"Jobs Extracted:       {len(parsed_cv_data.experience_history)}")
        
        # Verify the top welded entity timeline profile
        if parsed_cv_data.experience_history:
            first_job = parsed_cv_data.experience_history[0]
            print(f"\nMost Recent Welded Entity:")
            print(f" - Company:        {first_job.company}")
            print(f" - Title:          {first_job.job_title}")
            print(f" - Dates:          {first_job.start_date} to {first_job.end_date or 'Present'}")
            print(f" - Inferred Tier:  {first_job.inferred_seniority}")
            print(f" - Isolated Tech:  {first_job.skills_used}\n")
                            

        # 5. Step 3: Run Deterministic Math Scoring Engine
        print("--- STEP 3: RUNNING MATHEMATICAL MATRIX MATCH ---")
        candidate_dict = parsed_cv_data.model_dump()

        # Execute pure math matching evaluation
        final_analysis = evaluate_candidate(candidate_data=candidate_dict, jd_requirements=structured_jd, 
                                            client=OLLAMA_CLIENT, model_name=EMBED_MODEL)

        # 6. Step 4: Print Transparent Audit Trail
        print("\n====================================================")
        print("FINAL CANDIDATE ASSESSMENT AUDIT TRAIL")
        print("====================================================")
        print(json.dumps(final_analysis, indent=2))

        # --------------------------------------------------
        # NEW STEP: ISOLATE MISSING SKILLS GAP ANALYSIS
        # --------------------------------------------------
        missing_tech = identify_missing_skills(
            candidate_skills=parsed_cv_data.global_skills,
            required_skills=jd_skills_inventory,
            client=OLLAMA_CLIENT,
            model_name=EMBED_MODEL,
            threshold=0.75 
        )

        skills_gap_json = {
            "candidate_name": parsed_cv_data.candidate_name,
            "skills_gap_analysis": {
                "total_required_count": len(jd_skills_inventory),
                "missing_count": len(missing_tech),
                "missing_skills_isolated": missing_tech
            }
        }
        print("\n====================================================")
        print("🎯 ISOLATED CANDIDATE SKILLS GAP ANALYSIS JSON")
        print("====================================================")
        print(json.dumps(skills_gap_json, indent=2))


        print("\n====================================================")
        print("✨ FINAL REMEDIATION PLAN STRUCTURAL OUTPUT Object ✨")
        print("====================================================")
        target_skills_list = skills_gap_json["skills_gap_analysis"]["missing_skills_isolated"]
        remediation_result = generate_remediation_plan(raw_jd=sample_jd_txt, employment_history=parsed_cv_data.experience_history, missing_skills=target_skills_list)
        output_dict = remediation_result.model_dump()

        print(json.dumps(output_dict["remediation_plan"], indent=2))
        print("====================================================\n")
        elapsed = time.time() - start
        print(elapsed)

    except Exception as e:
        print(f"\n[PIPELINE FAILURE] An error occurred during processing: {str(e)}")
        raise e