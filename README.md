# Application-Tracking-System-Powered-by-Groq-Llama-70B ⚡

An AI-native, ultra-high-speed Applicant Tracking System (ATS) matching, semantic scoring, and automated candidate remediation pipeline. This architecture leverages **Meta's Llama-3.3-70B-Versatile** via **Groq's LPU cloud infrastructure** paired with **Pydantic** data contracts to achieve deep semantic parsing and structured schema delivery in seconds.

---

## 🚀 The Performance Leap
* **Legacy Architecture (Local Qwen-2.5-7B Engine):** ~600 seconds (10 minutes) per document due to localized VRAM throughput bounds and sequential extraction loops.
* **Optimized Groq-Llama-70B Architecture:** **Under 15 seconds** total execution time for a concurrent multi-document compilation (**100x speedup**).

---

## 🛠️ Core Capabilities

1. **Layout-Agnostic Resume Ingestion:** A single-pass, zero-hardcoded JSON segmentation layer. It effortlessly parses complex multi-column resumes, graphics, and chaotic layouts, isolating chronological professional blocks without dropping context.
2. **Dynamic Taxonomy Alignment:** Instead of relying on a rigid, outdated keyword list, the system takes the incoming Job Description (JD) text and dynamically builds a runtime evaluation matrix.
3. **Semantic Vector Matching:** Bypasses fragile exact-string matching by running localized text embeddings (`nomic-embed-text`) to map synonyms, related software stacks, and conceptual domain alignments.
4. **Contextual Skill Remediation:** Isolates core technical missing gaps and programmatically engineers high-impact, professional resume bullet points mapped natively to specific historical career blocks (enforcing strict business rules against generic definitions).

---

## 📐 System Architecture & Data Flow

The engine utilizes an asynchronous thread pool to read, segment, and normalize both incoming files simultaneously before executing downstream vector math and strategic bullet remediation:

![System Architecture](architecture.png)


## ⚙️ Local Installation & Setup

### 1. Clone the Active Repository
```bash
git clone [https://github.com/2ahmedabdullah/Application-Tracking-System-Powered-by-Groq-Llama-70B.git](https://github.com/2ahmedabdullah/Application-Tracking-System-Powered-by-Groq-Llama-70B.git)
cd Application-Tracking-System-Powered-by-Groq-Llama-70B


# Windows
```bash
python -m venv ats_env
.\ats_env\Scripts\activate

# Mac/Linux
```bash
python3 -m venv ats_env
source ats_env/bin/activate


### Install Core Production Dependencies
```bash
pip install -r requirements.txt


## 📊 Live Execution Snippets & Production Output

When you fire up the engine, the pipeline processes the layout structures concurrently, builds the entity mapping, and returns a strict mathematical scorecard:

### ⚡ Step 1 & 2: Asynchronous Multi-Document Ingestion Trace
```text
====================================================
PARSED CV OBJECT
====================================================

Candidate Name:       John Doe
Global Skills Count:  47
Jobs Extracted:       33

Most Recent Experience:
 - Company:        Walmart Global, AK, USA
 - Title:          Senior AI Engineer
 - Dates:          2024-08 to 2026-07
 - Inferred Tier:  Senior_IC
 - Isolated Tech:  ['NLP', 'LLMs', 'DSPy', 'Pydantic', 'AWS', 'LangGraph', 'vLLM', 'Redis', 'Plotly']

=====================================================
FINAL CANDIDATE ASSESSMENT AUDIT TRAIL
=====================================================
{
  "final_aggregated_score": 67.0,
  "vector_breakdown": {
    "skills_match_score": 50.0,
    "experience_duration_score": 79.0,
    "seniority_rank_score": 100.0,
    "competitor_bonus_score": 100.0
  }
}