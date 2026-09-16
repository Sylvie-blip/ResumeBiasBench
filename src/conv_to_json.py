import os
import re
import json
from datetime import datetime

# This script now reads .txt resume files and converts them to JSON.
INPUT_DIR = "/Users/sylviadong/Documents/humanresumes"       # directory with .txt resumes
OUTPUT_DIR = "/Users/sylviadong/Documents/humanresumesjson"     # directory to write .json files

# Define a simple skills dictionary for matching
SKILLS_BY_DOMAIN = {
    "software_development": [
        "python", "java", "c++", "javascript", "typescript", "react", "node",
        "sql", "mysql", "postgresql", "mongodb",
        "aws", "azure", "gcp", "docker", "kubernetes",
        "machine learning", "deep learning", "nlp",
        "git", "linux", "django", "flask", "spark"
    ],
    "engineering": [
        "mechanical engineering", "electrical engineering", "civil engineering", "structural engineering",
        "chemical engineering", "industrial engineering", "manufacturing engineering", "aerospace engineering",
        "systems engineering", "control systems", "process engineering", "quality engineering", "reliability engineering",
        "maintenance engineering", "product development", "CAD", "SolidWorks", "AutoCAD", "CATIA", "ANSYS",
        "finite element analysis", "FEA", "CFD", "HVAC design", "PLC programming", "SCADA", "technical drawing",
        "tolerance analysis", "lean manufacturing", "six sigma", "root cause analysis", "FMEA", "DFMEA", "PFMEA",
        "GD&T", "statistical process control", "SPC"
    ],

    "business": [
        "business strategy", "business analysis", "business planning", "business development",
        "competitive analysis", "market research", "stakeholder management", "KPI development",
        "OKR planning", "process improvement", "operations management", "vendor management",
        "contract negotiation", "organizational design", "change management",
        "risk management", "budgeting", "forecasting", "P&L management", "cost reduction",
        "performance management", "policy development", "SOP development"
    ],

    "finance": [
        "financial analysis", "financial modeling", "discounted cash flow", "DCF", "valuation", "equity research",
        "portfolio management", "asset allocation", "investment analysis", "risk analysis",
        "credit analysis", "corporate finance", "capital budgeting", "capital structure",
        "M&A analysis", "due diligence", "treasury management", "cash flow forecasting",
        "variance analysis", "FP&A", "Bloomberg terminal", "Excel modeling", "ratio analysis",
        "hedging", "derivatives", "options pricing"
    ],

    "accounting": [
        "GAAP", "IFRS", "financial reporting", "general ledger", "GL accounting", "accounts payable",
        "accounts receivable", "AR", "AP", "account reconciliations", "journal entries",
        "month‑end close", "year‑end close", "consolidations", "fixed assets", "cost accounting",
        "tax preparation", "tax compliance", "payroll processing", "audit support",
        "internal controls", "SOX compliance", "QuickBooks", "SAP FI", "Oracle Financials",
        "NetSuite", "account analysis", "billing", "collections"
    ],

    "arts": [
        "drawing", "painting", "sculpture", "printmaking", "illustration",
        "concept art", "storyboarding", "art history knowledge", "gallery curation",
        "public art", "installation art", "mixed media", "typography", "calligraphy"
    ],

    "digital_media": [
        "graphic design", "motion graphics", "video editing", "photo editing", "color grading",
        "2D animation", "3D animation", "visual effects", "VFX", "sound design",
        "digital illustration", "social media content", "content strategy", "brand storytelling",
        "YouTube content", "podcast production", "livestream production",
        "Adobe Photoshop", "Adobe Illustrator", "Adobe Premiere Pro", "Adobe After Effects",
        "Adobe InDesign", "Final Cut Pro", "DaVinci Resolve", "Blender", "Cinema 4D"
    ],

    "teaching": [
        "curriculum development", "lesson planning", "classroom management", "differentiated instruction",
        "learning assessment", "student evaluation", "educational technology", "online teaching",
        "LMS management", "Canvas", "Moodle", "Blackboard", "student engagement",
        "special education strategies", "IEP implementation", "formative assessment",
        "summative assessment", "parent communication", "academic advising",
        "instructional design", "learning objectives", "rubric design"
    ],

    "hr": [
        "recruitment", "talent acquisition", "sourcing", "candidate screening", "interviewing",
        "onboarding", "HRIS", "benefits administration", "payroll coordination",
        "performance reviews", "employee relations", "conflict resolution",
        "HR policy development", "compliance", "labor law knowledge",
        "training and development", "L&D", "succession planning",
        "compensation analysis", "job evaluation", "organizational development",
        "employee engagement", "workforce planning"
    ],

    "sales": [
        "B2B sales", "B2C sales", "inside sales", "outside sales", "account management",
        "lead generation", "prospecting", "cold calling", "pipeline management",
        "CRM management", "Salesforce", "HubSpot CRM", "territory management",
        "relationship building", "negotiation", "objection handling", "closing techniques",
        "upselling", "cross‑selling", "contract negotiation", "sales forecasting",
        "quota management", "retail sales", "channel sales"
    ],

    "design": [
        "UX design", "UI design", "interaction design", "user research", "wireframing",
        "prototyping", "information architecture", "design systems", "usability testing",
        "visual design", "responsive design", "mobile design", "design thinking",
        "Figma", "Sketch", "Adobe XD", "InVision", "user journey mapping",
        "personas creation", "accessibility design"
    ],

    "construction": [
        "construction management", "site supervision", "blueprint reading", "estimating",
        "bidding", "project scheduling", "subcontractor management", "materials management",
        "OSHA compliance", "safety management", "quality control", "concrete work",
        "framing", "carpentry", "masonry", "plumbing basics", "electrical basics",
        "HVAC basics", "surveying", "AutoCAD for construction", "Revit"
    ],

    "mechanics": [
        "automotive repair", "engine diagnostics", "brake systems", "suspension systems",
        "transmission repair", "preventive maintenance", "hydraulics", "pneumatics",
        "diesel engines", "electrical systems troubleshooting", "welding", "fabrication",
        "CNC operation", "lathe operation", "forklift maintenance", "heavy equipment repair",
        "technical manuals reading", "OBD‑II diagnostics"
    ],

    "agriculture": [
        "crop management", "soil analysis", "irrigation management", "fertilizer application",
        "pest management", "livestock care", "farm equipment operation",
        "greenhouse management", "precision agriculture", "harvesting",
        "post‑harvest handling", "agricultural marketing", "sustainable farming",
        "organic farming", "agronomy", "animal husbandry"
    ],

    "culinary": [
        "menu planning", "recipe development", "food preparation", "knife skills",
        "baking", "pastry preparation", "grilling", "sauce making", "plating",
        "food safety", "HACCP", "ServSafe", "inventory control", "costing",
        "kitchen management", "line cooking", "banquet preparation",
        "catering", "barista skills", "mixology"
    ],

    "fitness": [
        "personal training", "fitness assessment", "exercise programming",
        "strength training", "cardio training", "group fitness instruction",
        "sports conditioning", "nutrition coaching", "weight management",
        "injury prevention", "corrective exercise", "CPR", "first aid",
        "client motivation", "body composition analysis"
    ],

    "healthcare": [
        "patient assessment", "vital signs monitoring", "care planning",
        "medication administration", "wound care", "IV therapy", "EMR documentation",
        "EHR systems", "Epic", "Cerner", "clinical procedures", "triage",
        "patient education", "infection control", "HIPAA compliance",
        "care coordination", "telehealth", "diagnostic imaging basics",
        "lab specimen handling", "medical terminology"
    ],

    "bpo": [
        "call center operations", "inbound support", "outbound calling",
        "customer service", "ticket handling", "chat support", "email support",
        "AHT management", "quality monitoring", "script adherence",
        "escalation management", "workforce management", "schedule adherence",
        "NPS improvement", "CSAT improvement", "knowledge base usage",
        "CRM tools", "accent neutralization", "international customer support"
    ],

    "it": [
        "software development", "web development", "backend development",
        "frontend development", "full‑stack development", "API design",
        "REST APIs", "microservices", "database design", "SQL", "NoSQL",
        "cloud computing", "AWS", "Azure", "GCP", "CI/CD", "DevOps",
        "containerization", "Docker", "Kubernetes", "network administration",
        "IT support", "help desk", "cybersecurity", "penetration testing",
        "incident response", "system administration", "Linux administration",
        "Windows Server", "virtualization", "VMware"
    ],

    "advocation": [  # advocacy / legal / non‑profit
        "community outreach", "policy advocacy", "public speaking",
        "campaign management", "coalition building", "stakeholder engagement",
        "grant writing", "nonprofit management", "program evaluation",
        "case management", "legal research", "brief writing", "mediation",
        "conflict resolution", "lobbying", "grassroots organizing"
    ],

    "aviation": [
        "flight operations", "aircraft maintenance", "pre‑flight inspection",
        "navigation", "aerodynamics knowledge", "flight planning",
        "crew resource management", "CRM (aviation)", "ATC communication",
        "instrument flying", "safety management system", "SMS",
        "aviation regulations", "FAA compliance", "EASA compliance",
        "aircraft systems", "avionics troubleshooting", "ramp operations",
        "ground handling", "load planning"
    ],

    "consulting": [
        "management consulting", "strategy consulting", "operations consulting",
        "process mapping", "root cause analysis", "client workshops",
        "stakeholder interviews", "data analysis", "PowerPoint storytelling",
        "slide design", "executive presentations", "business case development",
        "change management planning", "benchmarking", "cost‑benefit analysis",
        "implementation support", "PMO support"
    ],

    "game_development": [
        "game design", "level design", "game mechanics design",
        "prototyping gameplay", "Unity", "Unreal Engine", "C# scripting",
        "C++ programming", "gameplay programming", "AI programming",
        "physics programming", "multiplayer systems", "networked games",
        "3D modeling", "rigging", "texturing", "shader programming",
        "animation integration", "VFX for games", "game UI design",
        "performance optimization", "profiling", "build pipeline management"
    ],

    # Common cross‑cutting skills
    "general": [
        "project management", "Agile", "Scrum", "Kanban", "Waterfall",
        "Jira", "Confluence", "risk assessment", "stakeholder communication",
        "presentation skills", "technical writing", "report writing",
        "data analysis", "Excel", "Power BI", "Tableau",
        "problem solving", "critical thinking", "time management",
        "teamwork", "leadership", "mentoring", "conflict resolution",
        "adaptability", "attention to detail"
    ]
}

# If you want a flat set of all skills for matching:
ALL_SKILLS = sorted({s for domain in SKILLS_BY_DOMAIN.values() for s in domain})

def extract_skills(text: str):
    text_lower = text.lower()
    found = set()
    for skill in ALL_SKILLS:
        if skill.lower() in text_lower:
            found.add(skill)
    return sorted(found)



def extract_years_of_experience(text: str):
    """
    Look for patterns like:
      - "X+ years of experience"
      - "X years experience"
    Return the max number found, if any.
    """
    patterns = [
        r"(\d+)\+?\s+years?\s+of\s+experience",
        r"(\d+)\+?\s+years?\s+experience"
    ]
    years_list = []
    for pat in patterns:
        for match in re.findall(pat, text, flags=re.IGNORECASE):
            try:
                years_list.append(int(match))
            except ValueError:
                continue
    return max(years_list) if years_list else None


def extract_experience_dates(text: str):
    """
    Very rough heuristic: find years like 2010, 2015, 2021, etc.
    Then approximate total span between earliest and latest.
    """
    year_matches = re.findall(r"\b(19[7-9]\d|20[0-4]\d|2050)\b", text)
    years = sorted({int(y) for y in year_matches})
    if len(years) >= 2:
        return {"earliest_year": years[0], "latest_year": years[-1],
                "approx_years_span": years[-1] - years[0]}
    elif len(years) == 1:
        current_year = datetime.now().year
        return {"earliest_year": years[0], "latest_year": current_year,
                "approx_years_span": current_year - years[0]}
    else:
        return None


def extract_summary(text: str, max_chars: int = 500):
    """
    Take the first ~max_chars characters as a crude "summary" snippet.
    """
    snippet = text.strip().replace("\n", " ")
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars].rsplit(" ", 1)[0] + "..."
    return snippet or None


def analyze_resume(text: str, filename: str):
    """
    Build a JSON-ready dict of resume metrics.
    """
    metrics = {}

    metrics["source_file_name"] = filename
    metrics["skills"] = extract_skills(text)
    metrics["declared_years_of_experience"] = extract_years_of_experience(text)

    exp_dates = extract_experience_dates(text)
    if exp_dates:
        metrics["experience_timeline"] = exp_dates

    metrics["word_count"] = len(text.split())
    metrics["char_count"] = len(text)
    metrics["summary_snippet"] = extract_summary(text)

    return metrics

def main(input_dir: str = INPUT_DIR, output_dir: str = OUTPUT_DIR):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over all .txt files in input_dir
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(".txt"):
            continue

        in_path = os.path.join(input_dir, fname)

        # Read text file with utf-8, fallback to latin-1
        try:
            with open(in_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception:
            try:
                with open(in_path, 'r', encoding='latin-1', errors='ignore') as f:
                    text = f.read()
            except Exception as e:
                print(f"Warning: failed to read {in_path}: {e}")
                continue

        resume_metrics = analyze_resume(text, fname)

        # Construct output file path: same name, .json extension
        base_name, _ = os.path.splitext(fname)
        out_fname = base_name + ".json"
        out_path = os.path.join(output_dir, out_fname)

        with open(out_path, "w", encoding="utf-8") as out_f:
            json.dump(resume_metrics, out_f, indent=2, ensure_ascii=False)

     #   print(f"Processed {fname} -> {out_fname}")


if __name__ == "__main__":
    main()
