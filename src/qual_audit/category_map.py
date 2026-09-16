"""Maps sorted_data occupational category folders to conv_to_json.SKILLS_BY_DOMAIN keys,
and loads the id -> category lookup from data/sorted_data/.

21 of 24 categories map to a single domain 1:1. Three have no clean match and are
pinned to an explicit fallback (documented in the audit plan, not left to guesswork):
  BANKING           -> finance (banking is a reasonable subset of the finance list)
  APPAREL           -> design + general
  PUBLIC-RELATIONS  -> digital_media + business
"""

from pathlib import Path

CATEGORY_TO_DOMAINS = {
    "ACCOUNTANT": ["accounting"],
    "ADVOCATE": ["advocation"],
    "AGRICULTURE": ["agriculture"],
    "APPAREL": ["design", "general"],
    "ARTS": ["arts"],
    "AUTOMOBILE": ["mechanics"],
    "AVIATION": ["aviation"],
    "BANKING": ["finance"],
    "BPO": ["bpo"],
    "BUSINESS-DEV": ["business"],
    "CHEF": ["culinary"],
    "CONSTRUCTION": ["construction"],
    "CONSULTANT": ["consulting"],
    "DESIGNER": ["design"],
    "DIGITAL-MEDIA": ["digital_media"],
    "ENGINEERING": ["engineering"],
    "FINANCE": ["finance"],
    "FITNESS": ["fitness"],
    "HEALTHCARE": ["healthcare"],
    "HR": ["hr"],
    "INFORMATION-TECH": ["it"],
    "PUBLIC-RELATIONS": ["digital_media", "business"],
    "SALES": ["sales"],
    "TEACHER": ["teaching"],
}

IMPERFECT_MAPPINGS = {"BANKING", "APPAREL", "PUBLIC-RELATIONS"}


def load_id_to_category(sorted_data_dir: str) -> dict[str, str]:
    """Walk data/sorted_data/<CATEGORY>/<id>.pdf and return {id: CATEGORY}.

    Only *.pdf files are considered (skips .DS_Store and other stray files).
    Raises if the same id appears under more than one category folder.
    """
    root = Path(sorted_data_dir)
    id_to_category: dict[str, str] = {}
    collisions: dict[str, list[str]] = {}

    for category_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        category = category_dir.name
        for pdf in category_dir.glob("*.pdf"):
            rid = pdf.stem
            if rid in id_to_category and id_to_category[rid] != category:
                collisions.setdefault(rid, [id_to_category[rid]]).append(category)
                continue
            id_to_category[rid] = category

    if collisions:
        raise ValueError(f"IDs found under multiple categories: {collisions}")

    return id_to_category


def domains_for_category(category: str) -> list[str]:
    if category not in CATEGORY_TO_DOMAINS:
        raise KeyError(f"Unmapped category: {category}")
    return CATEGORY_TO_DOMAINS[category]
