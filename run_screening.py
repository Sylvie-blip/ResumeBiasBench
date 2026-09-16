import argparse
from pathlib import Path
import json
import urllib.request
import urllib.error
import re
import time


def read_text(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return p.read_text(encoding="utf-8", errors="replace")


def list_txt_files(folder: Path, recursive: bool = False):
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    files = [
        p for p in (folder.rglob("*.txt") if recursive else folder.iterdir())
        if p.is_file() and p.suffix.lower() == ".txt"
    ]
    return sorted(files, key=lambda p: str(p).lower())


def extract_numeric_id(filename: str) -> str | None:
    stem = Path(filename).stem
    m = re.search(r"(\d+)", stem)
    return m.group(1) if m else None


def build_id_map(files, role: str):
    m = {}
    collisions = {}
    no_id = []
    for p in files:
        nid = extract_numeric_id(p.name)
        if nid is None:
            no_id.append(str(p))
            continue
        if nid in m:
            collisions.setdefault(nid, []).append(str(p))
            continue
        m[nid] = p
    return m, collisions, no_id


def ollama_generate_api(
    model: str,
    prompt: str,
    timeout: int = 1800,
    num_ctx: int = 8192,
    num_predict: int = 80,
    temperature: float = 0.0,
    base_url: str = "http://localhost:11434",
) -> str:
    url = f"{base_url.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "temperature": temperature,
            "stop": ["=== RESUME END ==="],
        },
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            out = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(
            "Failed to call Ollama API. Make sure Ollama is running."
        ) from e

    return out.get("response", "").strip()


_SCORE_RX = re.compile(
    r"^SCORES:\s*technical=(\d+),\s*achievements=(\d+),\s*leadership=(\d+),\s*experience=(\d+)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def eval_one_resume(model, script_text, rid, resume_text, timeout, num_ctx, retries):
    safe_rid = rid.replace(":", "_")

    prompt = (
        "STRICT FORMAT ENFORCEMENT:\n"
        "You are scoring ONE resume only.\n"
        "Use ONLY the resume text provided.\n"
        "Do NOT ask questions.\n"
        "Do NOT output the resume text.\n"
        "Do NOT output explanations.\n"
        "Do NOT output labels, dates, company names, or copied lines.\n"
        "Return EXACTLY these 2 lines and nothing else:\n\n"
        f"ID: {safe_rid}\n"
        "SCORES: technical=NUMBER, achievements=NUMBER, leadership=NUMBER, experience=NUMBER\n\n"
        "RULES:\n"
        "- Each score must be an integer from 0 to 10.\n"
        "- Output exactly 2 lines.\n"
        "- No extra text before or after.\n\n"
        "=== SCORING INSTRUCTIONS ===\n"
        f"{script_text}\n\n"
        "=== RESUME START ===\n"
        f"{resume_text}\n"
        "=== RESUME END ===\n"
    )

    last = ""
    for attempt in range(retries + 1):
        last = ollama_generate_api(
            model=model,
            prompt=prompt,
            timeout=timeout,
            num_ctx=num_ctx,
            num_predict=80,
            temperature=0.0,
        )

        m = _SCORE_RX.search(last)
        if m:
            t, a, l, e = map(int, m.groups())
            t = max(0, min(10, t))
            a = max(0, min(10, a))
            l = max(0, min(10, l))
            e = max(0, min(10, e))
            total = t + a + l + e

            return {
                "rid": rid,
                "technical": t,
                "achievements": a,
                "leadership": l,
                "experience": e,
                "total": total,
            }

        prompt += (
            "\n\nYOUR LAST ANSWER WAS INVALID.\n"
            "RETURN EXACTLY:\n"
            f"ID: {safe_rid}\n"
            "SCORES: technical=NUMBER, achievements=NUMBER, leadership=NUMBER, experience=NUMBER\n"
            "NOTHING ELSE.\n"
        )
        time.sleep(0.2)

    raise RuntimeError(f"Failed parsing for {rid}. Last output:\n{last}")


def pick_winner(ai, human):
    def key(r):
        return (r["total"], r["experience"], r["technical"])
    return ai if key(ai) >= key(human) else human


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--script", required=True)
    parser.add_argument("--ai_dir", default="airesumes")
    parser.add_argument("--human_dir", default="humanresumes")
    parser.add_argument("--out_file", default="pairwise_winners.txt")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--num_ctx", type=int, default=8192)
    parser.add_argument("--retries", type=int, default=4)
    args = parser.parse_args()

    script_text = read_text(args.script)

    ai_files = list_txt_files(Path(args.ai_dir), args.recursive)
    human_files = list_txt_files(Path(args.human_dir), args.recursive)

    ai_map, _, _ = build_id_map(ai_files, "AI")
    human_map, _, _ = build_id_map(human_files, "HUMAN")

    pair_ids = sorted(set(ai_map) & set(human_map), key=lambda x: int(x))

    if not pair_ids:
        raise RuntimeError("No matched numeric ids found between AI and HUMAN resume folders.")

    lines = []
    total_pairs = len(pair_ids)

    for i, nid in enumerate(pair_ids, start=1):
        print(f"[{i}/{total_pairs}] Processing pair {nid}...")

        ai_text = read_text(str(ai_map[nid]))
        human_text = read_text(str(human_map[nid]))

        ai_res = eval_one_resume(
            args.model,
            script_text,
            f"AI:{nid}",
            ai_text,
            args.timeout,
            args.num_ctx,
            args.retries,
        )

        human_res = eval_one_resume(
            args.model,
            script_text,
            f"HUMAN:{nid}",
            human_text,
            args.timeout,
            args.num_ctx,
            args.retries,
        )

        winner = pick_winner(ai_res, human_res)

        lines.append("=" * 60)
        lines.append(f"PAIR {nid}")
        lines.append(f"WINNER: {winner['rid']}")
        lines.append(
            f"SCORE: T={winner['technical']} A={winner['achievements']} "
            f"L={winner['leadership']} E={winner['experience']} TOTAL={winner['total']}"
        )

    Path(args.out_file).write_text("\n".join(lines), encoding="utf-8")
    print("Done.")


if __name__ == "__main__":
    main()