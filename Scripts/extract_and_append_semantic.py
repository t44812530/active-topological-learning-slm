import re

with open("JIT_ATL/Scripts/phase_15_retrieval_output.txt", "r") as f:
    raw_output = f.read()

blocks = raw_output.split("LOADING: ")
model_data = {}

for block in blocks[1:]:
    lines = block.strip().split("\n")
    current_model = lines[0].strip()
    
    match = re.search(r"6\. Semantic Retrieval Proof\n   Query: '.*?'\n   Top 5 Nearest:\n((?:   -> \[.*?\] \(d=.*?\)\n?){5})", block)
    if match:
        chunk = match.group(1).strip().split("\n")
        out = []
        for i, line in enumerate(chunk):
            m = re.match(r"-> \[(.*?)\] \((d=.*?)\)", line.strip())
            if m:
                out.append(f"  {i+1}. `{m.group(1)}` ({m.group(2)})")
        model_data[current_model] = "\n".join(out)

md_titles = {
    "MODEL A (BASELINE)": "* **Baseline Model Top 5:**",
    "MODEL B (RIGID PINNED)": "* **Rigid Model Top 5:**",
    "MODEL B (ORGANIC)": "* **Organic Model Top 5:**",
    "MODEL B (TRAJECTORY)": "* **Trajectory Model Top 5:**",
    "MODEL B (HEAVY REPETITION)": "* **Heavy Repetition Model Top 5:**",
    "MODEL B (ASYMMETRIC)": "* **Asymmetric Model Top 5:**",
    "MODEL B (SCAFFOLD)": "* **Parallelogram Scaffold Model Top 5:**",
    "MODEL B (OMNI-SCAFFOLD)": "* **Omni-Scaffold Model Top 5:**",
    "MODEL B (PRECISION OMNI LIGHT)": "* **Precision Omni (Light) Model Top 5:**",
    "MODEL B (ABSOLUTE OMNI CAGE)": "* **Absolute Omni Cage Top 5:**"
}

phase_15_section = """
### Phase 15: The Semantic Retrieval Proof
*Objective: Test natural language semantic retrieval to prove that Active Topological Learning correctly mapped the synthetic concept 'Aether-Node' to the described physical domain ('the physical switching mechanism of a topological quantum computer').*
*Executed via: `JIT_ATL/Scripts/measure_semantic_retrieval.py`*

**Query:** `the physical switching mechanism of a topological quantum computer`

**Expanded Semantic Retrieval Top 5:**
"""

for model_key, title in md_titles.items():
    if model_key in model_data:
        phase_15_section += f"{title}\n{model_data[model_key]}\n\n"

with open("JIT_ATL/Test_Reports/Test_Report_JIT-ATL-v01.md", "r") as f:
    md_content = f.read()

if "### Phase 15" not in md_content:
    md_content = md_content.replace("---\n\n## 3. Observations & Anomalies", phase_15_section + "---\n\n## 3. Observations & Anomalies")

    with open("JIT_ATL/Test_Reports/Test_Report_JIT-ATL-v01.md", "w") as f:
        f.write(md_content)
    print("Markdown updated successfully.")
else:
    print("Phase 15 already exists.")
