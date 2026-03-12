import re

with open("JIT_ATL/Scripts/phase_14_light_origin_output.txt", "r") as f:
    raw_output = f.read()

blocks = raw_output.split("LOADING: ")
model_data = {}

for block in blocks[1:]:
    lines = block.strip().split("\n")
    current_model = lines[0].strip()
    
    b_idx = block.find("4B. Light Origin")
    c_idx = block.find("4C. Light Origin")
    end_idx = block.find("5. Perimeter Shift")
    
    if b_idx != -1 and c_idx != -1:
        light1_chunk = block[b_idx:c_idx]
        light2_chunk = block[c_idx:end_idx]
        
        def extract_top5(chunk):
            matches = re.findall(r"-> \[(.*?)\] \((d=.*?)\)", chunk)
            out = []
            for i, (word, dist) in enumerate(matches[:5]):
                out.append(f"  {i+1}. `{word}` ({dist})")
            return "\n".join(out)
            
        model_data[current_model] = {
            "light1": extract_top5(light1_chunk),
            "light2": extract_top5(light2_chunk)
        }

print(f"Extracted data for {len(model_data)} models.")

target_map = {
    "MODEL A (BASELINE)": "* **Model A (Baseline) Top 5:**",
    "MODEL B (RIGID PINNED)": "* **Rigid Model Top 5:**",
    "MODEL B (ORGANIC)": "* **Organic Model Top 5:**",
    "MODEL B (TRAJECTORY)": "* **Trajectory Model Top 5:**",
    "MODEL B (HEAVY REPETITION)": "* **Heavy Repetition Model Top 5:**",
    "MODEL B (ASYMMETRIC)": "* **Asymmetric Model Top 5:**",
    "MODEL B (SCAFFOLD)": "* **Parallelogram Scaffold Model Top 5:**",
    "MODEL B (OMNI-SCAFFOLD)": "* **Omni-Scaffold (Heavy) Model Top 5:**",
    "MODEL B (PRECISION OMNI LIGHT)": "* **Precision Omni (Light) Model Top 5:**",
    "MODEL B (ABSOLUTE OMNI CAGE)": "* **Absolute Omni Cage Top 5:**"
}

md_titles = {
    "MODEL A (BASELINE)": "* **Baseline Model",
    "MODEL B (RIGID PINNED)": "* **Rigid Model",
    "MODEL B (ORGANIC)": "* **Organic Model",
    "MODEL B (TRAJECTORY)": "* **Trajectory Model",
    "MODEL B (HEAVY REPETITION)": "* **Heavy Repetition Model",
    "MODEL B (ASYMMETRIC)": "* **Asymmetric Model",
    "MODEL B (SCAFFOLD)": "* **Parallelogram Scaffold Model",
    "MODEL B (OMNI-SCAFFOLD)": "* **Omni-Scaffold (Heavy) Model",
    "MODEL B (PRECISION OMNI LIGHT)": "* **Precision Omni (Light) Model",
    "MODEL B (ABSOLUTE OMNI CAGE)": "* **Absolute Omni Cage"
}

with open("JIT_ATL/Test_Reports/Test_Report_JIT-ATL-v01.md", "r") as f:
    md_content = f.read()

for out_name, md_target in target_map.items():
    if out_name not in model_data:
        print(f"Missing data for {out_name}")
        continue
    
    title_prefix = md_titles[out_name]
    light_1_str = f"{title_prefix} Light Origin 1 (Origin = microchip):**\n{model_data[out_name]['light1']}\n"
    light_2_str = f"{title_prefix} Light Origin 2 (Origin = classical bit):**\n{model_data[out_name]['light2']}\n"
    
    pattern = r"(" + re.escape(md_target) + r"\n(?:  \d\. `.*?` \(d=.*?\)\n?){5})"
    
    if "Light Origin 1" in md_content[md_content.find(md_target):md_content.find(md_target)+400]:
        print(f"Already appended for {out_name}")
        continue
        
    def repl(m):
        return m.group(1) + "\n\n" + light_1_str + "\n" + light_2_str
        
    md_content, count = re.subn(pattern, repl, md_content)
    if count == 0:
        print(f"Failed to find match for {md_target}")
    else:
        print(f"Appended for {out_name}")

with open("JIT_ATL/Test_Reports/Test_Report_JIT-ATL-v01.md", "w") as f:
    f.write(md_content)

print("Markdown updated successfully.")
