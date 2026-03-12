import re

with open("JIT_ATL/Scripts/phase_14_triplets.txt", "r") as f:
    text = f.read()

start_marker = "# 6. THE ABSOLUTE OMNI-SCAFFOLD STRUCTURE (385-Point N+1 Dimensional Lock)"
triplets_match = re.search(r"(" + re.escape(start_marker) + r".*)", text, re.DOTALL)

if triplets_match:
    triplets_block = "        " + triplets_match.group(1).replace("\n", "\n        ").strip() + "\n"
else:
    print("Could not find triplets in output.")
    exit(1)

with open("JIT_ATL/Scripts/train_micro_corpus_absolute.py", "r") as f:
    train_script = f.read()

# Replace the specific block of triplets inside train_micro_corpus_absolute.py
old_start = "# 6. THE ABSOLUTE OMNI-SCAFFOLD STRUCTURE"
if old_start in train_script:
    # already replaced?
    pass
else:
    old_start = "# 6. THE OMNI-SCAFFOLD STRUCTURE"

replaced_script = re.sub(r'        ' + re.escape(old_start) + r'.*?(?=    \])', triplets_block, train_script, flags=re.DOTALL)

# Also update the save path and prints
replaced_script = replaced_script.replace('Model_B_Omni_Light', 'Model_B_Absolute')
replaced_script = replaced_script.replace('PHASE 10 - OMNI-SCAFFOLD OVERDRIVE', 'PHASE 14 - ABSOLUTE OMNI CAGE')
replaced_script = replaced_script.replace('Omni-Scaffold Micro-Corpus (20-Point Lock)', 'Absolute Omni-Scaffold Micro-Corpus (385-Point Lock)')

with open("JIT_ATL/Scripts/train_micro_corpus_absolute.py", "w") as f:
    f.write(replaced_script)

print("Absolute training script generated successfully.")
