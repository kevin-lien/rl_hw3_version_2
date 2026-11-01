"""Verify tested sources and refresh the four-file Astro submission."""
import ast
import hashlib
import json
from pathlib import Path
import fitz

root=Path(__file__).resolve().parents[1]
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
proof=json.loads((root/'results/reproduction_check.json').read_text())
assert proof['all_policy_tensors_equal'] and proof['maximum_absolute_difference']==0
assert sha(root/'train_astro_party.py')==proof['training_source_sha256']
assert sha(root/'envs/train_astro_party_env.py')==proof['training_env_sha256']
for filename in ['train_astro_party.py','envs/train_astro_party_env.py']:
    tree=ast.parse((root/filename).read_text(encoding='utf-8'))
    imports=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):
            imports.extend(alias.name.split('.')[0] for alias in node.names)
        elif isinstance(node,ast.ImportFrom):
            imports.append((node.module or '').split('.')[0])
    assert not set(imports)&{'sys','os','shutil','pathlib','subprocess'},imports
for entry in json.loads((root/'results/protected_hashes.json').read_text(encoding='utf-8-sig')):
    assert sha(Path(entry['Path'])).upper()==entry['Hash'].upper(),entry['Path']
assert sha(root/'models/model_astro_party.zip')==proof['reference_sha256']
pdf=root.parent/'output/pdf/Astro_Party_Report.pdf'
doc=fitz.open(pdf)
assert len(doc)==11,len(doc)
renders=root.parent/'tmp/pdfs/fast_cover'
renders.mkdir(parents=True,exist_ok=True)
for page_number in [8,9,10]:
    doc[page_number].get_pixmap(matrix=fitz.Matrix(1.25,1.25)).save(str(renders/f'page-{page_number+1}.png'))
target=root/'submission_astro_party'
target.mkdir(exist_ok=True)
for source,name in [(root/'train_astro_party.py','train_astro_party.py'),
                    (root/'envs/train_astro_party_env.py','train_astro_party_env.py'),
                    (root/'models/model_astro_party.zip','model_astro_party.zip'),(pdf,'report.pdf')]:
    (target/name).write_bytes(source.read_bytes())
assert len(list(target.iterdir()))==4
manifest=[dict(name=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(target.iterdir())]
(root/'results/submission_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(dict(pages=len(doc),tested_source_matches=True,protected_files_unchanged=True,
                     formal_checkpoint_unchanged=True,submission=manifest),indent=2))
