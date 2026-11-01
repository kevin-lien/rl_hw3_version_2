"""Package exactly the seven files required by the assignment."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile
import pymupdf

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'b13901072'
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
sources={
    'train_2048.py':ROOT/'2048/train_2048.py',
    'train_2048_env.py':ROOT/'2048/envs/train_2048_env.py',
    'model_2048.zip':ROOT/'2048/models/model_2048_weight1000.zip',
    'train_astro_party.py':ROOT/'astro_party/train_astro_party.py',
    'train_astro_party_env.py':ROOT/'astro_party/envs/train_astro_party_env.py',
    'model_astro_party.zip':ROOT/'astro_party/models/model_astro_party.zip',
    'report.pdf':ROOT/'report_work/latex/main.pdf',
}
for name,source in sources.items():
    (DEST/name).write_bytes(source.read_bytes())
    assert digest(DEST/name)==digest(source)
    if name.endswith('.py'):
        tree=ast.parse(source.read_text(encoding='utf-8'))
        imports=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):imports.extend(a.name.split('.')[0] for a in node.names)
            elif isinstance(node,ast.ImportFrom):imports.append((node.module or '').split('.')[0])
        assert not set(imports)&{'os','sys','shutil','pathlib','subprocess'},(name,imports)
assert {p.name for p in DEST.iterdir()}==set(sources)
assert '15/15' in (ROOT/'report_work/stock_2048.log').read_text()
assert '10/10' in (ROOT/'report_work/stock_astro_party.log').read_text()
doc=pymupdf.open(DEST/'report.pdf');assert len(doc)==7
pdf_output=ROOT/'output/pdf/Assignment_3_Report_b13901072.pdf'
pdf_output.write_bytes((DEST/'report.pdf').read_bytes())
edit=ROOT/'output/report_source';edit.mkdir(exist_ok=True)
(edit/'figures').mkdir(exist_ok=True)
for name in ['main.tex','student.sty']:(edit/name).write_bytes((ROOT/'report_work/latex'/name).read_bytes())
for p in (ROOT/'report_work/latex/figures').glob('*.png'):(edit/'figures'/p.name).write_bytes(p.read_bytes())
source_zip=ROOT/'output/Report_Source_b13901072.zip'
with zipfile.ZipFile(source_zip,'w',zipfile.ZIP_DEFLATED) as z:
    for p in edit.rglob('*'):
        if p.is_file():z.write(p,p.relative_to(edit).as_posix())
archive=ROOT/'b13901072.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for name in sorted(sources):z.write(DEST/name,'b13901072/'+name)
with zipfile.ZipFile(archive) as z:
    assert len(z.namelist())==7
    assert z.testzip() is None
manifest=dict(folder=str(DEST),archive=str(archive),report_pages=7,
    files=[dict(name=name,bytes=(DEST/name).stat().st_size,sha256=digest(DEST/name)) for name in sorted(sources)],
    verified_public_results={'2048':'15/15, max tile 512','astro_party':'10/10, Rusher 95%, Sniper 91%'},
    model_2048_source='2048/models/model_2048_weight1000.zip',
    note='Course report policy requires student review and own wording before submission.')
(ROOT/'report_work/results/submission_manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2))
