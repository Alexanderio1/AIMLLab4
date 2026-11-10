"""Execute the artifact-report notebook and export Markdown and standalone-image HTML."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
cache = ROOT / '.cache'
for name in ['ipython', 'jupyter', 'runtime', 'matplotlib']:
    (cache / name).mkdir(parents=True, exist_ok=True)
os.environ['IPYTHONDIR'] = str(cache / 'ipython')
os.environ['JUPYTER_DATA_DIR'] = str(cache / 'jupyter')
os.environ['JUPYTER_RUNTIME_DIR'] = str(cache / 'runtime')
os.environ['MPLCONFIGDIR'] = str(cache / 'matplotlib')

import json
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

spec = cache / 'jupyter/kernels/ccpp-report'
spec.mkdir(parents=True, exist_ok=True)
(spec / 'kernel.json').write_text(json.dumps({
    'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
    'display_name': 'CCPP artifact report', 'language': 'python',
}))
notebook = nbformat.read(ROOT / 'lab4.ipynb', as_version=4)
client = NotebookClient(notebook, timeout=180, kernel_name='ccpp-report', resources={'metadata': {'path': str(ROOT)}})
client.execute()
# The saved notebook opens with a standard Python kernel on another machine.
notebook.metadata['kernelspec'] = {'display_name': 'Python 3 (CCPP report)', 'language': 'python', 'name': 'python3'}
nbformat.write(notebook, ROOT / 'lab4.ipynb')

exporter = HTMLExporter(template_name='classic')
exporter.exclude_input_prompt = True
exporter.exclude_output_prompt = True
body, _ = exporter.from_notebook_node(notebook)
style = '''<style>
body { background: #f5f7fa; }
#notebook-container { max-width: 1280px; margin: 24px auto; padding: 32px 44px; box-shadow: 0 1px 10px #d9dfe7; }
.text_cell_render { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; font-size: 16px; line-height: 1.6; }
h1,h2,h3 { color: #162f4b; }
table { display: block; overflow-x: auto; max-width: 100%; font-size: 13px; }
th,td { padding: 8px 10px !important; vertical-align: top; }
.output_png img { max-width: 100%; height: auto; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; }
@media print { #notebook-container { box-shadow: none; margin: 0; padding: 0; } h3 { break-after: avoid; } img { max-height: 95vh; object-fit: contain; } }
</style>'''
body = body.replace('</head>', style + '\n</head>')
(ROOT / 'report.html').write_text('\n'.join(line.rstrip() for line in body.splitlines()) + '\n')

parts = []
for cell in notebook.cells:
    if cell.cell_type == 'markdown':
        parts.append(cell.source)
        continue
    parts.append('```python\n' + cell.source + '\n```')
    images = iter(cell.metadata.get('report_images', []))
    for output in cell.get('outputs', []):
        if output.output_type == 'stream':
            parts.append('```text\n' + output.text.rstrip() + '\n```')
        elif output.output_type in ['display_data', 'execute_result']:
            data = output.get('data', {})
            if 'image/png' in data:
                path = next(images)
                parts.append('![Иллюстрация отчёта](' + path + ')')
            elif 'text/markdown' in data:
                parts.append(data['text/markdown'])
            elif 'text/plain' in data:
                parts.append('```text\n' + data['text/plain'] + '\n```')
        elif output.output_type == 'error':
            raise RuntimeError('Notebook has an execution error: ' + output.get('evalue', ''))
(ROOT / 'report.md').write_text('\n\n'.join(parts) + '\n')
code_count = sum(cell.cell_type == 'code' for cell in notebook.cells)
print(f'Executed {code_count} code cells; wrote lab4.ipynb, report.md and report.html')
