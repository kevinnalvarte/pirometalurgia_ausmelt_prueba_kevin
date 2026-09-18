"""Comprobaciones estructurales y de trazabilidad del informe generado."""
from html.parser import HTMLParser
from pathlib import Path
from collections import Counter
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent
html = (ROOT / 'MODELO_GANADOR_EXPLICACION_PIROMETALURGICA.html').read_text(encoding='utf-8')

class Audit(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.ids = []
        self.hrefs = []
        self.stack = []
        self.errors = []
        self.text = []
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if 'id' in a: self.ids.append(a['id'])
        if 'href' in a: self.hrefs.append(a['href'])
        if tag not in {'meta','link','br','hr','input','img','source','wbr','area','base','embed','param','track','col'}:
            self.stack.append(tag)
    def handle_endtag(self, tag):
        if not self.stack or self.stack[-1] != tag:
            self.errors.append((tag, list(self.stack[-4:])))
        else: self.stack.pop()
    def handle_data(self, data):
        if 'script' not in self.stack and 'style' not in self.stack:
            self.text.append(data)

a = Audit()
a.feed(html)
assert not a.errors and not a.stack, (a.errors, a.stack)
assert all(n == 1 for n in Counter(a.ids).values()), 'IDs duplicados'
for href in a.hrefs:
    if href.startswith('#'):
        assert href[1:] in a.ids, href
    elif not href.startswith(('https://','http://')):
        assert (ROOT / href).exists(), href
assert '\ufffd' not in html
assert html.count('<section ') == 14
assert abs((-0.0058)*(-2)*10000 - 116) < 1e-10
node = shutil.which('node')
if node:
    script = re.search(r'<script>(.*?)</script>', html, re.S).group(1)
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp)/'informe.js'
        path.write_text(script, encoding='utf-8')
        subprocess.run([node,'--check',str(path)],check=True,capture_output=True,text=True)
print(f'OK: 14 secciones, {len(a.ids)} IDs únicos, {len(a.hrefs)} enlaces revisados, HTML balanceado y UTF-8 válido.')
print(f'Palabras visibles aproximadas: {len(" ".join(a.text).split())}. JavaScript: {"sintaxis verificada" if node else "no se encontró Node para verificación de sintaxis"}.')
