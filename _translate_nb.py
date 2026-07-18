import json, glob, re, sys, traceback
from deep_translator import GoogleTranslator

CJK = re.compile(r'[\u4e00-\u9fff]+')
trans = GoogleTranslator(source='zh-CN', target='en')
cache = {}

def tr(text):
    text = text.strip()
    if not text:
        return text
    if text in cache:
        return cache[text]
    try:
        out = trans.translate(text)
    except Exception:
        out = text
    if not out:
        out = text
    cache[text] = out
    return out

def translate_runs(s):
    parts = []
    last = 0
    for m in CJK.finditer(s):
        parts.append(s[last:m.start()])
        parts.append(tr(m.group()))
        last = m.end()
    parts.append(s[last:])
    return "".join(parts)

files = sorted(glob.glob("notebooks/*.ipynb"))
for f in files:
    nb = json.load(open(f, encoding="utf-8"))
    for cell in nb.get("cells", []):
        src = cell.get("source", "")
        if isinstance(src, list):
            cell["source"] = [translate_runs(line) for line in src]
        else:
            cell["source"] = translate_runs(src)
    json.dump(nb, open(f, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    rem = sum(1 for c in nb["cells"]
              if CJK.search("".join(c["source"]) if isinstance(c["source"], list) else c["source"]))
    print(f"{f}: done (remaining CJK cells: {rem})", flush=True)

print("ALL DONE", flush=True)
