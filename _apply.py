import json, glob, re

CJK = re.compile(r'[\u4e00-\u9fff]+')
tr = json.load(open("_tr.json", encoding="utf-8"))

def repl(s):
    out = []
    last = 0
    for m in CJK.finditer(s):
        out.append(s[last:m.start()])
        t = tr.get(m.group(), m.group())
        out.append(t)
        last = m.end()
    out.append(s[last:])
    return "".join(out)

for f in sorted(glob.glob("notebooks/*.ipynb")):
    nb = json.load(open(f, encoding="utf-8"))
    for c in nb.get("cells", []):
        src = c.get("source", "")
        if isinstance(src, list):
            c["source"] = [repl(line) for line in src]
        else:
            c["source"] = repl(src)
    json.dump(nb, open(f, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# verify
left = 0
for f in sorted(glob.glob("notebooks/*.ipynb")):
    nb = json.load(open(f, encoding="utf-8"))
    for c in nb.get("cells", []):
        src = c.get("source", "")
        if isinstance(src, list): src = "".join(src)
        if CJK.search(src): left += 1
print("cells still containing CJK:", left)
print("APPLY DONE")
