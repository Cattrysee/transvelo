import json, sys, time
from deep_translator import GoogleTranslator

t = GoogleTranslator(source='zh-CN', target='en')
runs = json.load(open("_runs.json", encoding="utf-8"))
a = int(sys.argv[1]); b = int(sys.argv[2]); idx = sys.argv[3]
out = {}
for s in runs[a:b]:
    try:
        out[s] = t.translate(s)
    except Exception:
        out[s] = s
    time.sleep(0.1)
json.dump(out, open(f"_tr_part{idx}.json", "w", encoding="utf-8"), ensure_ascii=False)
print(f"part{idx} done {a}-{b}", flush=True)
