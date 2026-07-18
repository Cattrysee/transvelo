import json, sys, time, traceback
from deep_translator import GoogleTranslator

t = GoogleTranslator(source='zh-CN', target='en')
runs = json.load(open("_runs.json", encoding="utf-8"))
out = {}
log = open("_tr_progress.log", "w", encoding="utf-8")

for i, s in enumerate(runs):
    try:
        out[s] = t.translate(s)
    except Exception as e:
        out[s] = s
        log.write(f"ERR {i}: {e}\n")
    if (i + 1) % 5 == 0:
        log.write(f"progress {i+1}/{len(runs)}\n")
        log.flush()
    time.sleep(0.4)

json.dump(out, open("_tr.json", "w", encoding="utf-8"), ensure_ascii=False)
log.write("ALL DONE\n")
log.flush()
log.close()
