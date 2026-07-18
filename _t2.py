from deep_translator import GoogleTranslator
t=GoogleTranslator(source="zh-CN", target="en")
print(t.translate("速度流场对比"))
print("OK")
