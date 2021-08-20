from readyml import texttranslation as rtt

model = rtt.Translation_EnglishToFrench()
text = model.infer("Hello world!")
print(f"English To French: Hello world! -> {text}")

model = rtt.Translation_EnglishToGerman()
text = model.infer("Hello world!")
print(f"English To German: Hello world! -> {text}")

model = rtt.Translation_GermanToEnglish()
text = model.infer("Guten Tag!")
print(f"German To English: Guten Tag! -> {text}")

model = rtt.Translation_EnglishToRussian()
text = model.infer("Hello world!")
print(f"English To Russian: Hello world! -> {text}")

model = rtt.Translation_RussianToEnglish()
text = model.infer("Hello world!")
print(f"Russian To English: Guten Tag! -> {text}")
