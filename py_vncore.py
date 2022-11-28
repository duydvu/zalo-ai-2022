import py_vncorenlp

# py_vncorenlp.download_model()
model = py_vncorenlp.VnCoreNLP(annotators=["wseg", "parse"])
print(model.annotate_text("Alo 1234. abc"))
