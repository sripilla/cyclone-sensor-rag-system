import json

with open("outputs/document_chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)

hits = [c for c in chunks if "draft" in c["text"].lower()]
print(f"{len(hits)} chunks mention draft")

for c in hits[:10]:
    snippet = c["text"][:150].replace("\n", " ")
    print(f"[{c['document']} p.{c['page']}] {snippet}...")
