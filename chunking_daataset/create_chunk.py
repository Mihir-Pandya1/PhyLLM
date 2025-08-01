import os
import re
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter

file_path = "/home/Group_1/data/"
pattern = re.compile(r"chapter_\d+\.json$")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1800,
    chunk_overlap=250
)

full_paths = [
    os.path.join(file_path, file)
    for file in os.listdir(file_path)
    if pattern.match(file)
]

print(full_paths)

final_chunks = []

for path in full_paths:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for entry in data:
            if 'content' in entry and 'topic' in entry:
                chunks = splitter.split_text(entry['content'])
                for chunk in chunks:
                    final_chunks.append({
                        "content": chunk,
                        "topic": entry['topic']
                    })

output_path = "chunking_of_1-6_chapter.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(final_chunks, f, indent=2, ensure_ascii=False)

print(f"? {len(final_chunks)} topic-tagged chunks saved to {output_path}")
