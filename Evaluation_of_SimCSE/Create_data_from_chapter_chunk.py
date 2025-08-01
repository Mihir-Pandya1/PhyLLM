import os
import json
import re
import dotenv as ev
from tqdm import tqdm
import ast
from openai import OpenAI

# Load environment variables (assuming LLAMA_API is in your .env file)
ev.load_dotenv()
api_key = os.getenv("LLAMA_API")

# Initialize OpenAI client for the specified model
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)

model = "meta/llama-3.2-3b-instruct" # Your specified model

# Function to construct the prompt for the LLM
def put_input(content, topic_name):
    prompt = f"""
You are a Physics question generator for CBSE standard content.

Below is a section of Physics textbook content. Your task is to generate a **JSON array** of as many **distinct question-answer pairs** as possible based only on this content.

### Rules:
- Cover all possible **long**, **short**, and **numerical** question types.
- Each question must be **independent**, based on a **single idea** from the content.
- Do **not integrate** multiple facts or concepts into one question or answer.
- Do not repeat the same question in different forms.
- The questions should include **definitions**, **conceptual explanations**, **derivations**, **facts**, and **numerical calculations**, if any exist in the content.

### Format (output only this JSON array):
[
  {{
    "question": "<well-formed question>",
    "answer": "<precise and direct answer based on the content only>",
    "belonging": "{topic_name}"
  }},
  ...
]

Each dictionary in the array must be complete and well-formed. Do NOT include any malformed, partial, or syntactically incorrect entries. If a question cannot be fully formed, skip it.

### Content:
\"\"\"
{content}
\"\"\"

Generate the full list of valid and diverse question-answer pairs in the required JSON format. Do not add anything outside the array.
"""
    return prompt.strip()

# Define paths and collect file list
path = "/home/Group_1/chunking_daataset/chunking_of_1-6_chapter.json"
pattern = re.compile(r"\d+\.json$")
file_list = [file for file in os.listdir(path) if pattern.search(file)]
full_path_list = [f'{path}/{file}' for file in file_list]
print(full_path_list)

# Function to collect raw question-answer data from the LLM
def collect_raw_data(file_paths):
    raw_outputs = []

    for file_path in tqdm(file_paths, desc="Processing files"):
        chapter_name = os.path.basename(file_path).split('/')[-1].split('.')[0]

        with open(file_path, encoding="utf-8") as f:
            try:
                loaded_json_data = json.load(f)
            except:
                print("failed to load file")

            for item in loaded_json_data:
                if "topic" in item and "content" in item:
                    topic_name = item["topic"]
                    section_content = item["content"]
                else:
                    print(f"Warning: Item in {file_path} is not in expected {{'topic': ..., 'content': ...}} format. Skipping item.")
                    continue
                
                # Now use topic_name and section_content for the prompt
                prompt = put_input(section_content, topic_name)

                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        top_p=0.8,
                        max_tokens=1028,
                        stream=True
                    )

                    collected = ""
                    for chunk in response:
                        collected += chunk.choices[0].delta.content or ""

                    collected_clean = collected.strip()
                    collected_clean = re.sub(r"^[^\[]*\[", "[", collected_clean, flags=re.DOTALL)
                    collected_clean = re.sub(r"\][^\]]*$", "]", collected_clean, flags=re.DOTALL)

                    recovered_qas = []
                    try:
                        parsed_array = json.loads(collected_clean)
                        if isinstance(parsed_array, list):
                            for qa_item in parsed_array:
                                if isinstance(qa_item, dict) and "question" in qa_item and "answer" in qa_item:
                                    recovered_qas.append(qa_item)
                                else:
                                    print(f"Warning: Malformed Q&A item in LLM output: {qa_item}")
                        else:
                            print(f"Warning: LLM did not return a JSON array: {collected_clean[:100]}...")
                    except :
                        print(f"Can not loads as json")
                        dict_chunks = re.findall(r"{[^}]*}", collected_clean, re.DOTALL)
                        for chunk in dict_chunks:
                            try:
                                qa = json.loads(chunk)
                                if isinstance(qa, dict) and "question" in qa and "answer" in qa:
                                    recovered_qas.append(qa)
                                else:
                                    print(f"Warning: Malformed Q&A chunk (not dict or missing keys): {chunk}")
                            except json.JSONDecodeError as e_json_chunk:
                                try:
                                    qa = ast.literal_eval(chunk)
                                    if isinstance(qa, dict) and "question" in qa and "answer" in qa:
                                        recovered_qas.append(qa)
                                    else:
                                        print(f"Warning: Malformed Q&A chunk (not dict or missing keys) after ast.literal_eval: {chunk}")
                                except Exception as e_ast:
                                    print(f"Warning: Could not parse chunk with json or ast: {chunk[:50]}... Error: {e_ast}")
                                    continue

                    raw_outputs.extend(recovered_qas)
                    raw_outputs.append({
                        "question": f"Explain the concept of {topic_name}",
                        "answer": section_content,
                        "type": "long",
                        "belonging": topic_name
                    })
                    print(raw_outputs)

                except Exception as e:
                    print(f"Error during LLM API call or processing for topic '{topic_name}': {e}. Skipping this topic.")
                    continue
                
    output_filename = "evaluation_based_data.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(raw_outputs, f, indent=2, ensure_ascii=False)
    print(f"\nGenerated {len(raw_outputs)} question-answer pairs. Saved to {output_filename}")

# Run the data collection process
collect_raw_data(full_path_list)