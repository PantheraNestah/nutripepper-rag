import os
import re
import json
import fitz
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from source_metadata import SOURCE_METADATA

RAW_DIR = "data/raw"
OUTPUT_DIR = "data/knowledge_base"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""

    for page in doc:
        full_text += page.get_text("text") + "\n"

    return full_text


def clean_text(text):
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'Page \d+', ' ', text)
    text = re.sub(r'\b\d+\b', ' ', text)
    text = re.sub(r'-\s+', '', text)
    text = re.sub(r'\.{2,}', ' ', text)
    return text.strip()


def chunk_text(text, min_words=100, max_words=220):
    try:
        sentences = sent_tokenize(text)
    except:
        sentences = text.split('. ')

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence = sentence.strip()

        if len(sentence.split()) < 5:
            continue

        sentence_len = len(sentence.split())

        if current_length + sentence_len <= max_words:
            current_chunk.append(sentence)
            current_length += sentence_len
        else:
            if current_length >= min_words:
                chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_len

    if current_chunk and current_length >= min_words:
        chunks.append(" ".join(current_chunk))

    return chunks


def process_documents():
    all_knowledge_chunks = []

    for filename in tqdm(os.listdir(RAW_DIR)):
        if not filename.endswith(".pdf"):
            continue

        print(f"\nProcessing {filename}...")

        pdf_path = os.path.join(RAW_DIR, filename)

        raw_text = extract_pdf_text(pdf_path)
        cleaned_text = clean_text(raw_text)
        chunks = chunk_text(cleaned_text)

        metadata = SOURCE_METADATA.get(filename, {})

        print(f"Generated {len(chunks)} chunks from {filename}")

        for idx, chunk in enumerate(chunks):
            knowledge_object = {
                "chunk_id": f"{filename.replace('.pdf', '')}_{idx}",
                "source_document": filename,
                "text": chunk,
                "authority": metadata.get("authority"),
                "year": metadata.get("year"),
                "domain": metadata.get("domain"),
                "kenya_relevance": metadata.get("kenya_relevance")
            }

            all_knowledge_chunks.append(knowledge_object)

    with open(os.path.join(OUTPUT_DIR, "nutrition_knowledge_base.json"), "w") as f:
        json.dump(all_knowledge_chunks, f, indent=4)

    print(f"\nTOTAL KNOWLEDGE CHUNKS GENERATED: {len(all_knowledge_chunks)}")


if __name__ == "__main__":
    process_documents()