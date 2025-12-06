"""
Upload embeddings to Upstash Vector from structured JSON data.

Prerequisites:
1. Create a free Upstash account at https://upstash.com
2. Create a Vector index with:
   - Dimensions: 768 (for text-embedding-004)
   - Distance Metric: Cosine
3. Set environment variables:
   - UPSTASH_VECTOR_REST_URL
   - UPSTASH_VECTOR_REST_TOKEN
   - API_TOKEN (Google Gemini API key)

Usage:
    python api/utils/upload_vectorstore_data.py           # Local (loads .env)
    python api/utils/upload_vectorstore_data.py --workflow  # CI/CD (uses env vars)
"""

import hashlib
import json
import os

import google.generativeai as genai
from upstash_vector import Index

EMBEDDING_MODEL = "models/text-embedding-004"
RAG_DATA_FILE = "../../rag_data.json"


def load_and_chunk_json(file_path: str) -> list[dict]:
    """
    Load JSON data and create semantic chunks for vector storage.

    Each section/item in the JSON becomes a separate chunk to maintain
    semantic coherence. This prevents splitting related information
    across multiple chunks.

    Chunk types created:
        - about: Single chunk with bio and highlights
        - skills: Single chunk with all skills categorized
        - experience: One chunk per job
        - startup_experience: One chunk per startup role
        - project: One chunk per project
        - volunteering: One chunk per volunteer role
        - publication: One chunk per publication
        - awards: Single chunk with all awards
        - links: Single chunk with contact info

    Args:
        file_path: Path to the rag_data.json file

    Returns:
        List of chunks with 'id', 'type', and 'text' keys
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    chunks = []

    # About section - single chunk
    if 'about' in data:
        about = data['about']
        text = f"""About {about['name']}
{about['title']}

{about['summary']}

Key highlights:
{chr(10).join('- ' + h for h in about.get('highlights', []))}"""
        chunks.append({"type": "about", "text": text.strip()})

    # Skills - single chunk
    if 'skills' in data:
        skills = data['skills']
        text = f"""Skills of Yashas Majmudar

Domains: {', '.join(skills.get('domains', []))}

Programming Languages: {', '.join(skills.get('languages', []))}

Frameworks & Libraries: {', '.join(skills.get('frameworks', []))}

Databases: {', '.join(skills.get('databases', []))}

Tools & Platforms: {', '.join(skills.get('tools', []))}"""
        chunks.append({"type": "skills", "text": text.strip()})

    # Experience - each job is a chunk
    for exp in data.get('experience', []):
        text = f"""Work Experience: {exp['role']} at {exp['company']}
Duration: {exp['duration']}
Type: {exp.get('type', 'full-time')}

Achievements:
{chr(10).join('- ' + h for h in exp.get('highlights', []))}"""
        chunks.append({"type": "experience", "text": text.strip()})

    # Startup Experience - each is a chunk
    for exp in data.get('startup_experience', []):
        text = f"""Startup Experience: {exp['role']} at {exp['company']}
Duration: {exp['duration']}

Achievements:
{chr(10).join('- ' + h for h in exp.get('highlights', []))}"""
        chunks.append({"type": "startup_experience", "text": text.strip()})

    # Projects - each project is a chunk
    for proj in data.get('projects', []):
        text = f"""Project: {proj['name']}

{proj['description']}

Technologies: {', '.join(proj.get('skills', []))}
Link: {proj.get('link', 'N/A')}"""
        chunks.append({"type": "project", "text": text.strip()})

    # Volunteering - each is a chunk
    for vol in data.get('volunteering', []):
        highlights = vol.get('highlights', [])
        highlights_text = f"\n\nHighlights:\n{chr(10).join('- ' + h for h in highlights)}" if highlights else ""
        text = f"""Volunteering: {vol['role']} at {vol['organization']}
Duration: {vol['duration']}{highlights_text}"""
        chunks.append({"type": "volunteering", "text": text.strip()})

    # Publications - each is a chunk
    for pub in data.get('publications', []):
        text = f"""Publication: {pub['title']}
Published in: {pub['journal']}
Date: {pub['date']}

Key findings:
{chr(10).join('- ' + h for h in pub.get('highlights', []))}"""
        chunks.append({"type": "publication", "text": text.strip()})

    # Awards - all in one chunk (they're short)
    if 'awards' in data:
        awards_text = []
        for award in data['awards']:
            note = f" - {award['note']}" if 'note' in award else ""
            project = f" for {award['project']}" if 'project' in award else ""
            awards_text.append(f"- {award['name']}{project}{note}")
        text = f"""Awards and Recognition of Yashas Majmudar

{chr(10).join(awards_text)}"""
        chunks.append({"type": "awards", "text": text.strip()})

    # Links - single chunk
    if 'links' in data:
        links = data['links']
        text = f"""Contact and Links for Yashas Majmudar

Website: {links.get('website', 'N/A')}
LinkedIn: {links.get('linkedin', 'N/A')}
GitHub: {links.get('github', 'N/A')}
Blog: {links.get('blog', 'N/A')}"""
        chunks.append({"type": "links", "text": text.strip()})

    # Add ID to each chunk
    for chunk in chunks:
        chunk["id"] = hashlib.md5(chunk["text"].encode()).hexdigest()

    return chunks


def get_embeddings(texts: list[str], api_key: str) -> list[list[float]]:
    """
    Generate embeddings for text chunks using Google's embedding API.

    Args:
        texts: List of text strings to embed
        api_key: Google Gemini API key

    Returns:
        List of 768-dimensional embedding vectors
    """
    genai.configure(api_key=api_key)

    embeddings = []
    for text in texts:
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            task_type="retrieval_document"
        )
        embeddings.append(result['embedding'])

    return embeddings


def upload_vector_data(chunks: list[dict], embeddings: list[list[float]]):
    """
    Upload vectors to Upstash Vector index.

    This function clears the existing index before uploading new data
    to ensure a clean state. Vectors are uploaded in batches of 100.

    Args:
        chunks: List of chunks with 'id', 'type', and 'text' keys
        embeddings: List of embedding vectors corresponding to chunks
    """
    index = Index(
        url=os.environ["UPSTASH_VECTOR_REST_URL"],
        token=os.environ["UPSTASH_VECTOR_REST_TOKEN"]
    )

    # Clear existing data first
    print("Clearing existing vectors...")
    index.reset()
    print("Datastore cleared")

    # Prepare vectors with metadata
    vectors = []
    for chunk, embedding in zip(chunks, embeddings):
        vectors.append({
            "id": chunk["id"],
            "vector": embedding,
            "metadata": {
                "text": chunk["text"],
                "type": chunk["type"]
            }
        })

    # Upsert in batches of 100
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)
        print(f"Uploaded batch {i // batch_size + 1}/{(len(vectors) - 1) // batch_size + 1}")

    print(f"Successfully uploaded {len(vectors)} vectors to Upstash")


def main():
    # Validate environment variables
    required_vars = ["UPSTASH_VECTOR_REST_URL", "UPSTASH_VECTOR_REST_TOKEN", "API_TOKEN"]
    missing = [v for v in required_vars if not os.environ.get(v)]
    if missing:
        print(f"Error: Missing environment variables: {', '.join(missing)}")
        print("\nSet them with:")
        for v in missing:
            print(f"  export {v}=your_value")
        return

    print("Loading and chunking JSON data...")
    chunks = load_and_chunk_json(RAG_DATA_FILE)
    print(f"Created {len(chunks)} chunks:")
    for chunk in chunks:
        print(f"  - {chunk['type']}: {len(chunk['text'])} chars")

    print("\nGenerating embeddings via Google API...")
    texts = [c["text"] for c in chunks]
    embeddings = get_embeddings(texts, os.environ["API_TOKEN"])
    print(f"Generated {len(embeddings)} embeddings")

    print("\nUploading to Upstash Vector...")
    upload_vector_data(chunks, embeddings)

    print("\nDone! Your RAG data is now in Upstash Vector.")


if __name__ == "__main__":
    import sys

    if "--workflow" not in sys.argv:
        from dotenv import load_dotenv

        load_dotenv('../../.env')
    main()
