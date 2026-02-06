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

EMBEDDING_MODEL = 'models/gemini-embedding-001'
RAG_DATA_FILE = os.path.join(os.path.dirname(__file__), "../../rag_data.json")
ENV_FILE = os.path.join(os.path.dirname(__file__), "../../.env")

def _format_value(value, indent=0) -> str:
    """Recursively format a value into readable text."""
    prefix = "  " * indent

    if isinstance(value, str):
        return value
    elif isinstance(value, list):
        if all(isinstance(item, str) for item in value):
            return ", ".join(value)
        else:
            lines = []
            for item in value:
                if isinstance(item, dict):
                    lines.append(f"{prefix}- {_format_dict(item, indent + 1)}")
                else:
                    lines.append(f"{prefix}- {item}")
            return "\n".join(lines)
    elif isinstance(value, dict):
        return _format_dict(value, indent)
    else:
        return str(value)


def _format_dict(d: dict, indent=0) -> str:
    """Format a dictionary into readable text."""
    lines = []
    for key, value in d.items():
        formatted_key = key.replace("_", " ").title()
        formatted_value = _format_value(value, indent)

        if "\n" in formatted_value:
            lines.append(f"{formatted_key}:\n{formatted_value}")
        else:
            lines.append(f"{formatted_key}: {formatted_value}")

    return "\n".join(lines)


def load_and_chunk_json(file_path: str) -> list[dict]:
    """
    Load JSON data and dynamically create semantic chunks for vector storage.

    Automatically handles any JSON structure:
        - Dict values: Single chunk per key
        - List values: One chunk per item in the list

    Args:
        file_path: Path to the rag_data.json file

    Returns:
        List of chunks with 'id', 'type', and 'text' keys
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    chunks = []

    for key, value in data.items():
        chunk_type = key.replace("_", " ").lower()

        if isinstance(value, list):
            # Each item in list becomes a separate chunk
            for item in value:
                if isinstance(item, dict):
                    # Try to create a meaningful title
                    title_keys = ['name', 'title', 'role', 'repository']
                    title = next((item[k] for k in title_keys if k in item), chunk_type)
                    text = f"{chunk_type.title()}: {title}\n\n{_format_dict(item)}"
                else:
                    text = f"{chunk_type.title()}: {item}"
                chunks.append({"type": chunk_type, "text": text.strip()})

        elif isinstance(value, dict):
            # Single chunk for dict
            text = f"{chunk_type.title()}\n\n{_format_dict(value)}"
            chunks.append({"type": chunk_type, "text": text.strip()})

        else:
            # Simple value
            text = f"{chunk_type.title()}: {value}"
            chunks.append({"type": chunk_type, "text": text.strip()})

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
            task_type="retrieval_document",
            output_dimensionality=768
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


def test_chunks():
    """
    Test function to display all chunks converted from JSON.
    Run with: python api/utils/upload_vectorstore_data.py --test
    """
    print("=" * 60)
    print("TESTING: JSON to Chunks Conversion")
    print("=" * 60)

    chunks = load_and_chunk_json(RAG_DATA_FILE)

    print(f"\nTotal chunks created: {len(chunks)}\n")

    for i, chunk in enumerate(chunks, 1):
        print(f"{'─' * 60}")
        print(f"CHUNK {i} | Type: {chunk['type']} | ID: {chunk['id'][:8]}...")
        print(f"{'─' * 60}")
        print(chunk['text'])
        print()

    print("=" * 60)
    print(f"Summary: {len(chunks)} chunks ready for embedding")
    print("=" * 60)


if __name__ == "__main__":
    import sys

    if "--test" in sys.argv:
        test_chunks()
    elif "--workflow" not in sys.argv:
        from dotenv import load_dotenv

        load_dotenv(ENV_FILE)
        main()
    else:
        main()
