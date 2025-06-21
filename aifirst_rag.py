import streamlit as st
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
import pandas as pd
import docx
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
import uuid

# --- PAGE CONFIG ---
st.set_page_config(page_title="AIFirst RAG Assistant", page_icon="📚", layout="wide")
st.title("🔍 AIFirst RAG Assistant")
st.markdown("Upload PDFs, DOCX, Excel, TXT, and more — then ask natural language questions.")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.title("🔐 Configuration")

# Input fields
openai_key_input = st.sidebar.text_input("OpenAI API Key", type="password")
qdrant_api_input = st.sidebar.text_input("Qdrant API Key", type="password")
qdrant_url_input = st.sidebar.text_input(
    "Qdrant URL (e.g., https://yourhost.cloud:6333)",
    value="https://6a7820c2-43e6-45f7-bd2e-6e1f73bc6906.eu-central-1-0.aws.cloud.qdrant.io:6333"
)

# Session state defaults
for key in ["openai_valid", "qdrant_valid", "qdrant_client", "openai_client"]:
    if key not in st.session_state:
        st.session_state[key] = None

# Validate on button click
if st.sidebar.button("🔄 Connect & Validate"):
    # OpenAI validation
    try:
        openai_client = OpenAI(api_key=openai_key_input)
        openai_client.models.list()
        st.session_state["openai_valid"] = True
        st.session_state["openai_client"] = openai_client
    except Exception as e:
        st.session_state["openai_valid"] = False
        st.sidebar.error(f"❌ OpenAI key error: {e}")

    # Qdrant validation
    try:
        qdrant_client = QdrantClient(url=qdrant_url_input, api_key=qdrant_api_input)
        qdrant_client.get_collections()
        st.session_state["qdrant_valid"] = True
        st.session_state["qdrant_client"] = qdrant_client
    except Exception as e:
        st.session_state["qdrant_valid"] = False
        st.sidebar.error(f"❌ Qdrant error: {e}")

# Show validation results
if st.session_state["openai_valid"] is True:
    st.sidebar.success("✅ OpenAI API key is valid!")
elif st.session_state["openai_valid"] is False:
    st.sidebar.error("❌ Invalid OpenAI API key")

if st.session_state["qdrant_valid"] is True:
    st.sidebar.success("✅ Qdrant connected!")
elif st.session_state["qdrant_valid"] is False:
    st.sidebar.error("❌ Qdrant not connected")

# --- HALT IF NOT VALIDATED ---
if not st.session_state.get("openai_valid"):
    st.warning("Please validate your OpenAI API key to continue.")
    st.stop()

if not st.session_state.get("qdrant_valid"):
    st.warning("Please validate your Qdrant credentials to continue.")
    st.stop()

client = st.session_state["openai_client"]
qdrant = st.session_state["qdrant_client"]
COLLECTION_NAME = "rag_demo"

# --- INIT COLLECTION IF NEEDED ---
existing = qdrant.get_collections().collections
if not any(c.name == COLLECTION_NAME for c in existing):
    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

# --- EMBEDDING MODEL ---
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# --- FILE EXTRACTOR ---
def extract_text_from_file(uploaded_file, file_type):
    try:
        if file_type == "txt":
            return uploaded_file.read().decode("utf-8")

        elif file_type == "pdf":
            with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
                return "\n\n".join([page.get_text() for page in doc])

        elif file_type == "docx":
            doc = docx.Document(uploaded_file)
            return "\n\n".join([p.text for p in doc.paragraphs if p.text.strip()])

        elif file_type in ["xlsx", "xls"]:
            df = pd.read_excel(uploaded_file)
            return df.to_string(index=False)

        elif file_type == "csv":
            df = pd.read_csv(uploaded_file)
            return df.to_string(index=False)

        elif file_type == "html":
            soup = BeautifulSoup(uploaded_file.read(), "html.parser")
            return soup.get_text()

        else:
            return ""
    except Exception as e:
        st.error(f"❌ Failed to extract text: {e}")
        return ""

# --- UPLOAD + EMBED ---
uploaded_file = st.file_uploader("📄 Upload a document (PDF, DOCX, Excel, CSV, TXT, HTML)", 
                                  type=["txt", "pdf", "docx", "xlsx", "xls", "csv", "html"])

if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1].lower()
    text = extract_text_from_file(uploaded_file, file_type)

    if not text:
        st.warning("⚠️ No extractable text found.")
    else:
        chunks = [chunk.strip() for chunk in text.split("\n\n") if len(chunk.strip()) > 30]
        with st.spinner("🔎 Embedding and storing text chunks..."):
            vectors = embedder.encode(chunks).tolist()
            points = [
                PointStruct(id=str(uuid.uuid4()), vector=vec, payload={"text": chunk})
                for vec, chunk in zip(vectors, chunks)
            ]
            qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        st.success(f"✅ {len(chunks)} chunks embedded into Qdrant!")

# --- QUERY ---
st.header("🧠 Ask a Question")
user_query = st.text_input("Enter your question:")

if st.button("Get RAG Answer", disabled=not user_query.strip()):
    try:
        query_vec = embedder.encode([user_query])[0]
        results = qdrant.search(collection_name=COLLECTION_NAME, query_vector=query_vec, limit=5)
        retrieved_chunks = [hit.payload['text'] for hit in results]

        context = "\n\n".join(retrieved_chunks)

        st.subheader("📚 Retrieved Context")
        st.write(context)

        prompt = f"Answer the question based on the following context:\n\n{context}\n\nQuestion: {user_query}"
        with st.spinner("🤖 Thinking..."):
            response = client.chat.completions.create(
                model="gpt-4o",  # 👈 GPT-4o used here
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            answer = response.choices[0].message.content.strip()

        st.subheader("💬 RAG Answer")
        st.write(answer)

    except Exception as e:
        st.error(f"❌ Error during RAG answering: {e}")

import streamlit as st

# Streamlit file uploader
uploaded_file = st.file_uploader("Choose a file", type="txt")

if uploaded_file is not None:
    chunks = []
    current_chunk = ""
    
    # Read the uploaded file content
    text = uploaded_file.read().decode('utf-8')
    
    # Split the text by participant markers
    participant_sections = text.split("PARTICIPANT_")[1:]  # Skip first empty element
    
    for i, section in enumerate(participant_sections, 1):
        # Check if this is engagement data section
        if section.strip().startswith("=== ENGAGEMENT TRACKING DATA ==="):
            # Handle engagement data separately
            engagement_lines = section.split('\n')
            current_participant = None
            
            for line in engagement_lines:
                if line.strip().startswith("PARTICIPANT_") and "Engagement History:" in line:
                    # Save previous engagement chunk if exists
                    if current_chunk:
                        chunks.append({
                            "content": current_chunk.strip(),
                            "metadata": {
                                "chunk_type": "engagement_data",
                                "participant_id": current_participant
                            }
                        })
                    
                    # Start new engagement chunk
                    current_participant = line.split()[0]
                    current_chunk = line + "\n"
                elif current_participant and line.strip():
                    current_chunk += line + "\n"
            
            # Add final engagement chunk
            if current_chunk:
                chunks.append({
                    "content": current_chunk.strip(),
                    "metadata": {
                        "chunk_type": "engagement_data",
                        "participant_id": current_participant
                    }
                })
        else:
            # Handle regular participant profile
            participant_content = "PARTICIPANT_" + section
            
            # Extract metadata from the content
            lines = section.split('\n')
            name = ""
            location = ""
            occupation = ""
            
            for line in lines:
                if line.startswith("Name:"):
                    name = line.replace("Name:", "").strip()
                elif line.startswith("Location:"):
                    location = line.replace("Location:", "").strip()
                elif line.startswith("Occupation:"):
                    occupation = line.replace("Occupation:", "").strip()
            
            chunks.append({
                "content": participant_content.strip(),
                "metadata": {
                    "participant_id": f"PARTICIPANT_{i:03d}",
                    "participant_name": name,
                    "location": location,
                    "occupation": occupation,
                    "chunk_type": "participant_profile"
                }
            })
    
    # Reset for any remaining content
    current_chunk = ""
    
    st.success(f"Created {len(chunks)} chunks")
    
    # Optional: Display some chunks for verification
    if st.checkbox("Show sample chunks"):
        for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
            st.write(f"**Chunk {i+1}:**")
            st.write(chunk["content"][:200] + "..." if len(chunk["content"]) > 200 else chunk["content"])
            st.write("Metadata:", chunk["metadata"])
            st.write("---")
