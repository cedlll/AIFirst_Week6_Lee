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
import re

# --- PAGE CONFIG ---
st.set_page_config(page_title="Participant Finder", page_icon="🔍", layout="wide")
st.title("🔍 Participant Matcher")

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
for key in ["openai_valid", "qdrant_valid", "qdrant_client", "openai_client", "data_loaded"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "data_loaded" else False

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
try:
    existing = qdrant.get_collections().collections
    if not any(c.name == COLLECTION_NAME for c in existing):
        qdrant.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        st.session_state["data_loaded"] = False
except Exception as e:
    st.error(f"Error initializing collection: {e}")

# --- EMBEDDING MODEL ---
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# --- HELPER FUNCTIONS ---
def extract_participant_info(content):
    """Extract structured information from participant content"""
    info = {}
    
    # Extract name
    name_match = re.search(r'##\s*Participant\s*\d+:\s*([^\n]+)', content)
    if name_match:
        info['name'] = name_match.group(1).strip()
    
    # Extract response rate
    response_rate_match = re.search(r'Response rate:\s*(\d+)%', content)
    if response_rate_match:
        info['response_rate'] = int(response_rate_match.group(1))
    
    # Extract other fields
    fields = ['Age', 'Gender', 'Location', 'Studies participated', 'Preferred communication', 
              'Best response times', 'No-show rate', 'Specialties', 'Education', 'Availability']
    
    for field in fields:
        pattern = rf'\*\*{field}:\*\*\s*([^\n]+)'
        match = re.search(pattern, content)
        if match:
            value = match.group(1).strip()
            if field == 'No-show rate':
                # Extract percentage
                no_show_match = re.search(r'(\d+)%', value)
                if no_show_match:
                    info['no_show_rate'] = int(no_show_match.group(1))
            elif field == 'Studies participated':
                # Extract number
                studies_match = re.search(r'(\d+)', value)
                if studies_match:
                    info['studies_participated'] = int(studies_match.group(1))
            elif field == 'Age':
                # Extract age
                age_match = re.search(r'(\d+)', value)
                if age_match:
                    info['age'] = int(age_match.group(1))
            else:
                info[field.lower().replace(' ', '_')] = value
    
    return info

def parse_markdown_participants(text):
    """Parse markdown format participant data"""
    chunks = []
    
    # Split by participant sections
    sections = re.split(r'##\s*Participant\s*\d+:', text)
    
    for i, section in enumerate(sections[1:], 1):  # Skip first empty section
        if section.strip():
            # Reconstruct the participant section
            participant_content = f"## Participant {i:03d}:" + section
            
            # Extract structured info
            info = extract_participant_info(participant_content)
            
            chunk = {
                "content": participant_content.strip(),
                "metadata": {
                    "participant_id": f"PARTICIPANT_{i:03d}",
                    "chunk_type": "participant_profile",
                    **info
                }
            }
            chunks.append(chunk)
    
    return chunks

def parse_text_participants(text):
    """Parse text format with PARTICIPANT_ markers"""
    chunks = []
    
    # Split the text by participant markers
    participant_sections = text.split("PARTICIPANT_")[1:]  # Skip first empty element
    
    for i, section in enumerate(participant_sections, 1):
        # Check if this is engagement data section
        if section.strip().startswith("=== ENGAGEMENT TRACKING DATA ==="):
            # Handle engagement data separately
            engagement_lines = section.split('\n')
            current_participant = None
            current_chunk = ""
            
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
    
    return chunks

# --- FILE UPLOAD AND PROCESSING ---
st.header("📁 Upload Participant Details")
uploaded_file = st.file_uploader("Choose a file", type=["txt", "md"])

if uploaded_file is not None:
    with st.spinner("🔄 Processing file..."):
        try:
            # Read the uploaded file content
            text = uploaded_file.read().decode('utf-8')
            
            # Determine file format and parse accordingly
            if "## Participant" in text:
                # Markdown format
                chunks = parse_markdown_participants(text)
                st.info("📝 Detected markdown format")
            elif "PARTICIPANT_" in text:
                # Text format with PARTICIPANT_ markers
                chunks = parse_text_participants(text)
                st.info("📄 Detected text format with PARTICIPANT_ markers")
            else:
                # Generic text processing
                chunks = []
                sentences = text.split('.')
                for i, sentence in enumerate(sentences):
                    if sentence.strip():
                        chunks.append({
                            "content": sentence.strip(),
                            "metadata": {
                                "chunk_id": f"chunk_{i:03d}",
                                "chunk_type": "generic_text"
                            }
                        })
                st.info("📋 Processed as generic text")
            
            st.success(f"✅ Created {len(chunks)} chunks")
            
            # Optional: Display some chunks for verification
            if st.checkbox("Show sample chunks"):
                for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
                    st.write(f"**Chunk {i+1}:**")
                    st.write(chunk["content"][:300] + "..." if len(chunk["content"]) > 300 else chunk["content"])
                    st.write("**Metadata:**", chunk["metadata"])
                    st.write("---")
            
            # Automatically process and store embeddings
            if chunks:
                with st.spinner("🔄 Automatically processing embeddings and storing in vector database..."):
                    try:
                        # Clear existing collection
                        qdrant.recreate_collection(
                            collection_name=COLLECTION_NAME,
                            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
                        )
                        
                        # Generate embeddings and create points
                        points = []
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i, chunk in enumerate(chunks):
                            # Update progress
                            progress = (i + 1) / len(chunks)
                            progress_bar.progress(progress)
                            status_text.text(f"Processing chunk {i + 1} of {len(chunks)}...")
                            
                            # Generate embedding for the chunk content
                            embedding = embedder.encode(chunk["content"])
                            
                            # Create point for Qdrant
                            point = PointStruct(
                                id=str(uuid.uuid4()),
                                vector=embedding.tolist(),
                                payload={
                                    "text": chunk["content"],
                                    "metadata": chunk["metadata"]
                                }
                            )
                            points.append(point)
                        
                        # Upload to Qdrant in batches
                        status_text.text("Uploading to vector database...")
                        batch_size = 100
                        for i in range(0, len(points), batch_size):
                            batch = points[i:i + batch_size]
                            qdrant.upsert(collection_name=COLLECTION_NAME, points=batch)
                        
                        # Clean up progress indicators
                        progress_bar.empty()
                        status_text.empty()
                        
                        st.session_state["data_loaded"] = True
                        st.success(f"✅ Automatically processed and stored {len(points)} embeddings in Qdrant!")
                        st.balloons()  # Celebration effect
                        
                    except Exception as e:
                        st.error(f"❌ Error storing embeddings: {e}")
                        st.write("Debug info:", str(e))
                        
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

# --- QUERY SECTION ---
st.header("🧠 Ask a Question")

if not st.session_state.get("data_loaded"):
    st.warning("⚠️ Please upload and process a file first before asking questions.")
else:
    st.success("✅ Data loaded and ready for queries!")

user_query = st.text_input("Enter your question:", placeholder="e.g., give me the names of those with at least 50% response rate")

if st.button("Find now", disabled=not user_query.strip() or not st.session_state.get("data_loaded")):
    try:
        with st.spinner("🔍 Searching for relevant information..."):
            # Generate query embedding
            query_vec = embedder.encode([user_query])[0]
            
            # Search in Qdrant
            results = qdrant.search(
                collection_name=COLLECTION_NAME, 
                query_vector=query_vec, 
                limit=10,  # Increased limit for better coverage
                with_payload=True
            )
            
            # Extract context and metadata
            retrieved_chunks = []
            participant_data = []
            
            for hit in results:
                retrieved_chunks.append(hit.payload['text'])
                metadata = hit.payload.get('metadata', {})
                if metadata.get('chunk_type') == 'participant_profile':
                    participant_data.append(metadata)
            
            context = "\n\n".join(retrieved_chunks)
            
            # Show retrieved context
            with st.expander("📚 Retrieved Context", expanded=False):
                st.write(context)
            
            # Special handling for structured queries
            if "response rate" in user_query.lower() and "%" in user_query:
                # Extract threshold from query
                threshold_match = re.search(r'(\d+)%', user_query)
                if threshold_match:
                    threshold = int(threshold_match.group(1))
                    
                    # Filter participants based on response rate
                    qualified_participants = []
                    for participant in participant_data:
                        if participant.get('response_rate', 0) >= threshold:
                            qualified_participants.append(participant)
                    
                    if qualified_participants:
                        st.subheader("🎯 Direct Results")
                        names = [p.get('name', 'Unknown') for p in qualified_participants]
                        st.write(f"**Participants with at least {threshold}% response rate:**")
                        for name in names:
                            st.write(f"• {name}")
                        st.write(f"\n**Total: {len(names)} participants**")
            
            # Generate AI response with strict factual constraints
            prompt = f"""You are a research assistant that MUST be strictly factual and accurate. Follow these critical rules:

STRICT CONSTRAINTS:
1. ONLY use information that is explicitly provided in the context below
2. NEVER invent, assume, or hallucinate any participant names, data, or details
3. If information is not available in the context, clearly state "This information is not available in the provided data"
4. Do not make up response rates, participant counts, or any other statistics
5. If you cannot find specific information requested, explicitly say so
6. Only cite participants and data that are explicitly mentioned in the context

Context (Research Participant Data):
{context}

Question: {user_query}

Provide a clear, structured answer based ONLY on the information explicitly provided in the context above. If any requested information is missing or unclear in the context, clearly state that it's not available rather than guessing or inventing details."""

            with st.spinner("🤖 Generating AI response..."):
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are a factual research assistant. You must ONLY provide information that is explicitly stated in the provided context. Never invent, assume, or hallucinate any details. If information is not available, clearly state that it's not available in the data."
                        },
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ],
                    temperature=0.1  # Lower temperature for more factual responses
                )
                answer = response.choices[0].message.content.strip()

            st.subheader("💬 AI Response")
            st.info("ℹ️ This response is based strictly on the uploaded data. No information has been invented or assumed.")
            st.write(answer)

    except Exception as e:
        st.error(f"❌ Error during RAG query: {e}")
        st.write("Debug info:", str(e))
