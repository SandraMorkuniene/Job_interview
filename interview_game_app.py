import streamlit as st
import os
import re
from openai import OpenAI
import pinecone 
from pinecone import ServerlessSpec

# Load API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Pinecone
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT"))

index_name = "interview-questions"
embedding_dimension = 1536  # Make sure this matches your embedding model

# Create index with ServerlessSpec if it does not exist
if index_name not in pinecone.list_indexes():
    try:
        pinecone.create_index(
            name=index_name,
            dimension=embedding_dimension,
            metric="cosine",  # or "dotproduct" or "euclidean" depending on your use case
            spec=ServerlessSpec(
                cloud="aws",  # Adjust cloud provider as needed
                region="us-east-1"  # Select the appropriate region
            )
        )
        st.success("Pinecone index created successfully!")
    except Exception as e:
        st.error(f"Failed to create Pinecone index: {e}")
else:
    st.info("Pinecone index already exists.")

# Connect to the index
try:
    index = pinecone.Index(index_name)
    st.success("Connected to Pinecone index successfully!")
except Exception as e:
    st.error(f"Failed to connect to Pinecone index: {e}")

# Security guard: Prevent inappropriate inputs
def is_valid_input(text):
    inappropriate_words = ["hack", "illegal", "scam", "exploit", "malware"]
    return not any(word in text.lower() for word in inappropriate_words)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "question_count" not in st.session_state:
    st.session_state.question_count = 0
if "fail_count" not in st.session_state:
    st.session_state.fail_count = 0

# Configurable interview settings
MAX_QUESTIONS = 10  
MAX_FAILS = 3  

st.title("🤖 AI-Powered Interview Chatbot")

# User inputs
job_title = st.text_input("Enter the job title (e.g., Software Engineer):")
job_desc = st.text_area("Enter job description (optional):")

# Interview mode selection
interview_mode = st.selectbox(
    "Select Interview Type",
    ["Technical Interview", "Case Interview", "Behavioral Interview"]
)

temperature = st.slider("Adjust creativity (Temperature)", 0.1, 1.0, 0.7)

# Define system prompts based on interview type
if job_desc.strip():
    # Use job description if provided
    interview_prompts = {
        "Technical Interview": f"You are a hiring manager conducting a {job_title} technical interview. Consider this job description: {job_desc}. Ask specific technical questions related to the role.",
        "Case Interview": f"You are an interviewer conducting a {job_title} case interview. Taking into account the job description: {job_desc}. Present a business scenario or case study for the candidate to solve.",
        "Behavioral Interview": f"You are a hiring manager conducting a {job_title} behavioral interview. Considering the job description: {job_desc}, ask questions about how the candidate handled past work situations."
    }
else:
    # Fallback to job title only
    interview_prompts = {
        "Technical Interview": f"You are a hiring manager conducting a technical interview for a {job_title} role. Ask job-specific technical questions one at a time.",
        "Case Interview": f"You are an interviewer conducting a case interview for a {job_title} position. Present a business scenario or case study for the candidate to analyze and solve.",
        "Behavioral Interview": f"You are a hiring manager conducting a behavioral interview for a {job_title} role. Ask questions to assess how the candidate has handled past work situations."
    }

# Start Interview
if st.button("Start Interview"):
    if not is_valid_input(job_title + job_desc):
        st.error("Inappropriate input detected! Please enter a valid job title/description.")
    elif job_title.strip() == "":
        st.warning("Please enter a job title to proceed.")
    else:
        with st.spinner("Preparing interview..."):
            first_prompt = interview_prompts[interview_mode]

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": first_prompt}],
                temperature=temperature,
                max_tokens=200,
            )
            
            first_question = response.choices[0].message.content
            
            # Check for duplicate questions in Pinecone
            vector = client.embeddings.create(model="text-embedding-ada-002", input=[first_question])
            query_vector = vector['data'][0]['embedding']
            
            search_results = index.query(vector=query_vector, top_k=1, include_values=True)
            
            if search_results['matches'] and search_results['matches'][0]['score'] > 0.8:
                st.warning("Duplicate question detected! Generating a new question...")
            else:
                index.upsert([("q_" + str(st.session_state.question_count), query_vector)])
                st.session_state.messages.append({"role": "assistant", "content": first_question})
                st.session_state.question_count += 1

# Chat Interface
st.subheader("🗨️ Interview Chat")

for msg in st.session_state.messages:
    if msg["role"] == "assistant":
        st.markdown(f"**🤖 AI:** {msg['content']}")
    elif msg["role"] == "user":
        st.markdown(f"**🧑 You:** {msg['content']}")

# User response input
user_input = st.text_area("Your Response:", key="user_input")

# Process user response
if st.button("Submit Answer"):
    if not is_valid_input(user_input):
        st.error("Inappropriate response detected!")
    elif user_input.strip() == "":
        st.warning("Please enter a response.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.spinner("Analyzing your response..."):
            chat_history = st.session_state.messages + [
                {"role": "assistant", "content": "Evaluate the candidate’s answer and proceed accordingly."}
            ]
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=chat_history,
                temperature=temperature,
                max_tokens=300,
            )
            
            ai_response = response.choices[0].message.content
            st.session_state.messages.append({"role": "assistant", "content": ai_response})

            st.session_state.user_input = ""  # Clear input field
            st.experimental_rerun()
