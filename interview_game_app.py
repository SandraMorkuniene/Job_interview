import streamlit as st
import pinecone
from pinecone import Pinecone, ServerlessSpec
import os
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, Tool, AgentType
from openai import OpenAI
from langchain.tools import Tool
from openai import Image as OpenAIImage
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone


# Initialize OpenAI client and Pinecone
client = ChatOpenAI(model="gpt-4o", openai_api_key=os.getenv("OPENAI_API_KEY"))
#client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "interview-questions"
embedding_dimension = 1536  # For OpenAI text-embedding-ada-002 model

# Check if index exists, if not, create it
if index_name not in pc.list_indexes().names():
    try:
        pc.create_index(
            name=index_name,
            dimension=embedding_dimension,
            metric="cosine",  # Use 'cosine', 'dotproduct', or 'euclidean' as needed
            spec=ServerlessSpec(
                cloud="aws",  # Your cloud provider
                region="us-east-1"  # Appropriate region
            )
        )
    except Exception as e:
        print(f"Error creating index: {e}")

# Connect to the index
index = pc.Index(index_name)


# Initialize LangChain memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Function to get the question chain
def get_question_chain():
    template = "Generate a question for the interview of a {job_title} with interview type {interview_mode}. Job description: {job_description}"
    prompt = PromptTemplate(input_variables=["job_title", "interview_mode", "job_description"], template=template)
    llm_chain = LLMChain(prompt=prompt, llm=client)
    return llm_chain

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=os.getenv("OPENAI_API_KEY"))

# Function to check for duplicate questions
def check_duplicate_question(question):
    vector = embeddings.embed_query(question)
    #query_vector = vector['data'][0]['embedding']
    
    search_results = index.query(vector=vector, top_k=1, include_values=True)
    
    if search_results['matches'] and search_results['matches'][0]['score'] > 0.8:
        return True
    else:
        index.upsert([("q_" + str(len(search_results['matches'])), query_vector)])
        return False

# Function to generate image with DALL·E for visual aid
def generate_image_prompt(question):
    image_prompt = f"Create an image that visually represents the concept of: {question}. The image should help explain the concept in a clear and easy-to-understand way."
    response = OpenAIImage.create(prompt=image_prompt, n=1, size="1024x1024")
    image_url = response['data'][0]['url']
    return image_url

# Function to create the interview agent
def get_interview_agent():
    duplicate_tool = Tool.from_function(
        func=check_duplicate_question,
        name="DuplicateChecker",
        description="Checks if a question is a duplicate using Pinecone embeddings."
    )
    tools = [duplicate_tool]
    
    agent = initialize_agent(
        tools=tools, 
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
        llm=client,
        memory=memory, 
        verbose=True
    )
    return agent

# Function to evaluate the response
def evaluate_response(response, question):
    prompt = f"Evaluate the candidate's response to the following question:\n\nQuestion: {question}\nResponse: {response}\n\nProvide feedback on the quality of the response, whether it's correct, and any improvements or suggestions."
    feedback = client.completions.create(model="gpt-4o", prompt=prompt, max_tokens=150)
    return feedback['choices'][0]['text'].strip()

# Streamlit app interface
st.title("🤖 AI-Powered Adaptive Interview Chatbot")

job_title = st.text_input("Enter the job title (e.g., Software Engineer):")
job_desc = st.text_area("Enter job description (optional):")
interview_mode = st.selectbox("Select Interview Type", ["Technical Interview", "Case Interview", "Behavioral Interview"])

# Initialize question chain and agent
question_chain = get_question_chain()
agent = get_interview_agent()

# Start the interview button
if st.button("Start Interview"):
    # Generate the initial question
    question = question_chain.run({
        "job_title": job_title,
        "job_description": job_desc,
        "interview_mode": interview_mode
    })

    # Check if the generated question is a duplicate
    is_duplicate = check_duplicate_question(question)
    
    if is_duplicate:
        st.warning("Duplicate question detected! Generating a new question...")
        # Generate a new question if a duplicate is found
        question = question_chain.run({
            "job_title": job_title,
            "job_description": job_desc,
            "interview_mode": interview_mode
        })
    
    # Show the interview question
    st.session_state.messages.append({"role": "assistant", "content": question})
    st.markdown(f"**🤖 AI:** {question}")

    # Simulate collecting candidate response (you can integrate an actual input field here)
    candidate_response = st.text_area("Your response:", key="candidate_response")

    # Process the response and provide feedback
    if candidate_response:
        st.session_state.messages.append({"role": "user", "content": candidate_response})
        
        # Evaluate the candidate's response
        feedback = evaluate_response(candidate_response, question)
        st.markdown(f"**🤖 Feedback:** {feedback}")

        # Optionally provide visual aid if the candidate is struggling
        if 'struggling' in candidate_response.lower():
            st.warning("Candidate is struggling. Generating a visual aid...")
            image_url = generate_image_prompt(question)
            st.image(image_url, caption="Visual Aid", use_column_width=True)
