import streamlit as st
import os
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool

# Initialize the OpenAI LLM (replace 'your-api-key' with a valid OpenAI API key)
llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))

# Set up Streamlit app
st.title('AI Interviewer Based on Job Description')
st.markdown("This app generates interview questions based on a job description and provides feedback.")

# Input job description
job_description = st.text_area('Enter the Job Description:', height=200)

# Initialize memory to avoid question repetition
memory = ConversationBufferMemory(memory_key='conversation_history', return_messages=True)

# Define prompt template for question generation
question_template = PromptTemplate(
    input_variables=['job_description', 'conversation_history'],
    template="""
    Based on the job description: "{job_description}",
    and considering the previous questions: "{conversation_history}",
    generate a relevant interview question that has not been asked before.
    """
)

# Define feedback template
feedback_template = PromptTemplate(
    input_variables=['response', 'job_description'],
    template="""
    Considering the response: "{response}",
    and the job description: "{job_description}",
    provide feedback on how well this response aligns with the job requirements.
    """
)

# Create LangChain for generating questions
question_chain = LLMChain(
    llm=llm,
    prompt=question_template,
    memory=memory
)

# Create LangChain for generating feedback
feedback_chain = LLMChain(
    llm=llm,
    prompt=feedback_template
)

# Initialize session state for persistence
if 'question' not in st.session_state:
    st.session_state['question'] = ''
if 'response' not in st.session_state:
    st.session_state['response'] = ''
if 'feedback' not in st.session_state:
    st.session_state['feedback'] = ''

# Handle interview process
if st.button('Generate Interview Question') and job_description:
    st.session_state['question'] = question_chain.run({'job_description': job_description, 'conversation_history': memory.load_memory_variables({})['conversation_history']})

if st.session_state['question']:
    st.subheader('Interview Question:')
    st.write(st.session_state['question'])

    st.session_state['response'] = st.text_input('Candidate Response:', st.session_state['response'])

    if st.button('Get Feedback') and st.session_state['response']:
        st.session_state['feedback'] = feedback_chain.run({'response': st.session_state['response'], 'job_description': job_description})

if st.session_state['feedback']:
    st.subheader('Feedback:')
    st.write(st.session_state['feedback'])

st.markdown("---")
st.markdown("Developed using LangChain and Streamlit.")
