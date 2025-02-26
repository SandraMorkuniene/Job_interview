import streamlit as st
import os
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool

# Initialize the OpenAI LLM (replace 'your-api-key' with a valid OpenAI API key)
if 'llm_temperature' not in st.session_state:
    st.session_state['llm_temperature'] = 0.7

llm = OpenAI(openai_api_key='your-api-key', temperature=st.session_state['llm_temperature'])

# Set up Streamlit app
st.title('AI Interviewer Based on Job Description')
st.markdown("This app generates interview questions based on a job description and provides feedback.")

# Input job title, description, and interview type
job_title = st.text_input('Enter the Job Title:')
job_description = st.text_area('Enter the Job Description:', height=200)
interview_type = st.selectbox('Select Interview Type:', ['Technical', 'Business Case Scenario', 'Behavioral'])
st.session_state['llm_temperature'] = st.slider('Set LLM Temperature:', 0.0, 1.0, 0.7)

# Initialize memory to avoid question repetition
memory = ConversationBufferMemory(memory_key='conversation_history', return_messages=True)

# Define prompt template for question generation
question_template = PromptTemplate(
    input_variables=['job_title', 'job_description', 'interview_type', 'conversation_history'],
    template="""
    Based on the job title: "{job_title}",
    job description: "{job_description}",
    and the interview type: "{interview_type}",
    considering the previous questions: "{conversation_history}",
    generate a relevant interview question that has not been asked before.
    """
)

# Define feedback template
feedback_template = PromptTemplate(
    input_variables=['response', 'job_description', 'interview_type'],
    template="""
    Considering the response: "{response}",
    the job description: "{job_description}",
    and the interview type: "{interview_type}",
    provide detailed feedback on how well this response aligns with the job requirements and expectations for this type of interview.
    Assess the response based on relevance, clarity, depth of knowledge, and appropriateness for the interview type.
    """
)

# Create LangChain for generating questions and feedback
question_chain = LLMChain(
    llm=llm,
    prompt=question_template,
    memory=memory
)

feedback_chain = LLMChain(
    llm=llm,
    prompt=feedback_template
)

# Initialize session state for persistence
if 'questions' not in st.session_state:
    st.session_state['questions'] = []
if 'current_question_index' not in st.session_state:
    st.session_state['current_question_index'] = 0
if 'response' not in st.session_state:
    st.session_state['response'] = ''
if 'feedback' not in st.session_state:
    st.session_state['feedback'] = ''

# Generate a batch of 10 questions
if st.button('Generate 10 Interview Questions') and job_description and job_title:
    st.session_state['questions'] = [
        question_chain.run({
            'job_title': job_title,
            'job_description': job_description,
            'interview_type': interview_type,
            'conversation_history': memory.load_memory_variables({})['conversation_history']
        }) for _ in range(10)
    ]
    st.session_state['current_question_index'] = 0
    st.session_state['feedback'] = ''
    st.session_state['response'] = ''

# Display the current question and handle responses
if st.session_state['questions'] and st.session_state['current_question_index'] < len(st.session_state['questions']):
    st.subheader('Interview Question:')
    st.write(st.session_state['questions'][st.session_state['current_question_index']])

    st.session_state['response'] = st.text_input('Candidate Response:', st.session_state['response'])

    if st.button('Submit Response') and st.session_state['response']:
        st.session_state['feedback'] = feedback_chain.run({
            'response': st.session_state['response'],
            'job_description': job_description,
            'interview_type': interview_type
        })
        st.session_state['current_question_index'] += 1
        st.session_state['response'] = ''

# Show feedback if available
if st.session_state['feedback']:
    st.subheader('Feedback:')
    st.write(st.session_state['feedback'])

# Completion message
if st.session_state['current_question_index'] >= len(st.session_state['questions']):
    st.markdown('### Interview Completed! Thank you for your responses.')

st.markdown("---")
st.markdown("Developed using LangChain and Streamlit.")

