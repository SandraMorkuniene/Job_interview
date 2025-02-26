import streamlit as st
import os
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI

# Initialize the OpenAI LLM (replace 'your-api-key' with a valid OpenAI API key)
if 'llm_temperature' not in st.session_state:
    st.session_state['llm_temperature'] = 0.7
if 'llm_model_name' not in st.session_state:
    st.session_state['llm_model_name'] = 'gpt-4o'

# Set up Streamlit app
st.title('AI Interviewer')
st.markdown("This app generates job interview questions and provides feedback.")

# Input job title, description, interview type, and model selection
job_title = st.text_input('Enter the Job Title:')
job_description = st.text_area('Enter the Job Description:', height=200)
interview_type = st.selectbox('Select Interview Type:', ['Technical', 'Business Case Scenario', 'Behavioral'])
st.session_state['llm_temperature'] = st.slider('Set LLM Temperature:', 0.0, 1.0, 0.7)
st.session_state['llm_model_name'] = st.selectbox('Select LLM Model:', ['gpt-3.5-turbo', 'gpt-4', 'gpt-4o'])

# Initialize the LLM with selected model and temperature
llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model_name=st.session_state['llm_model_name'], temperature=st.session_state['llm_temperature'])

# Initialize memory to avoid question repetition
memory = ConversationBufferMemory(memory_key='conversation_history', return_messages=True,  input_key='job_description', 
    output_key='question')

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
    memory=memory,
    output_key='question'
)

feedback_chain = LLMChain(
    llm=llm,
    prompt=feedback_template,
    output_key='feedback'
)

# Initialize session state for persistence
if 'questions' not in st.session_state:
    st.session_state['questions'] = []
if 'current_question_index' not in st.session_state:
    st.session_state['current_question_index'] = -1
if 'feedback' not in st.session_state:
    st.session_state['feedback'] = ''
if 'conversation' not in st.session_state:
    st.session_state['conversation'] = []

# Generate questions
if st.button('Start Interview') and job_description and job_title:
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
    st.session_state['conversation'] = []

# Display chat-like conversation
for message in st.session_state['conversation']:
    if message['type'] == 'question':
        st.chat_message("assistant").markdown(f"**Question:** {message['content']}")
    elif message['type'] == 'response':
        st.chat_message("user").markdown(f"**Your Response:** {message['content']}")
    elif message['type'] == 'feedback':
        st.chat_message("assistant").markdown(f"**Feedback:** {message['content']}")

# If questions are available, continue the interview flow
if st.session_state['questions'] and st.session_state['current_question_index'] < len(st.session_state['questions']):
    current_question = st.session_state['questions'][st.session_state['current_question_index']]
    
    # Show the current question in a chat-like interface
    st.chat_message("assistant").markdown(f"**Question:** {current_question}")
    
    # Capture candidate's response
    response = st.text_input('Your Response:', key='response_input')

    # Automatically generate feedback and move to the next question
    if st.button('Submit Response') and response:
        st.session_state['conversation'].append({'type': 'question', 'content': current_question})
        st.session_state['conversation'].append({'type': 'response', 'content': response})
        
        feedback = feedback_chain.run({
            'response': response,
            'job_description': job_description,
            'interview_type': interview_type
        })
        
        st.session_state['feedback'] = feedback
        st.session_state['conversation'].append({'type': 'feedback', 'content': feedback})
        
        # Move to the next question automatically
        st.session_state['current_question_index'] += 1
        st.session_state['response'] = ''

# End of interview
if st.session_state['current_question_index'] >= len(st.session_state['questions']) and st.session_state['current_question_index'] > 0:
    st.markdown('### Interview Completed! Thank you for your responses.')
