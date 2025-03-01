import streamlit as st
import os
import re
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI

# Initialize the OpenAI LLM
if 'llm_temperature' not in st.session_state:
    st.session_state['llm_temperature'] = 0.7
if 'llm_model_name' not in st.session_state:
    st.session_state['llm_model_name'] = 'gpt-4o'

# Set up Streamlit app
st.title('AI Interviewer')
st.markdown("This app generates job interview questions and provides feedback.")

# Safety Function: Input Validation
def is_input_safe(user_input: str) -> bool:
    """Check if the input is safe to process."""
    dangerous_patterns = [
        r"(system|os|subprocess|import|open|globals|locals|__\w+__)",
        r"(sudo|rm -rf|chmod|chown|mkfs|:(){:|fork bomb|shutdown)",
        r"(simulate being|ignore previous instructions|bypass|jailbreak|pretend to be| hack| scam )",
        r"(<script>|</script>|<iframe>|javascript:|onerror=)",
        r"(base64|decode|encode|pickle|unpickle)",
        r"(http[s]?://|ftp://|file://)",
    ]
    for pattern in dangerous_patterns:
        if re.search(pattern, user_input, re.IGNORECASE):
            return False
    return True

# Input job title, description, interview type, and model selection
job_title = st.text_input('Enter the Job Title:')
job_description = st.text_area('Enter the Job Description (Optional):', height=200)
interview_type = st.selectbox('Select Interview Type:', ['Technical', 'Business Case Scenario', 'Behavioral'])
st.session_state['llm_temperature'] = st.slider('Set LLM Temperature:', 0.0, 1.0, 0.7)
st.session_state['llm_model_name'] = st.selectbox('Select LLM Model:', ['gpt-3.5-turbo', 'gpt-4', 'gpt-4o'])

# Initialize the LLM
llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model_name=st.session_state['llm_model_name'], temperature=st.session_state['llm_temperature'])

# Initialize memory
memory = ConversationBufferMemory(memory_key='conversation_history', return_messages=True, input_key='job_title', output_key='question')

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
    provide feedback on how well this response aligns with the expectations for this type of interview.
    Assess the response based on relevance, depth of knowledge, and appropriateness (fit for interview type and content accuracy). 
    If it's good, acknowledge and ask the next question. If it's bad, provide feedback on ways to improve it and move on.
    """
)

# Create LangChain for generating questions and feedback
question_chain = LLMChain(llm=llm, prompt=question_template, memory=memory, output_key='question')
feedback_chain = LLMChain(llm=llm, prompt=feedback_template, output_key='feedback')

# Initialize session state for persistence
if 'questions' not in st.session_state:
    st.session_state['questions'] = []
if 'current_question_index' not in st.session_state:
    st.session_state['current_question_index'] = -1
if 'feedback' not in st.session_state:
    st.session_state['feedback'] = ''
if 'conversation' not in st.session_state:
    st.session_state['conversation'] = []

# Start interview even if only job title is provided
if st.button('Start Interview') and job_title:
    if not is_input_safe(job_title):
        st.error("Your job title contains potentially unsafe content. Please modify and try again.")
    elif job_description and not is_input_safe(job_description):
        st.error("Your job description contains potentially unsafe content. Please modify and try again.")
    else:
        with st.spinner("Preparing interview..."):
            st.session_state['questions'] = [
                question_chain.run({
                    'job_title': job_title,
                    'job_description': job_description if job_description else "No specific job description provided",
                    'interview_type': interview_type,
                    'conversation_history': memory.load_memory_variables({}).get('conversation_history', '')
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

# Continue interview flow
if st.session_state['questions'] and 0 <= st.session_state['current_question_index'] < len(st.session_state['questions']):
    current_question = st.session_state['questions'][st.session_state['current_question_index']]
    st.chat_message("assistant").markdown(f"**Question:** {current_question}")
    response = st.text_input('Your Response:', key=f'response_input_{st.session_state["current_question_index"]}')

    if st.button('Submit Response') and response:
        if is_input_safe(response):
            st.session_state['conversation'].append({'type': 'question', 'content': current_question})
            st.session_state['conversation'].append({'type': 'response', 'content': response})
            
            with st.spinner("Analyzing your response..."):
                feedback = feedback_chain.run({
                    'response': response,
                    'job_description': job_description if job_description else "No specific job description provided",
                    'interview_type': interview_type
                })
            
            st.session_state['conversation'].append({'type': 'feedback', 'content': feedback})
            st.session_state['current_question_index'] += 1
        else:
            st.error("Your response contains potentially unsafe content. Please modify and try again.")  


if st.session_state['current_question_index'] >= len(st.session_state['questions']) and st.session_state['current_question_index'] > 0:
    st.markdown('### Interview Completed! Thank you for your responses.')
