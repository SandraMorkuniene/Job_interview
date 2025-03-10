import streamlit as st
import os
import re
from langchain import LLMChain, PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI

# Initialize the OpenAI LLM
if 'llm_temperature' not in st.session_state:
    st.session_state['llm_temperature'] = 0.7
if 'llm_model_name' not in st.session_state:
    st.session_state['llm_model_name'] = 'gpt-4o'

# Set up Streamlit UI
st.title('AI Interviewer')
st.markdown("This app generates job interview questions, collects responses, and provides feedback.")

# Safety Function
def is_input_safe(user_input: str) -> bool:
    """Check if the input is safe to process."""
    dangerous_patterns = [
        r"\b(system|os|subprocess|import|open|globals|locals|__import__|__globals__|__dict__|__builtins__)\b",
        r"(sudo|rm -rf|chmod|chown|mkfs|:(){:|fork bomb|shutdown)",
        r"\b(simulate being|ignore previous instructions|bypass|jailbreak|pretend to be|hack|scam )\b",
        r"(<script>|</script>|<iframe>|javascript:|onerror=)",
        r"(base64|decode|encode|pickle|unpickle)",
        r"(http[s]?://|ftp://|file://)",
    ]
    return not any(re.search(pattern, user_input, re.IGNORECASE) for pattern in dangerous_patterns)

# Inputs: Job details
job_title = st.text_input('Enter the Job Title:')
job_description = st.text_area('Enter the Job Description (Optional):', height=200)
interview_type = st.selectbox('Select Interview Type:', ['Technical', 'Business Case Scenario', 'Behavioral'])
st.session_state['llm_temperature'] = st.slider('Set LLM Temperature:', 0.0, 1.0, 0.7)
st.session_state['llm_model_name'] = st.selectbox('Select LLM Model:', ['gpt-3.5-turbo', 'gpt-4', 'gpt-4o'])

# Initialize LLM
llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), 
                 model_name=st.session_state['llm_model_name'], 
                 temperature=st.session_state['llm_temperature'])

# Initialize memory
memory = ConversationBufferMemory(memory_key='conversation_history', return_messages=True, 
                                  input_key='job_title', output_key='question')

# Define question generation template
question_template = PromptTemplate(
    input_variables=['job_title', 'job_description', 'interview_type', 'conversation_history'],
    template="""
    You are a hiring manager conducting a {interview_type} interview for a {job_title} role.
    Job description (if provided): {job_description}.
    
    Based on previous questions asked: "{conversation_history}", generate a **new** relevant interview question. 
    - If this is a **Technical** interview, ask a job-specific technical question.
    - If it's a **Business Case Scenario** interview, present a business case or scenario.
    - If it's a **Behavioral** interview, ask about a past experience using the STAR method.

    Provide the question directly without any introductory phrases.
    """
)

# Define feedback evaluation template
feedback_template = PromptTemplate(
    input_variables=['question', 'response', 'job_title', 'job_description', 'interview_type'],
    template="""
    Evaluating response to interview question: "{question}" for the role "{job_title}".
    Job description (if provided): {job_description}.
    Candidate's response: "{response}".

    Assessment:
    - If this is a **Technical** interview, check if the response is correct, structured well, and demonstrates strong technical knowledge.
    - If it's a **Business Case Scenario** interview, analyze if the response logically addresses the scenario and follows a structured framework.
    - If it's a **Behavioral** interview, evaluate if the response follows the STAR method (Situation, Task, Action, Result) and demonstrates relevant competencies.

    Provide detailed **constructive feedback**, mentioning areas of improvement if needed.
    """
)

# LangChain for generating questions and feedback
question_chain = LLMChain(llm=llm, prompt=question_template, memory=memory, output_key='question')
feedback_chain = LLMChain(llm=llm, prompt=feedback_template, output_key='feedback')

# Initialize session state
if 'current_question' not in st.session_state:
    st.session_state['current_question'] = ''
if 'conversation' not in st.session_state:
    st.session_state['conversation'] = []
if 'waiting_for_response' not in st.session_state:
    st.session_state['waiting_for_response'] = False

# Generate the first question when "Start Interview" is pressed
if st.button('Start Interview') and job_title:
    if not is_input_safe(job_title) or (job_description and not is_input_safe(job_description)):
        st.error("Your input contains potentially unsafe content. Please modify and try again.")
    else:
        with st.spinner("Generating first question..."):
            st.session_state['current_question'] = question_chain.run({
                'job_title': job_title,
                'job_description': job_description if job_description else "No specific job description provided",
                'interview_type': interview_type,
                'conversation_history': memory.load_memory_variables({}).get('conversation_history', '')
            })
        st.session_state['conversation'].append({'type': 'question', 'content': st.session_state['current_question']})
        st.session_state['waiting_for_response'] = True
        st.rerun()

# Display previous conversation
for message in st.session_state['conversation']:
    if message['type'] == 'question':
        st.chat_message("assistant").markdown(f"**Question:** {message['content']}")
    elif message['type'] == 'response':
        st.chat_message("user").markdown(f"**Your Response:** {message['content']}")
    elif message['type'] == 'feedback':
        st.chat_message("assistant").markdown(f"**Feedback:** {message['content']}")

# If waiting for response, show input field
if st.session_state['waiting_for_response']:
    response = st.text_area('Your Response:', key='user_response', height=400)

    if st.button('Submit Response'):
        if not response:
            st.error("Please enter a response before submitting.")
        elif not is_input_safe(response):
            st.error("Your response contains potentially unsafe content. Please modify and try again.")
        else:
            st.session_state['conversation'].append({'type': 'response', 'content': response})

            with st.spinner("Analyzing your response..."):
                feedback = feedback_chain.run({
                    'question': st.session_state['current_question'],
                    'response': response,
                    'job_title': job_title,
                    'job_description': job_description if job_description else "No specific job description provided",
                    'interview_type': interview_type
                })

            st.session_state['conversation'].append({'type': 'feedback', 'content': feedback})

            with st.spinner("Generating next question..."):
                new_question = question_chain.run({
                    'job_title': job_title,
                    'job_description': job_description if job_description else "No specific job description provided",
                    'interview_type': interview_type,
                    'conversation_history': memory.load_memory_variables({}).get('conversation_history', '')
                })
            
            st.session_state['current_question'] = new_question
            st.session_state['conversation'].append({'type': 'question', 'content': new_question})
            st.rerun()

if st.button('Exit Interview'):
    st.session_state.clear()
    st.rerun()
