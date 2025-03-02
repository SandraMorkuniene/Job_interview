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
        r"\b(system|os|subprocess|import|open|globals|locals|__import__|__globals__|__dict__|__builtins__)\b",
        r"(sudo|rm -rf|chmod|chown|mkfs|:(){:|fork bomb|shutdown)",
        r"\b(simulate being|ignore previous instructions|bypass|jailbreak|pretend to be| hack| scam )\b",
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

# Define prompt templates
question_template = PromptTemplate(
    input_variables=['job_title', 'job_description', 'interview_type', 'conversation_history'],
    template="""
    You are a hiring manager conducting a {interview_type} interview for a {job_title} position. 
    The job description: {job_description}.
    
    Considering the past questions and responses: "{conversation_history}", generate the next relevant question.
    Ensure the question is logically progressing and relevant to the previous responses.
    Provide the question directly, without any introductory phrases or formalities.
    Do not deviate from your role as an interviewer.
    """
)

feedback_template = PromptTemplate(
    input_variables=['response', 'job_description', 'interview_type'],
    template="""
    Considering the response: "{response}",
    the job description: "{job_description}",
    and the interview type: "{interview_type}",
    evaluate the response for clarity, relevance, and depth. 
    If the response is unclear or lacks detail, suggest a follow-up question to guide the candidate.
    Otherwise, provide constructive feedback.
    """
)

# Create LangChain for generating questions and feedback
question_chain = LLMChain(llm=llm, prompt=question_template, memory=memory, output_key='question')
feedback_chain = LLMChain(llm=llm, prompt=feedback_template, output_key='feedback')

# Initialize session state variables
if 'current_question' not in st.session_state:
    st.session_state['current_question'] = None
if 'waiting_for_followup' not in st.session_state:
    st.session_state['waiting_for_followup'] = False
if 'conversation' not in st.session_state:
    st.session_state['conversation'] = []

# Start Interview
if st.button('Start Interview') and job_title:
    if not is_input_safe(job_title):
        st.error("Your job title contains unsafe content. Modify and try again.")
    elif job_description and not is_input_safe(job_description):
        st.error("Your job description contains unsafe content. Modify and try again.")
    else:
        st.session_state['conversation'] = []
        st.session_state['waiting_for_followup'] = False
        st.session_state['current_question'] = question_chain.run({
            'job_title': job_title,
            'job_description': job_description if job_description else "No specific job description provided",
            'interview_type': interview_type,
            'conversation_history': memory.load_memory_variables({}).get('conversation_history', '')
        })
        st.rerun()

# Interview flow
if st.session_state['current_question']:
    st.chat_message("assistant").markdown(f"**Question:** {st.session_state['current_question']}")
    
    response = st.text_area('Your Response:', height=400)

    if st.button('Submit Response'):
        if not response:
            st.error("Please enter a response before submitting.")
        elif not is_input_safe(response):
            st.error("Your response contains unsafe content. Modify and try again.")
        else:
            st.session_state['conversation'].append({'type': 'question', 'content': st.session_state['current_question']})
            st.session_state['conversation'].append({'type': 'response', 'content': response})

            with st.spinner("Analyzing your response..."):
                feedback = feedback_chain.run({
                    'response': response,
                    'job_description': job_description if job_description else "No specific job description provided",
                    'interview_type': interview_type
                })

            # Extract follow-up question if response is weak
            if "Response Strength Classification: Weak" in feedback:
                followup_match = re.search(r'Follow-up Question: (.+)', feedback)
                if followup_match:
                    followup_question = followup_match.group(1)
                    st.session_state['current_question'] = followup_question
                    st.session_state['waiting_for_followup'] = True
                else:
                    st.session_state['waiting_for_followup'] = False
            else:
                st.session_state['waiting_for_followup'] = False

            # Append feedback
            st.session_state['conversation'].append({'type': 'feedback', 'content': feedback})
            
            # If response was strong, ask a new question
            if not st.session_state['waiting_for_followup']:
                st.session_state['current_question'] = question_chain.run({
                    'job_title': job_title,
                    'job_description': job_description if job_description else "No specific job description provided",
                    'interview_type': interview_type,
                    'conversation_history': memory.load_memory_variables({}).get('conversation_history', '')
                })

            st.rerun()

# Display conversation history
for message in st.session_state['conversation']:
    role = "assistant" if message['type'] in ['question', 'feedback'] else "user"
    st.chat_message(role).markdown(f"**{message['type'].capitalize()}:** {message['content']}")

# Exit interview button
if st.button('Exit Interview'):
    st.session_state.clear()
    st.rerun()
