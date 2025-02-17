import streamlit as st
import os
import re
from openai import OpenAI

# Load OpenAI API Key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Security guard: Prevent inappropriate inputs
def is_valid_input(text):
    inappropriate_words = ["hack", "illegal", "scam", "exploit", "malware"]
    return not any(word in text.lower() for word in inappropriate_words)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a professional interviewer conducting a mock interview."}
    ]
if "question_count" not in st.session_state:
    st.session_state.question_count = 0
if "fail_count" not in st.session_state:
    st.session_state.fail_count = 0

# Configurable interview settings
MAX_QUESTIONS = 10  
MAX_FAILS = 3  # End interview if user fails 3 times in a row

st.title("🤖 AI-Powered Interview Chatbot")

# User inputs
job_title = st.text_input("Enter the job title (e.g., Software Engineer):")
job_desc = st.text_area("Enter job description (optional):")
temperature = st.slider("Adjust creativity (Temperature)", 0.1, 1.0, 0.7)

# Start Interview
if st.button("Start Interview"):
    if not is_valid_input(job_title + job_desc):
        st.error("Inappropriate input detected! Please enter a valid job title/description.")
    elif job_title.strip() == "":
        st.warning("Please enter a job title to proceed.")
    else:
        with st.spinner("Preparing interview..."):
            first_prompt = f"You are a hiring manager for a {job_title}. Conduct a professional interview and ask one question at a time. Start by introducing yourself and asking the first question."
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": first_prompt}],
                temperature=temperature,
                max_tokens=300,
            )
            
            first_question = response.choices[0].message.content
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
            # AI evaluates the answer and determines the next step
            chat_history = st.session_state.messages + [
                {"role": "assistant", "content": "Evaluate the candidate’s answer. If it's good, acknowledge and ask the next question. If it's bad, provide feedback and either ask them to retry or move on."}
            ]
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=chat_history,
                temperature=temperature,
                max_tokens=300,
            )
            
            ai_response = response.choices[0].message.content

            # Track success or failure
            if "incorrect" in ai_response.lower() or "needs improvement" in ai_response.lower():
                st.session_state.fail_count += 1
            else:
                st.session_state.fail_count = 0  # Reset fail count on correct answer

            # Check stopping conditions
            if st.session_state.question_count >= MAX_QUESTIONS:
                final_feedback = "You've completed the interview! Here is your overall evaluation:\n\n" + ai_response
                st.session_state.messages.append({"role": "assistant", "content": final_feedback})
                st.success("Interview Complete!")
            elif st.session_state.fail_count >= MAX_FAILS:
                final_feedback = "You've struggled with multiple questions. Consider reviewing key concepts before retrying. Here are some improvement areas:\n\n" + ai_response
                st.session_state.messages.append({"role": "assistant", "content": final_feedback})
                st.error("Interview Ended Early Due to Low Performance.")
            else:
                # Continue asking the next question
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
                st.session_state.question_count += 1

            st.rerun()  # Refresh UI to show updated conversation
