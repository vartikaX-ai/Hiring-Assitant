"""
TalentScout - AI Hiring Assistant
----------------------------------
Streamlit-based AI chatbot that conducts technical screening interviews.

Features:
- Collects candidate information
- Generates 3–5 technical questions based on tech stack
- Evaluates answers using LLM
- Maintains session context
- Stores candidate data locally (CSV)
"""

import streamlit as st
import os
import re
import csv
from dotenv import load_dotenv
from groq import Groq
from textblob import TextBlob
from prompts import SYSTEM_PROMPT


# ==========================================================
# Environment & LLM Setup
# ==========================================================

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def call_llm(messages: list) -> str:
    """
    Calls the Groq LLM model and returns the generated response.
    """
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.5,
    )
    return completion.choices[0].message.content


# ==========================================================
# Validation Utilities
# ==========================================================

def validate_email(email: str) -> bool:
    """Validate email format using regex."""
    return bool(re.match(r"[^@]+@[^@]+\.[^@]+", email))


def validate_phone(phone: str) -> bool:
    """Validate phone number (10–15 digits, optional +)."""
    return bool(re.match(r"^\+?\d{10,15}$", phone))


def analyze_sentiment(text: str) -> str:
    """Return sentiment classification using TextBlob."""
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.2:
        return "Positive"
    elif polarity < -0.2:
        return "Negative"
    return "Neutral"

import os

def save_to_csv(data: dict) -> None:
    """
    Save candidate data into data/candidates.csv.
    Creates the folder and file if they do not exist.
    """

    # Ensure data folder exists
    os.makedirs("data", exist_ok=True)

    file_path = os.path.join("data", "candidates.csv")
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=data.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(data)


# ==========================================================
# Session Initialization
# ==========================================================

def initialize_session():
    """Initialize all required session state variables."""

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    if "stage" not in st.session_state:
        st.session_state.stage = "greeting"

    if "candidate_data" not in st.session_state:
        st.session_state.candidate_data = {
            "name": None,
            "email": None,
            "phone": None,
            "experience": None,
            "position": None,
            "location": None,
            "tech_stack": None,
        }

    if "questions" not in st.session_state:
        st.session_state.questions = []

    if "current_question_index" not in st.session_state:
        st.session_state.current_question_index = 0

    if "max_questions" not in st.session_state:
        st.session_state.max_questions = 5  # Between 3–5 only


# ==========================================================
# Question Generation
# ==========================================================

def generate_technical_questions(candidate_data: dict) -> list:
    """
    Generate exactly 3–5 technical questions based on tech stack.
    """

    tech_list = [tech.strip() for tech in candidate_data["tech_stack"].split(",")]

    prompt = f"""
    Generate exactly {st.session_state.max_questions} technical interview questions.

    Candidate Profile:
        - Experience: {candidate_data['experience']} years
        - Position: {candidate_data['position']}
        - Tech Stack: {tech_list}

    Rules:
        1. Generate exactly {st.session_state.max_questions} questions.
        2. Do NOT include explanations.
        3. Output only numbered questions.
    """

    response = call_llm(
        st.session_state.messages + [{"role": "user", "content": prompt}]
    )

    questions = re.findall(r"\d+\.\s(.+)", response)

    return questions[:st.session_state.max_questions]


def evaluate_answer(candidate_data: dict, answer: str) -> str:
    """
    Evaluate candidate answer and return constructive feedback.
    """

    prompt = f"""
    Evaluate the candidate's answer.

    Candidate Profile:
        - Experience: {candidate_data['experience']} years
        - Position: {candidate_data['position']}

    Candidate Answer:
        {answer}

    Provide short constructive feedback.
    """

    return call_llm(
        st.session_state.messages + [{"role": "user", "content": prompt}]
    )


# ==========================================================
# Streamlit UI Setup
# ==========================================================

st.set_page_config(page_title="TalentScout Hiring Assistant")
st.title("TalentScout - AI Hiring Assistant")

st.info(
    "This AI Hiring Assistant collects your information for screening purposes only. "
    "All data is stored locally and handled securely. "
    "Type 'exit' anytime to end the conversation."
)

initialize_session()


# ==========================================================
# Greeting Stage
# ==========================================================

if st.session_state.stage == "greeting":
    with st.chat_message("assistant"):
        st.markdown(
            "Hello 👋 I'm TalentScout AI Hiring Assistant.\n\n"
            "I will conduct your technical screening.\n\n"
            "Please provide your Full Name."
        )
    st.session_state.stage = "collecting_info"


# ==========================================================
# Display Chat History
# ==========================================================

for message in st.session_state.messages[1:]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ==========================================================
# Main Interaction Logic
# ==========================================================

if prompt := st.chat_input("Type your response..."):

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    sentiment = analyze_sentiment(prompt)
    response = ""

    # ---------------- EXIT ----------------
    if prompt.lower() in ["exit", "quit", "bye"]:
        response = (
            "Thank you for participating in the TalentScout screening.\n\n"
            "Our recruitment team will contact you soon."
        )
        st.session_state.stage = "finished"

    # ---------------- INFO COLLECTION ----------------
    elif st.session_state.stage == "collecting_info":

        data = st.session_state.candidate_data

        if data["name"] is None:
            data["name"] = prompt
            response = "Please provide your Email Address."

        elif data["email"] is None:
            if validate_email(prompt):
                data["email"] = prompt
                response = "Please provide your Phone Number."
            else:
                response = "Invalid email format. Please enter a valid email."

        elif data["phone"] is None:
            if validate_phone(prompt):
                data["phone"] = prompt
                response = "How many years of experience do you have?"
            else:
                response = "Invalid phone number. Enter 10–15 digits."

        elif data["experience"] is None:
            if prompt.isdigit():
                data["experience"] = int(prompt)
                response = "What position are you applying for?"
            else:
                response = "Please enter numeric experience (e.g., 2)."

        elif data["position"] is None:
            data["position"] = prompt
            response = "What is your current location?"

        elif data["location"] is None:
            data["location"] = prompt
            response = "Please list your Tech Stack (comma-separated)."

        elif data["tech_stack"] is None:
            data["tech_stack"] = prompt
            save_to_csv(data)

            st.session_state.questions = generate_technical_questions(data)
            st.session_state.current_question_index = 0
            st.session_state.stage = "tech_questions"

            response = f"Question 1:\n\n{st.session_state.questions[0]}"

    # ---------------- TECH QUESTIONS ----------------
    elif st.session_state.stage == "tech_questions":

        index = st.session_state.current_question_index
        total = len(st.session_state.questions)

        if index >= total:
            response = (
                "Interview completed.\n\n"
                "Thank you for your time. Our team will contact you soon."
            )
            st.session_state.stage = "finished"

        else:
            feedback = evaluate_answer(
                st.session_state.candidate_data,
                prompt,
            )

            st.session_state.current_question_index += 1
            index += 1

            if index < total:
                next_question = st.session_state.questions[index]
                response = f"{feedback}\n\n---\n\nQuestion {index + 1}:\n\n{next_question}"
            else:
                response = (
                    f"{feedback}\n\n---\n\n"
                    "Interview completed.\n\n"
                    "Thank you for your time."
                )
                st.session_state.stage = "finished"

    # ---------------- SENTIMENT FEEDBACK ----------------
    if sentiment == "Negative":
        st.info("Take your time — you're doing well.")
    elif sentiment == "Positive":
        st.success("Great confidence! Keep it up.")

    st.session_state.messages.append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        st.markdown(response)