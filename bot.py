import os
import fitz  # PyMuPDF
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq


load_dotenv()
key = os.getenv("bot")


if not key:
    st.error("chat_bot is not set. Please ensure it is defined in your .env file.")
    st.stop()


chat_model = ChatGroq(api_key=key, model_name="llama3-8b-8192")

st.title("ChatGroq PDF Assistant")


if 'logged_in_user' not in st.session_state:
    st.session_state.logged_in_user = None

USER_DATA_FILE = 'users.csv'


user_data = pd.read_csv(USER_DATA_FILE) if os.path.exists(USER_DATA_FILE) else pd.DataFrame(columns=["email", "password"])

if st.session_state.logged_in_user is None:
    st.sidebar.header("Options")
    page_option = st.sidebar.selectbox("Choose an option:", ["Sign Up", "Login"])

    if page_option == "Sign Up":
        st.header("Create an Account")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        if st.button("Sign Up"):
            if email in user_data['email'].values:
                st.error("Email already exists.")
            elif not email:
                st.error("Email cannot be empty.")
            elif not password:
                st.error("Password cannot be empty.")
            else:
                new_entry = pd.DataFrame({"email": [email], "password": [password]})
                user_data = pd.concat([user_data, new_entry], ignore_index=True)
                user_data.to_csv(USER_DATA_FILE, index=False)
                st.success("Account created successfully!")

    elif page_option == "Login":
        st.header("Log In")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        if st.button("Log In"):
            if email not in user_data['email'].values:
                st.error("Email not found.")
            elif user_data.loc[user_data['email'] == email, 'password'].values[0] != password:
                st.error("Incorrect password.")
            else:
                st.session_state.logged_in_user = email
                st.success("Logged in successfully!")

else:
    st.sidebar.header("Options")
    option = st.sidebar.selectbox("Choose an option:", ["Upload PDF", "Ask Question", "Chat with Bot"])

    if 'pdf_text' not in st.session_state:
        st.session_state.pdf_text = ""

    if option == "Upload PDF":
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        if uploaded_file:
            pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            pdf_text = ""
            for page in pdf_document:
                pdf_text += page.get_text()
            st.session_state.pdf_text = pdf_text
            st.success("PDF uploaded and processed.")
            st.text_area("PDF Content", st.session_state.pdf_text, height=300)

    elif option == "Ask Question":
        if not st.session_state.pdf_text:
            st.warning("Please upload a PDF first.")
        else:
            prompt = st.chat_input("What question do you have about the PDF?")
            if prompt:
                pdf_text = st.session_state.pdf_text
                if len(pdf_text) > 2000:
                    chunks = [pdf_text[i:i + 2000] for i in range(0, len(pdf_text), 2000)]
                    responses = [chat_model.invoke(
                        f"Based on the following text, answer the question: {prompt}\n\nText: {chunk}") for chunk in chunks]
                    full_response = "\n\n".join(
                        [response.content if hasattr(response, 'content') else str(response) for response in responses])
                else:
                    prompt_full = f"Based on the following text, answer the question: {prompt}\n\nText: {pdf_text}"
                    response = chat_model.invoke(prompt_full)
                    full_response = response.content if hasattr(response, 'content') else response
                st.text_area("ChatGroq Response", full_response, height=300)

    elif option == "Chat with Bot":
        st.header("Chat with Bot")
        prompt = st.chat_input("Say something")
        if prompt:
            response = chat_model.invoke(prompt)
            st.text_area("ChatGroq Response", response.content if hasattr(response, 'content') else response, height=300)
