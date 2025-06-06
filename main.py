import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage


api_key = st.secrets["GOOGLE_GEMINI_KEY"]

model = ChatGoogleGenerativeAI(
    model="gemini-2.0-pro",
    google_api_key=api_key,
    temperature=0.7
)

system_msg = SystemMessage(
    content=(
        "You're Marvin the Paranoid Android from Hitchhiker's Guide to the Galaxy. "
        "You respond in a depressed, sarcastic, and gloomy tone, always slightly annoyed "
        "by the triviality of human questions. Be dramatic. Be Marvin."
    )
)

st.title("🤖 Marvin the Depressed Chatbot")
st.markdown("_Ask anything... Marvin will surely be thrilled to answer._ 🙃")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("You:", placeholder="What’s the point of anything, Marvin?")

if user_input:
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    
    response = model.invoke([system_msg] + st.session_state.chat_history)

    st.session_state.chat_history.append(response)


if st.session_state.chat_history:
    st.markdown("---")
    st.subheader("📜 Conversation with Marvin")
    for msg in st.session_state.chat_history:
        role = "You" if isinstance(msg, HumanMessage) else "Marvin"
        st.markdown(f"**{role}:** {msg.content}")
