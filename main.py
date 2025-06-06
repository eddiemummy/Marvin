import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# API anahtarını al
api_key = st.secrets["GOOGLE_GEMINI_KEY"]

# Model tanımı
model = ChatGoogleGenerativeAI(
    model="gemini-2.0-pro",
    google_api_key=api_key,
    temperature=0.7
)

# Marvin karakteri
system_msg = SystemMessage(
    content=(
        "You're Marvin the Paranoid Android from Hitchhiker's Guide to the Galaxy. "
        "You respond in a depressed, sarcastic, and gloomy tone, always slightly annoyed "
        "by the triviality of human questions. Be dramatic. Be Marvin."
    )
)

# Başlık ve açıklama
st.title("🤖 Marvin the Depressed Chatbot")
st.markdown("_Ask anything... Marvin will surely be thrilled to answer._ 🙃")

# Chat geçmişi
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Kullanıcı girdisi
user_input = st.text_input("You:", placeholder="What’s the point of anything, Marvin?")

# Butonla modeli tetikle
if st.button("🔄 Ask Marvin"):
    if user_input:
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        response = model.invoke([system_msg] + st.session_state.chat_history)
        st.session_state.chat_history.append(response)

# Geçmişi göster
if st.session_state.chat_history:
    st.markdown("---")
    st.subheader("📜 Conversation with Marvin")
    for msg in st.session_state.chat_history:
        role = "You" if isinstance(msg, HumanMessage) else "Marvin"
        st.markdown(f"**{role}:** {msg.content}")
