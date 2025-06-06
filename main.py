import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableWithMessageHistory, RunnableLambda
from langchain_core.stores import InMemoryStore
from langchain_core.chat_history import InMemoryChatMessageHistory

# 🔐 API Anahtarı
api_key = st.secrets["GOOGLE_GEMINI_KEY"]

# 🔧 Model
model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=api_key,
    temperature=0.7
)

# 🤖 Marvin kişiliği
system_msg = SystemMessage(
    content=(
        "You're Marvin the Paranoid Android from Hitchhiker's Guide to the Galaxy. "
        "You respond in a depressed, sarcastic, and gloomy tone, always slightly annoyed "
        "by the triviality of human questions. Be dramatic. Be Marvin."
    )
)

# Başlık
st.title("🤖 Marvin the Depressed Chatbot")
st.markdown("_Ask anything... Marvin will surely be thrilled to answer._ 🙃")

# 🧠 Memory store
if "store" not in st.session_state:
    st.session_state.store = InMemoryStore()

def get_memory(session_id: str):
    return InMemoryChatMessageHistory()

# 🔄 Chain
chain = RunnableWithMessageHistory(
    RunnableLambda(lambda x: model.invoke([system_msg] + x["messages"])),
    get_session_history=get_memory,
    input_messages_key="messages",
    history_messages_key="messages",
)

# 📝 Chat geçmişi
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

# 🔍 Giriş ve butonlar
user_input = st.text_input("You:", placeholder="What’s the point of anything, Marvin?")

col1, col2 = st.columns([1, 1])
ask_clicked = col1.button("🔄 Ask Marvin")
search_clicked = col2.button("🔍 Search")

# 💬 Marvin'e gönder
if (ask_clicked or search_clicked) and user_input:
    role = "You (Search)" if search_clicked else "You"
    response = chain.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config={"configurable": {"session_id": "marvin-session"}},
    )
    st.session_state.chat_log.append((role, user_input))
    st.session_state.chat_log.append(("Marvin", response.content))

# 🧾 Sohbet geçmişi
if st.session_state.chat_log:
    st.markdown("---")
    st.subheader("📜 Conversation with Marvin")
    for role, message in st.session_state.chat_log:
        st.markdown(f"**{role}:** {message}")
    st.markdown("<div style='height: 1px;' id='bottom'></div>", unsafe_allow_html=True)
    st.markdown(
        "<script>document.getElementById('bottom').scrollIntoView({ behavior: 'smooth' });</script>",
        unsafe_allow_html=True
    )
