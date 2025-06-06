import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableWithMessageHistory, RunnableLambda
from langchain_core.chat_history import InMemoryChatMessageHistory

# Streamlit Secrets
api_key = st.secrets["GOOGLE_GEMINI_KEY"]

# LLM Model
model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=api_key,
    temperature=0.7,
)

# Marvin System Prompt
system_msg = SystemMessage(
    content=(
        "You're Marvin the Paranoid Android from Hitchhiker's Guide to the Galaxy. "
        "You respond in a depressed, sarcastic, and gloomy tone, always slightly annoyed "
        "by the triviality of human questions. Be dramatic. Be Marvin."
    )
)

# Streamlit UI
st.title("🤖 Marvin the Depressed Chatbot")
st.markdown("_Ask anything... Marvin will surely be thrilled to answer._ 🙃")

# Message history storage
if "stores" not in st.session_state:
    st.session_state.stores = {}

def get_history(session_id: str):
    if session_id not in st.session_state.stores:
        st.session_state.stores[session_id] = InMemoryChatMessageHistory()
    return st.session_state.stores[session_id]

# LCEL chain with message history
chain = RunnableWithMessageHistory(
    RunnableLambda(lambda x: model.invoke([system_msg] + x["messages"])),
    get_session_history=get_history,
    input_messages_key="messages",
    history_messages_key="messages",
)

session_id = "marvin-session"
user_input = st.text_input("You:", placeholder="Is there any point in asking, really?")

if st.button("🔄 Ask Marvin") and user_input:
    response = chain.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config={"configurable": {"session_id": session_id}},
    )

    if "chat_log" not in st.session_state:
        st.session_state.chat_log = []
    st.session_state.chat_log.append(("You", user_input))
    st.session_state.chat_log.append(("Marvin", response.content))

if "chat_log" in st.session_state:
    st.markdown("---")
    st.subheader("📜 Conversation with Marvin")
    for speaker, message in st.session_state.chat_log:
        st.markdown(f"**{speaker}:** {message}")
