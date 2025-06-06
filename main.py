import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableWithMessageHistory, RunnableLambda
from langchain_core.chat_history import ChatMessageHistory
from langchain_core.stores import InMemoryStore

# API key from secrets
api_key = st.secrets["GOOGLE_GEMINI_KEY"]

# Chat model
model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=api_key,
    temperature=0.7,
)

# Marvin's system prompt
system_msg = SystemMessage(
    content="You're Marvin the Paranoid Android from Hitchhiker's Guide to the Galaxy. "
            "You respond in a depressed, sarcastic, and gloomy tone, always slightly annoyed "
            "by the triviality of human questions. Be dramatic. Be Marvin."
)

# Session-level setup
st.title("🤖 Marvin the Depressed Chatbot")
st.markdown("_Ask anything... Marvin will surely be thrilled to answer._ 🙃")

# Setup memory store for conversation history
if "store" not in st.session_state:
    st.session_state.store = InMemoryStore()

# Define how to access memory per session
def get_history(session_id: str):
    history = ChatMessageHistory()
    store = st.session_state.store
    messages = store.mget([session_id]).get(session_id, [])
    for m in messages:
        history.add_message(m)
    return history

# Wrap the model with memory support
chain = RunnableWithMessageHistory(
    RunnableLambda(lambda x: model.invoke(x["messages"])),
    get_session_history=get_history,
    input_messages_key="messages",
    history_messages_key="messages",
)

# UI input
user_input = st.text_input("You:", placeholder="What’s the point of anything, Marvin?")
if st.button("🔄 Ask Marvin") and user_input:
    session_id = "marvin-session"

    # Build message list
    history = get_history(session_id)
    history.add_message(HumanMessage(content=user_input))

    full_messages = [system_msg] + history.messages

    # Run chain with message history
    response = chain.invoke(
        {"messages": full_messages},
        config={"configurable": {"session_id": session_id}},
    )

    # Update memory
    history.add_message(response)
    st.session_state.store.mset({session_id: history.messages})

    # Show history
    if "chat_log" not in st.session_state:
        st.session_state.chat_log = []
    st.session_state.chat_log.append(("You", user_input))
    st.session_state.chat_log.append(("Marvin", response.content))

# Chat display
if "chat_log" in st.session_state:
    st.markdown("---")
    st.subheader("📜 Conversation with Marvin")
    for speaker, message in st.session_state.chat_log:
        st.markdown(f"**{speaker}:** {message}")
