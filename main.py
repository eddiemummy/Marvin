import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.stores import InMemoryHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda

api_key = st.secrets["GOOGLE_GEMINI_KEY"]

model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=api_key,
    temperature=0.7,
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

if "store" not in st.session_state:
    st.session_state.store = {}

memory = InMemoryHistory()
chain = RunnableWithMessageHistory(
    RunnableLambda(lambda x: model.invoke([system_msg, HumanMessage(content=x["input"])])),
    lambda session_id: memory,
    input_messages_key="input",
    history_messages_key="messages",
)

user_input = st.text_input("You:", placeholder="What’s the point of anything, Marvin?")

if st.button("🔄 Ask Marvin") and user_input:
    response = chain.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": "marvin-session"}},
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
