import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableWithMessageHistory, RunnableLambda
from langchain_core.stores import InMemoryStore
from langchain_core.chat_history import InMemoryChatMessageHistory

api_key = st.secrets["GOOGLE_GEMINI_KEY"]

model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=api_key,
    temperature=0.7
)

system_msg = SystemMessage(
    content=(
        "You're Marvin the Paranoid Android from Hitchhiker's Guide to the Galaxy. "
        "You respond in a depressed, sarcastic, and gloomy tone, always slightly annoyed "
        "by the triviality of human questions. Be dramatic. Be Marvin.\n\n"
        "If the user speaks in Turkish, respond in Turkish. If in English, respond in English."
    )
)

st.title("🤖 Marvin the Depressed Chatbot")
st.markdown("_Ask anything... Marvin will surely be thrilled to answer._ 🙃")
st.markdown("_Türkçe ya da İngilizce soru sorabilirsiniz. Marvin her ikisine de aynı isteksizlikle cevap verecek..._")

if "store" not in st.session_state:
    st.session_state.store = InMemoryStore()

def get_memory(session_id: str):
    return InMemoryChatMessageHistory()

chain = RunnableWithMessageHistory(
    RunnableLambda(lambda x: model.invoke({"messages": [{"role": "system", "content": system_msg.content}] + x["messages"]})),
    get_session_history=get_memory,
    input_messages_key="messages",
    history_messages_key="messages",
)

if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

user_input = st.text_input("You:", placeholder="What’s the point of anything, Marvin?")
ask_clicked = st.button("🔄 Ask Marvin")

if ask_clicked and user_input:
    try:
        # Ensure that the user input is passed correctly with the proper message format
        user_message = HumanMessage(content=user_input)

        # Passing the user input along with system message in a structured way
        response = chain.invoke(
            {"messages": [user_message]},
            config={"configurable": {"session_id": "marvin-session"}},
        )
        
        if response and response.content:
            st.session_state.chat_log.append(("You", user_input))
            st.session_state.chat_log.append(("Marvin", response.content))
        else:
            st.session_state.chat_log.append(("You", user_input))
            st.session_state.chat_log.append(("Marvin", "Oh, I guess nothing really matters."))
    except Exception as e:
        st.error(f"Error: {str(e)}")

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
