import streamlit as st
import nltk

#app.py
import streamlit as st
from datetime import datetime

# -------------------------
# Function to initialize session state
# -------------------------
def init_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # Store all messages
    if "input_box" not in st.session_state:
        st.session_state.input_box = ""  # Store current input

# -------------------------
# Function to handle sending message
# -------------------------
def send_message():
    user_input = st.session_state.input_box.strip()
    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({"sender": "You", "message": user_input, "time": datetime.now().strftime("%H:%M")})
        
        # Generate bot reply (basic example, replace with your AI logic)
        bot_reply = f"🌸 Hello! You said: '{user_input}'"
        st.session_state.chat_history.append({"sender": "Bot", "message": bot_reply, "time": datetime.now().strftime("%H:%M")})
        
        # Clear input box
        st.session_state.input_box = ""

# -------------------------
# Main app
# -------------------------
def main():
    st.set_page_config(page_title="Universal AI Chatbot", page_icon="🤖")
    st.title("🌸 Universal AI Chatbot")
    st.write("Hello! I am your friendly AI chatbot. You can ask me anything—from counseling to general queries. 😊")

    # Initialize session state
    init_session_state()

    # Display chat history
    for chat in st.session_state.chat_history:
        if chat["sender"] == "You":
            st.markdown(f"**You ({chat['time']}):** {chat['message']}")
        else:
            st.markdown(f"**Bot ({chat['time']}):** {chat['message']}")

    # Input box for new messages
    st.text_input("You:", key="input_box", on_change=send_message)

if __name__ == "__main__":
    main()










# ---------------- NLTK setup ----------------
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.chat.util import Chat, reflections
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# ---------------- Rule-based chatbot ----------------
pairs = [
    [r"(hi|hello|hey)", ["Hello! I’m your friendly AI assistant. How can I help you today? 😊"]],
    [r"(how are you|how do you do)", ["I’m doing great! Thanks for asking. How about you?"]],
    [r"(sad|upset|low|depressed)", ["I’m here to listen. It’s okay to feel this way sometimes."]],
    [r"(stress|stressed|anxious)", ["Take a deep breath. You’re doing your best, and that’s enough."]],
    [r"(motivation|encourage|inspire)", ["Keep going! Every small step counts. You can do it!"]],
    [r"(bye|exit)", ["Goodbye! Take care of yourself and stay positive! 🌸"]],
    [r"(why is your chatbot not working|why not responding|issues with chatbot)",
     ["I try my best to answer, but sometimes I may not understand everything perfectly. "
      "This could be due to my programming or dataset limitations. I’m always learning!"]],
    [r"(what can you do|what are your abilities)",
     ["I can chat with you politely, provide counseling advice, answer general questions, "
      "and explain my limitations if needed."]],
    [r"(who made you|who created you)",
     ["I was created by a talented student as a Capstone AI project."]],
]

chatbot = Chat(pairs, reflections)

# ---------------- ML-based domain model ----------------
X = [
    "I am feeling sad",
    "I am stressed about exams",
    "I feel tired",
    "I need motivation",
    "I am worried about my job",
    "I have health issues",
    "Why is my chatbot not working",
    "How can I use this chatbot",
    "What can you do",
    "Who created you",
    "How do I interact with you",
    "What are your limitations",
    "Tell me about AI"
]

y = [
    "emotion", "education", "health", "motivation", "career", "health",
    "self_query", "self_query", "self_query", "self_query", "self_query", "self_query", "general"
]

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X_vec, y)

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Universal AI Chatbot")
st.title("🌸 Universal AI Chatbot")
st.write("Hello! I am your friendly AI chatbot. You can ask me anything—from counseling to general queries. 😊")

# Initialize chat history
if "history" not in st.session_state:
    st.session_state.history = []

def respond(user_input):
    # Rule-based first
    response = chatbot.respond(user_input.lower())
    if response:
        return response
    else:
        # ML-based fallback
        pred = model.predict(vectorizer.transform([user_input]))[0]
        if pred == "self_query":
            return ("I try my best to answer, but sometimes I may not understand everything perfectly. "
                    "This may be due to my programming or dataset limitations. I’m always improving!")
        elif pred == "general":
            return "That’s an interesting question! I may not have a perfect answer, but I’m always learning."
        else:
            return f"I understand this relates to {pred}. I’m here to assist you politely and helpfully."

# Input box with automatic clear
user_input = st.text_input("You:", key="input_box")

if user_input:
    bot_response = respond(user_input)
    st.session_state.history.append(("You", user_input))
    st.session_state.history.append(("Bot", bot_response))
    
    # Clear input after submitting
    st.session_state.input_box = ""

# Display chat history
for speaker, message in st.session_state.history:
    if speaker == "You":
        st.markdown(f"**{speaker}:** {message}")
    else:
        st.markdown(f"**{speaker}:** {message}")
