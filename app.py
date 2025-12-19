import streamlit as st
from nltk.chat.util import Chat, reflections
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# ---------------- Rule-based chatbot pairs ----------------
pairs = [
    [r"(hi|hello|hey)", ["Hello! How can I help you today?"]],
    [r"(sad|upset|low|depressed)", ["I'm sorry you're feeling this way."]],
    [r"(stress|stressed|anxious)", ["Stress can affect anyone. Take it one step at a time."]],
    [r"(motivation|encourage|inspire)", ["You are doing better than you think. Keep going!"]],
    [r"(bye|exit)", ["Goodbye! Take care."]]
]

chatbot = Chat(pairs, reflections)

# ---------------- ML-based model ----------------
X = [
    "I am feeling sad",
    "I am stressed about exams",
    "I feel tired",
    "I need motivation",
    "I am worried about my job",
    "I have health issues"
]

y = ["emotion", "education", "health", "motivation", "career", "health"]

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X_vec, y)

# ---------------- Streamlit UI ----------------
st.title("Universal AI Chatbot")
st.write("Hello! I am your friendly AI chatbot. Type anything and I’ll try to help. Type 'bye' to exit.")

user_input = st.text_input("You:")

if user_input:
    response = chatbot.respond(user_input.lower())
    if response:
        st.write("Bot:", response)
    else:
        pred = model.predict(vectorizer.transform([user_input]))[0]
        st.write("Bot: I understand this relates to", pred)
