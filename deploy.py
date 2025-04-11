import streamlit as st
import numpy as np
import os
import nltk
from nltk.stem import WordNetLemmatizer
import random
import json
import pickle
from tensorflow.keras.models import load_model
from datetime import datetime
import time
import base64
nltk.data.path.append(os.path.join(os.path.dirname(__file__), 'nltk_data'))

# Download NLTK data
# nltk.download('punkt', quiet=True)
# nltk.download('wordnet', quiet=True)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents
intents = json.loads(open('intents.json', encoding='utf-8').read())

# Load trained model and data
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

# Set page configuration
st.set_page_config(
    page_title="AI Chatbot Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to get base64 encoded image
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Custom CSS for styling (with default dark responsive theme)
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 6rem;
        max-width: 1000px;
    }
    
    /* Main background styling */
    .main {
        background-color: #1e1e1e;
        color: #f0f0f0;
    }
    
    /* Header styling */
    .header-container {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .header-container h1 {
        color: white;
        font-size: 2.5rem !important;
        margin: 0;
        font-weight: 700;
    }
    
    .header-container img {
        margin-right: 1rem;
    }
    
    /* Chat message styling */
    .stChatMessage {
        padding: 15px;
        border-radius: 20px !important;
        margin-bottom: 15px;
        max-width: 85%;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        animation: fadeIn 0.3s ease-in-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .stChatMessage [data-testid="stMarkdownContainer"] p {
        font-size: 16px;
        line-height: 1.5;
        margin: 0;
    }
    
    /* User message styling */
    .user-message {
        background: linear-gradient(135deg, #6e8efb 0%, #a777e3 100%);
        color: white;
        margin-left: auto;
        border-bottom-right-radius: 5px !important;
    }
    
    /* Bot message styling */
    .bot-message {
        background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
        color: #f0f0f0;
        border-bottom-left-radius: 5px !important;
    }
    
    /* Chat input styling */
    [data-testid="stChatInput"] {
        bottom: 20px;
        position: fixed;
        width: calc(100% - 21rem);
        padding: 10px;
        background: #2d3748;
        border: 1px solid #4a5568;
        border-radius: 50px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    }
    
    [data-testid="stChatInput"] {
        background: transparent !important;
    }
    
    [data-testid="stChatInput"] input {
        border-radius: 50px;
        padding-left: 20px;
        font-size: 16px;
        color: #f0f0f0;
    }
    
    /* Time stamp styling */
    .stChatMessage .stChatMessageContent small {
        opacity: 0.7;
        font-size: 12px;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #2d3748;
        border-right: 1px solid #4a5568;
        padding: 2rem 1rem;
        color: #f0f0f0;
    }
    
    .sidebar-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .sidebar-content {
        padding: 1rem;
        background: #1e1e1e;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        border-radius: 50px !important;
        padding: 0.5rem 2rem !important;
        border: none !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 10px rgba(0, 0, 0, 0.15) !important;
    }
    
    /* Typing indicator animation */
    .typing-indicator {
        display: flex;
        padding: 10px;
    }
    
    .typing-indicator span {
        height: 10px;
        width: 10px;
        background-color: #4b6cb7;
        border-radius: 50%;
        margin: 0 2px;
        display: inline-block;
        animation: bounce 1.5s infinite ease-in-out;
    }
    
    .typing-indicator span:nth-child(2) {
        animation-delay: 0.2s;
    }
    
    .typing-indicator span:nth-child(3) {
        animation-delay: 0.4s;
    }
    
    @keyframes bounce {
        0%, 80%, 100% { transform: translateY(0); }
        40% { transform: translateY(-10px); }
    }
    
    /* Badge styling */
    .badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 50px;
        font-size: 12px;
        font-weight: 600;
        margin-left: 10px;
        background-color: #4b6cb7;
        color: white;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .header-container h1 {
            font-size: 1.8rem !important;
        }
        
        [data-testid="stChatInput"] {
            width: calc(100% - 2rem);
        }
        
        .stChatMessage {
            max-width: 90%;
        }
    }
</style>
""", unsafe_allow_html=True)

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    
    if not return_list:
        return [{'intent': 'unknown', 'probability': '1.0'}]
    
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            
            # Handle special placeholders
            if '%TIME%' in result:
                current_time = datetime.now().strftime("%H:%M")
                result = result.replace('%TIME%', current_time)
            if '%DATE%' in result:
                current_date = datetime.now().strftime("%B %d, %Y")
                result = result.replace('%DATE%', current_date)
                
            break
    return result

# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-header">', unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=100)
    st.markdown('<h2 style="text-align: center;">Chatbot Settings</h2>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    chatbot_personality = st.selectbox(
        "Personality Mode",
        ("Friendly", "Professional", "Humorous", "Technical", "Supportive"),
        index=0
    )
    
    response_speed = st.slider(
        "Response Speed",
        min_value=0.5,
        max_value=2.0,
        value=1.0,
        step=0.1,
        help="Adjust how quickly the chatbot responds"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # st.markdown('<div class="sidebar-content" style="margin-top: 1rem;">', unsafe_allow_html=True)
    st.markdown("**About this bot**")
    st.markdown("""
    This AI assistant can help with:
    - General questions
    - Jokes and entertainment
    - Information lookup
    - Casual conversation
    - And much more!
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.experimental_rerun()
        
    # st.markdown('<div class="sidebar-content" style="margin-top: 1rem;">', unsafe_allow_html=True)
    st.markdown("**Chat Stats**")
    if "message_count" not in st.session_state:
        st.session_state.message_count = 0
    st.markdown(f"Messages: {st.session_state.message_count}")
    st.markdown(f"Session started: {datetime.now().strftime('%H:%M')}")
    st.markdown('</div>', unsafe_allow_html=True)

# Main app header
st.markdown(
    '''
    <div class="header-container">
        <img src="https://cdn-icons-png.flaticon.com/512/4712/4712035.png" width="60">
        <h1>AI Assistant</h1>
    </div>
    ''', 
    unsafe_allow_html=True
)

st.markdown('<p style="text-align: center; color: #f0f0f0;">Powered by Python, TensorFlow and Streamlit</p>', unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add welcome message
    welcome_messages = [
        "Hello! How can I help you today? 😊",
        "Hi there! What can I do for you? 🤖",
        "Greetings! I'm here to assist you. 💡",
        "Welcome! I'm ready to chat whenever you are. 🌟"
    ]
    st.session_state.messages.append({
        "role": "assistant",
        "content": random.choice(welcome_messages),
        "timestamp": datetime.now().strftime("%H:%M")
    })
    st.session_state.message_count = 1

# Create a container for chat messages
chat_container = st.container()

# Display chat messages
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            st.caption(message["timestamp"])

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Add user message to chat history
    timestamp = datetime.now().strftime("%H:%M")
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "timestamp": timestamp
    })
    st.session_state.message_count += 1
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
        st.caption(timestamp)
    
    # Simulate typing effect
    with st.chat_message("assistant"):
        typing_placeholder = st.empty()
        typing_placeholder.markdown(
            """
            <div class="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Get bot response with a delay based on response speed
        ints = predict_class(prompt)
        response = get_response(ints, intents)
        
        # Add the tag as a badge if confidence is high
        # if float(ints[0]['probability']) > 0.7:
        #     tag = ints[0]['intent']
        #     # Don't add badge for unknown intent
        #     if tag != "unknown":
        #         response += f"<span class='badge'>{tag}</span>"
        
        # Personality adjustments
        if chatbot_personality == "Professional":
            response = response.replace("!", ".").replace("😊", "").replace("🤖", "")
            response = response.replace("Aww", "Thank you").replace("LOL", "I appreciate that")
        elif chatbot_personality == "Humorous":
            funny_emojis = ["😂", "🤣", "😆", "😜", "🙃"]
            response = random.choice(funny_emojis) + " " + response
            if not any(em in response for em in funny_emojis):
                response += " " + random.choice(funny_emojis)
        elif chatbot_personality == "Technical":
            response = response.replace("simple", "optimized").replace("help", "assist")
            if "!" in response:
                response = response.replace("!", ".")
            if not "data" in response and random.random() > 0.7:
                response += " Would you like more technical details?"
        elif chatbot_personality == "Supportive":
            supportive_phrases = [
                " I'm here for you.",
                " How does that sound to you?",
                " Let me know if there's anything else you need.",
                " I'm happy to help further."
            ]
            response += random.choice(supportive_phrases)
        
        # Simulate typing with delay based on response length and speed setting
        typing_time = min(len(response) * 0.01 / response_speed, 3)
        time.sleep(typing_time)
        
        # Replace typing indicator with response
        typing_placeholder.empty()
        st.markdown(response, unsafe_allow_html=True)
        timestamp = datetime.now().strftime("%H:%M")
        st.caption(timestamp)
    
    # Add assistant response to chat history
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "timestamp": timestamp
    })
    st.session_state.message_count += 1

# Footer
st.markdown(
    '''
    <div style="position: fixed; bottom: 3px; width: calc(100% - 21rem); text-align: center; padding: 10px;">
        <p style="color: #666; font-size: 12px;">© 2025 AI Assistant | Powered by TensorFlow & Streamlit</p>
    </div>
    ''',
    unsafe_allow_html=True
)