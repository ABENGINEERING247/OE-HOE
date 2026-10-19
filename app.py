import io
from gtts import gTTS
import spacy
import speech_recognition as sr
import streamlit as st

from extractor import extract_structured_data

# Page Setup
st.set_page_config(
    page_title="OE HOE - Offline Voice & NLP Assistant",
    page_icon="🤖",
    layout="wide",
)


# Load local spaCy NLP model once
@st.cache_resource
def load_nlp_model():
  return spacy.load("en_core_web_sm")


nlp = load_nlp_model()

# Session State Initialization
if "chat_history" not in st.session_state:
  st.session_state.chat_history = []
if "extracted_records" not in st.session_state:
  st.session_state.extracted_records = []


def local_intent_analyzer(text: str):
  """Analyzes intent locally using spaCy tokenization and lemmatization."""
  doc = nlp(text.lower())
  lemmas = [token.lemma_ for token in doc]

  if any(word in lemmas for word in ["extract", "parse", "save", "record"]):
    return "API_EXTRACTION_REQUIRED"
  elif any(word in lemmas for word in ["hello", "hi", "hey", "greetings"]):
    return "GREETING"
  elif any(word in lemmas for word in ["clear", "reset"]):
    return "RESET"
  else:
    return "GENERAL_CHAT"


def text_to_speech(text: str):
  """Converts text response to audio stream using gTTS."""
  tts = gTTS(text=text, lang="en")
  fp = io.BytesIO()
  tts.write_to_fp(fp)
  fp.seek(0)
  return fp


# UI Layout
st.title("🤖 OE HOE Assistant")
st.caption(
    "Local Speech, NLP & Audio Pipeline. Uses `gpt-6-luna` (temp=1.0) strictly"
    " for data extraction."
)

col1, col2 = st.columns([1, 1])

with col1:
  st.subheader("🎤 Voice & Text Input")

  audio_file = st.file_uploader(
      "Upload audio prompt (.wav)", type=["wav"], help="Upload voice input"
  )
  text_input = st.text_input("Or type prompt here:")

  user_prompt = ""

  if audio_file:
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
      audio_data = recognizer.record(source)
      try:
        user_prompt = recognizer.recognize_google(audio_data)
        st.success(f"Recognized: **{user_prompt}**")
      except Exception:
        st.error("Could not transcribe audio locally.")

  if text_input:
    user_prompt = text_input

  if st.button("Submit / Process", type="primary"):
    if user_prompt:
      # Step 1: Local NLP Processing
      intent = local_intent_analyzer(user_prompt)
      bot_response = ""

      if intent == "GREETING":
        bot_response = (
            "Hello! I am OE HOE, ready to listen and process your requests."
        )
      elif intent == "RESET":
        st.session_state.chat_history = []
        st.session_state.extracted_records = []
        bot_response = "Session reset complete."
      elif intent == "API_EXTRACTION_REQUIRED":
        st.info(
            "Triggering `gpt-6-luna` (temp=1.0) for structured data extraction..."
        )
        extracted_data = extract_structured_data(user_prompt)
        st.session_state.extracted_records.append(extracted_data)
        bot_response = (
            "I have extracted the structured data using gpt-6-luna and updated"
            " your record table."
        )
      else:
        bot_response = (
            f"Local NLP recognized your query: '{user_prompt}'. No API call"
            " was required."
        )

      # Store Chat
      st.session_state.chat_history.append({
          "user": user_prompt,
          "bot": bot_response,
          "intent": intent,
      })

      # Generate Audio Response
      audio_stream = text_to_speech(bot_response)
      st.audio(audio_stream, format="audio/mp3", autoplay=True)

with col2:
  st.subheader("💬 OE HOE Logs & Chat History")

  # Show Extracted API Data
  if st.session_state.extracted_records:
    st.markdown("### 📊 Extracted Data via API (`gpt-6-luna`)")
    st.json(st.session_state.extracted_records)

  # Show Chat History with Custom Chatbot Icon ("🤖")
  st.markdown("### 📜 Conversation Log")
  for item in reversed(st.session_state.chat_history):
    with st.chat_message("user", avatar="🧑‍💻"):
      st.write(item["user"])
    with st.chat_message("assistant", avatar="🤖"):
      st.write(item["bot"])
      st.caption(f"Intent: `{item['intent']}`")
