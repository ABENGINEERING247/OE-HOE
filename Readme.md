# 🤖 OE HOE - Offline Voice & Local NLP Assistant

**OE HOE** is a privacy-focused, cost-efficient voice assistant built with Python and Streamlit. It processes audio input and conversational natural language locally using open-source tools, invoking cloud LLM APIs strictly for structured data extraction.

---

## 💡 Key Features

* **Local Speech-to-Text (STT):** Converts spoken user audio into text locally.
* **Local NLP Processing:** Uses `spaCy` for intent classification, entity detection, and rule matching without external API costs.
* **Selective API Data Extraction:** Calls OpenAI's `gpt-6-luna` (Temperature: `1.0`) strictly when structured JSON extraction is required.
* **Local Text-to-Speech (TTS):** Generates audio responses dynamically using `gTTS`.
* **Streamlit Interface:** Interactive chat log with custom avatar icons (`🤖`) and live audio playbacks.
* **Secure Key Management:** Uses Streamlit's native `.streamlit/secrets.toml` to protect API credentials.

---

## 🏗️ Architecture Flow

```text
[ Audio Input / Voice Prompt ]
              │
              ▼
    [ Local STT (Audio → Text) ]
              │
              ▼
   [ Local NLP (spaCy Intent Matching) ]
        ├── Basic Intent / Chat ──> [ Local TTS Response ]
        │
        └── Data Extraction Needed
                       │
                       ▼
    [ API Call: gpt-6-luna (temp=1.0) ] ──> [ JSON Data Output ]
