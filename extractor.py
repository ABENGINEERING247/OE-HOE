import json
import streamlit as st
from openai import OpenAI


def get_openai_client():
  """Initialize OpenAI client using Streamlit secrets."""
  if "OPENAI_API_KEY" in st.secrets:
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
  elif "api_keys" in st.secrets and "openai" in st.secrets["api_keys"]:
    return OpenAI(api_key=st.secrets["api_keys"]["openai"])
  else:
    st.error("API Key missing! Please set OPENAI_API_KEY in secrets.toml.")
    return None


def extract_structured_data(user_text: str) -> dict:
  """Calls API strictly for structured entity/data extraction using gpt-6-luna at temp 1.0."""
  client = get_openai_client()
  if not client:
    return {"error": "API Key configuration missing."}

  prompt = f"""
    Extract structured entity information from the following text.
    Return ONLY a valid JSON object with key fields found (e.g., intent, entities, dates, numbers, action_items).

    Text: "{user_text}"
    """

  try:
    settings = st.secrets.get("extraction_settings", {})
    model_name = settings.get("model", "gpt-6-luna")
    temp = float(settings.get("temperature", 1.0))

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a structured data extraction engine. Output raw"
                    " JSON only."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=temp,
        response_format={"type": "json_object"},
    )
    extracted_content = response.choices[0].message.content
    return json.loads(extracted_content)
  except Exception as e:
    return {"error": f"API extraction failed: {str(e)}"}
