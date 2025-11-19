from textwrap import dedent

import whisper
from google import genai
from google.genai import types
from google.genai.types import GenerateContentResponse
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

client = genai.Client()

def stt():
    model = whisper.load_model("turbo")
    result = model.transcribe("Hindi_a8atgEkE_Banking_Reema_SharmaRiya_Pal_combined.wav")
    print(result["text"])
    return result["text"]

def translate(text):
    # Second LLM call with 'toon_data'
    response: GenerateContentResponse = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=dedent(
            f"""
            You are Helpful Translation Assistant which can translate from Hindi to English.
            Given the Hindi Text below, your job is to translate now.\n
            ===============================================================================\n 
            {text}\n
            ===============================================================================\n
            Assistant:
            """
        ),
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        ),
    )
    print(response.text)

if __name__ == "__main__":
    translate(stt())