# pyright: reportPrivateImportUsage=false

from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
import os
from dotenv import load_dotenv
import json
import re
from google import genai
from google.genai import types
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

app = FastAPI()

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Make sure an API key is present in the .env file
if not GOOGLE_API_KEY:
    raise RuntimeError("Missing GOOGLE_API_KEY in environment.")

# Initialize Gemini client with API key
client = genai.Client(api_key=GOOGLE_API_KEY)

# Allowed classes
ALLOWED_DOC_TYPES = {"1040", "W2", "1099", "ID Card", "Handwritten note", "OTHER"}

# System prompt with few-shot prompting
SYSTEM_PROMPT = """You are an expert document classifier and data extractor.
Given the following PDF document, classify it into one of these categories:
"1040", "W2", "1099", "ID Card", "Handwritten note", "OTHER".
Note that handwritten notes are typically written on paper.
Also, extract the primary year the document was issued, or pertains to.
If no year at all is found, return "N/A" in the year field.
Do not consider expiration dates.

Output the result in a strict JSON format:
{
  "document_type": "[CLASSIFIED_TYPE]",
  "year": "[EXTRACTED_YEAR]"
}

Example for a 1099 from 2022:
{
  "document_type": "1099",
  "year": "2022"
}

Example for a handwritten note from 2019:
{
  "document_type": "Handwritten note",
  "year": "2019"
}

"""

@app.post("/classify")
async def schedule_classify_task(file: Optional[UploadFile] = File(None)):
    """Endpoint to classify a document into "w2", "1099int", etc"""

    # Make sure a PDF is input
    if file is None or not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")    
    
    try: 
        # Read the uploaded PDF file content
        file_content = await file.read()

        # Call the LLM with the defined system prompt and PDF input
        # Includes safety and hyperparameter settings
        response = await client.aio.models.generate_content(
            model="gemini-2.0-flash",
            contents = [
                SYSTEM_PROMPT, 
                types.Part.from_bytes(
                    data=file_content,
                    mime_type='application/pdf')
            ],
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=75,
                safety_settings=[
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                        threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                        threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
                        threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                    ),
                ]

            )
        )

        # Extract text from response
        if response and isinstance(response.text, str):
            raw_text = response.text.strip()
        else:
            raise HTTPException(status_code=500, detail="Unexpected response type from model.")
        
        # Remove JSON markdown code block wrappers
        if raw_text.startswith("```") and raw_text.endswith("```"):
            raw_text = re.sub(r"^```[a-z]*\n", "", raw_text)
            raw_text = re.sub(r"\n```$", "", raw_text)

        # Make sure the extracted text is not empty
        if not raw_text:
            raise HTTPException(status_code=500, detail="Model response was empty.")

        # Normalize unexpected quotes
        raw_text = raw_text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")

        # Parse the llm's JSON output 
        try:
            parsed_output = json.loads(raw_text)
            document_type = parsed_output.get("document_type", "OTHER")
            year = parsed_output.get("year", "N/A")
        except json.JSONDecodeError:
            # Fallback if Gemini doesn't return perfect JSON
            logging.warning(f"Warning: Gemini output was not perfect JSON: {raw_text}")
            # Attempt to extract from imperfect output
            type_match = re.search(r'["\']?document_type["\']?\s*[:=]\s*["\']?([\w\s]+)["\']?', raw_text)
            year_match = re.search(r'["\']?year["\']?\s*[:=]\s*["\']?(?P<year>\d{4}|N/A)["\']?', raw_text)
            if type_match and year_match:
                document_type = type_match.group(1)
                year = year_match.group(1)
            else:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to parse LLM response: invalid JSON and could not recover from output."
                )
                
        # Validate document_type against allowed types
        if document_type not in ALLOWED_DOC_TYPES:
            logging.warning(f"Warning: LLM returned an unexpected document_type: '{document_type}'. Defaulting to OTHER.")
            document_type = "OTHER"

        # Validate year format
        if not (year == "N/A" or (isinstance(year, str) and re.fullmatch(r"\d{4}", year))):
             logging.warning(f"LLM returned an invalid year format: '{year}'. Defaulting to N/A.")
             year = "N/A"

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to classify document: {e}")

    return {"document_type": document_type, "year": year}
