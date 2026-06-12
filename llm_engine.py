"""
llm_engine.py
=============
Module to handle contextual generation utilizing LLMs (Gemini) or an offline mock generator fallback.
"""
import os
import httpx

def generate_llm_response(prompt: str, api_key: str) -> str:
    """Sends a content generation request to Google Gemini API."""
    if not api_key:
        return "Error: Gemini API Key is missing."
        
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }
    
    try:
        response = httpx.post(url, headers=headers, json=payload, timeout=15.0)
        if response.status_code == 200:
            data = response.json()
            try:
                return data["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError) as e:
                return f"Error parsing Gemini response: {str(e)}\nResponse: {response.text}"
        else:
            return f"Error from Gemini API (Status {response.status_code}): {response.text}"
    except Exception as e:
        return f"Failed to connect to Gemini API: {str(e)}"

def generate_mock_response(query: str, documents: list, features: list, workflow: str) -> str:
    """Offline domain-specific mock generator for demonstrating Neurosymbolic RAG answers."""
    doc_titles = [d["doc"]["title"] for d in documents] if documents else ["No documents retrieved"]
    features_list = [f"`{x}`" for x in features]
    features_str = ", ".join(features_list) if features_list else "None"
    
    if workflow == "PHQ-9":
        return (
            f"**[MOCK LLM GENERATION - Set GEMINI_API_KEY in .env for real responses]**\n\n"
            f"### Clinical Context & Analysis\n"
            f"The patient presented with symptoms of: {features_str}. "
            f"Based on the clinical guidelines retrieved (including *{doc_titles[0]}*), there is a strong correlation "
            f"between sleep disturbances, childhood adverse experiences, and low mood.\n\n"
            f"### Action Plan & Workflow Checkpoints\n"
            f"1. **Initial Screening**: Screening protocols indicate active depressive mood indicators.\n"
            f"2. **Clinical Consultation**: Contextualized durational profile shows chronic issues requiring structured follow-up.\n"
            f"3. **Recommendation**: Schedule a DSM-5 structured diagnostic interview immediately to establish a definitive diagnosis before initiating guided CBT interventions."
        )
    elif workflow == "Software_Release":
        return (
            f"**[MOCK LLM GENERATION - Set GEMINI_API_KEY in .env for real responses]**\n\n"
            f"### Software Release Assessment\n"
            f"For release request: *\"{query}\"*, the active DevOps context includes build variables: {features_str}.\n\n"
            f"### Release Checklist Actions\n"
            f"1. **CI Build Status**: Build compile stage is active. Build validation requires clean linting reports.\n"
            f"2. **Security Controls**: Static application security testing (SAST/DAST) vulnerability scans must compile green.\n"
            f"3. **Next Steps**: Promote build to staging verification environment and prepare canary rollout telemetry checks once vulnerability scans are verified."
        )
    else:
        # Generic fallback
        docs_summary = "\n".join(f"- *{title}*" for title in doc_titles[:3])
        return (
            f"**[MOCK LLM GENERATION - Set GEMINI_API_KEY in .env for real responses]**\n\n"
            f"### RAG Context Summary\n"
            f"Found {len(documents)} relevant document(s) matching query: *\"{query}\"*.\n\n"
            f"**Session Context Tags**:\n"
            f"{features_str}\n\n"
            f"**Retrieved Sources**:\n"
            f"{docs_summary}\n\n"
            f"Please check your `.env` configuration file to input a real `GEMINI_API_KEY` to query Gemini-1.5-Flash live."
        )
