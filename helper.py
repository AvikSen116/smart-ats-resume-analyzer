import google.generativeai as genai # type: ignore
import PyPDF2 as pdf # type: ignore
import json
import re

def configure_genai(api_key):
    """Configure the Generative AI API with error handling."""
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        raise Exception(f"Failed to configure Generative AI: {str(e)}")

def get_gemini_response(prompt):
    """Generate a response using Gemini and ensure strict JSON compliance."""
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)

        # Ensure response is not empty
        if not response or not response.text:
            raise Exception("Empty response received from Gemini")

        # Debugging: Print raw response (uncomment if needed)
        # print("RAW RESPONSE:", response.text)

        # Extract JSON directly
        try:
            response_json = json.loads(response.text)
            if not isinstance(response_json, dict):
                raise ValueError("Response is not a valid JSON object")
            return response_json  # Already a dictionary
        except json.JSONDecodeError:
            # If response is not valid JSON, attempt to extract JSON-like content
            json_pattern = r'\{.*\}'
            match = re.search(json_pattern, response.text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    raise Exception("Could not extract valid JSON from response")
            else:
                raise Exception("Gemini response is not in JSON format")

    except Exception as e:
        raise Exception(f"Error generating response: {str(e)}")

def extract_pdf_text(uploaded_file):
    """Extract text from PDF with enhanced error handling."""
    try:
        reader = pdf.PdfReader(uploaded_file)
        if len(reader.pages) == 0:
            raise Exception("PDF file is empty")

        text = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)

        if not text:
            raise Exception("No text could be extracted from the PDF")

        return " ".join(text)

    except Exception as e:
        raise Exception(f"Error extracting PDF text: {str(e)}")

def prepare_prompt(resume_text, job_description):
    """Prepare the input prompt with strict JSON enforcement."""
    if not resume_text or not job_description:
        raise ValueError("Resume text and job description cannot be empty")

    prompt_template = """
    You are an expert ATS (Applicant Tracking System) evaluator.
    Analyze the following resume against the given job description.

    Resume:
    {resume_text}

    Job Description:
    {job_description}

    Return ONLY a JSON response in this exact format:
    {{
        "JD Match": "percentage between 0-100",
        "MissingKeywords": ["keyword1", "keyword2", ...],
        "Profile Summary": "Detailed feedback for resume improvement"
    }}
    Do not include any extra text, explanation, or commentary.
    """

    return prompt_template.format(
        resume_text=resume_text.strip(),
        job_description=job_description.strip()
    )
