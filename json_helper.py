from langchain_ollama import OllamaLLM
import tensorflow as tf

json_content = """{
    "name": "",
    "email_1": "",
    "phone_1": "",
    "country": "",
    "professional_experience_in_years": "",
    "job_title": "",
    "nationality": "",
    "date_of_birth": "",
    "relationship_status": "Single/Married/Prefer not to Mention",
    "languages": ["",""],
    "drivers_license": "Yes/No",
    "education": [
        {
            "university_name": "",
            "degree_name": "",
            "gradution_date": "",
            "country": ""
        }
    ],
    "professional_experience": [
        {
            "organisation_name": "",
            "position": "",
            "start_date": "",
            "end_date": "",
            "country": "",
            "duties": ["",""],
            "projects": ["",""]
        }
    ],
    "memberships": "",
    "key_qualifications": "",
    "professinal_training": ""
}"""

class InputData:
    @staticmethod
    def input_data(text):
        input = f"""
        Resume text chunk:
        {text}

        JSON template:
        {json_content}

        Extract information from this resume chunk following these rules:
        1. Fill only the information explicitly present in this chunk
        2. Mark missing fields as 'Not provided'
        3. For list fields (education, experience), add only new entries
        4. Never repeat existing information from previous chunks
        5. Format dates as YYYY-month or "Present" if it is mentioned
        6. For companies, aggregate all roles/duties/projects
        7. For key_qualification just show their qualifications with no extra data (year, name).
        8. Output only JSON without comments

        Current chunk context: This is part of a larger resume. 
        Some fields might be split across chunks. Only add new information.
        """
        return input


    @staticmethod
    def llm():
        try:
            # Check if GPU is available
            physical_devices = tf.config.list_physical_devices('GPU')
            if len(physical_devices) > 0:
                tf.config.experimental.set_memory_growth(physical_devices[0], True)

            # Initialize Ollama LLM
            llm = OllamaLLM(model="llama3")
            return llm
        except Exception as e:
            print(f"Error initializing LLM: {e}")
            return None
