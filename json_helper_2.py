from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

json_content = """{
    "name": "",
    "email" : "",
    "phone_1": "",
    "phone_2": "",
    "address": "",
    "city": "",
    "linkedin": "",
    "professional_experience_in_years": "",
    "highest_education": "",
    "is_fresher": "yes/no",
    "is_student": "yes/no",
    "skills": ["",""],
    "applied_for_profile": "",
    "education": [
        {
            "institute_name": "",
            "year_of_passing": "",
            "score": ""
        },
        {
            "institute_name": "",
            "year_of_passing": "",
            "score": ""
        }
    ],
    "professional_experience": [
        {
            "organisation_name": "",
            "duration": "",
            "profile": ""
        },
        {
            "organisation_name": "",
            "duration": "",
            "profile": ""
        }
    ]
    "memberships":"",
    "key_qualifications":"",
    "professinal_training":""
}"""

class InputData:
    @staticmethod
    def input_data(text):
        input = f"""Extract relevant information from the following resume text and fill the provided JSON template. Ensure all keys in the template are present in the output, even if the value is empty or unknown. If a specific piece of information is not found in the text, use 'Not provided' as the value.

        Resume text:
        {text}

        JSON template:
        {json_content}

        Instructions:
        1. Carefully analyse the resume text.
        2. Extract relevant information for each field in the JSON template.
        3. If a piece of information is not explicitly stated, make a reasonable inference based on the context.
        4. Ensure all keys from the template are present in the output JSON.
        5. Format the output as a valid JSON string.

        Output the filled JSON template only, without any additional text or explanations."""

        return input

    @staticmethod
    def llm():
        # Load the Pythia model and tokenizer
        model_name = "EleutherAI/pythia-1.4b"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        return model, tokenizer