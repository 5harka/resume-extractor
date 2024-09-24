from pdfminer.high_level import extract_text
from json_helper import InputData as input
import tensorflow as tf

def extract_text_from_pdf(pdf_path):
    return extract_text(pdf_path)

text = extract_text_from_pdf(r'C:\Users\Sharmarke.Kadir\Downloads\Example_Resume_Jane_Smith.pdf')

llm = input.llm()

data = llm.invoke(input.input_data(text))
print(data)

