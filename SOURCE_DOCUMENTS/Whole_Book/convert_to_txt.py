import PyPDF2
import os

def convert_pdf_to_txt(pdf_path, txt_path):
    with open(pdf_path, 'rb') as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ''
        for page_num in range(reader.numPages):
            page = reader.getPage(page_num)
            text += page.extractText()
    
    with open(txt_path, 'w', encoding='utf-8') as txt_file:
        txt_file.write(text)

# Usage
files = os.listdir("/Users/rdongare/Desktop/Local_GPT/Source_Documents/Whole_Book")
for file in files:
    filename = file.split(".")[0]
    if filename:
        pdf_path = file
        txt_path = f'{filename}.txt'
        convert_pdf_to_txt(pdf_path, txt_path)
