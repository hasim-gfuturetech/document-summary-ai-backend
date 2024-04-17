import PyPDF2
from transformers import pipeline
import torch
def read_pdf(file_path):
    text = ""
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfFileReader(file)
        num_pages = reader.numPages
        for page_num in range(num_pages):
            page = reader.getPage(page_num)
            text += page.extractText()
    return text


# file_path = "Brochure.pdf"
# pdf_text = read_pdf(file_path)
# pdf_text = "Because I was only nine years old when I started, the lessons my rich dad taught me were simple. And when it was all said and done, there were only six main lessons, repeated over 30 years. This book is about those six lessons, put as simply as possible, just as simply as my rich dad put forth those lessons to me. The lessons are meant not to be answers, but guideposts that will assist you and your children to grow wealthier no matter what happens in a world of increasing change and uncertainty."
# with open('result.txt', 'w', encoding='utf-8') as f:
#     f.write(str(pdf_text))
# with open('output.txt', 'r', encoding='utf-8') as f:
#     pdf_text = f.read()
# print(type(pdf_text))

print('reading...')
# facebook/bart-large-cnn
# summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
# result = summarizer(str(pdf_text), truncation=True, max_length=1024, min_length=30, do_sample=False)

hf_name = 'pszemraj/led-large-book-summary'

def processing(file):
    file_content = ""
    reader = PyPDF2.PdfFileReader(file)
    num_pages = reader.numPages
    for page_num in range(num_pages):
            page = reader.getPage(page_num)
            file_content += page.extractText()

    summarizer = pipeline(
        "summarization",
        hf_name,
        device=0 if torch.cuda.is_available() else -1,
    )
    result = summarizer(
        file_content,
        min_length=16,
        max_length=1024,
        no_repeat_ngram_size=3,
        encoder_no_repeat_ngram_size=3,
        repetition_penalty=3.5,
        num_beams=4,
        early_stopping=True,
    )
    return result



# print('almost done...')

# print("result:", result)
