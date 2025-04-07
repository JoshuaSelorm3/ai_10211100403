import PyPDF2
with open("data/student_policy.pdf", "rb") as file:
    pdf = PyPDF2.PdfReader(file)
    text = pdf.pages[0].extract_text()
    print(text)