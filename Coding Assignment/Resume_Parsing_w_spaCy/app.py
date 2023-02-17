
from fastapi import FastAPI, UploadFile, File, HTTPException
from resume import readPDF
app = FastAPI()

@app.post("/pdf")
async def create_pdf(pdf: UploadFile):
    
    # Do some processing with the file
    skills,educations = readPDF(pdf.file)
    dict = {"skills":skills,"educations":educations}
    return dict
   