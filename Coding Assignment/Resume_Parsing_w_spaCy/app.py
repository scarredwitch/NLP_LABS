from fastapi import FastAPI, File, UploadFile
import pandas as pd
import requests
from resume import readPDF
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/pdf_to_dataframe")
async def pdf_to_dataframe(pdf: UploadFile):
    response = readPDF(pdf.file)
    return response