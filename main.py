from fastapi import FastAPI
from qa import mrc

app = FastAPI()

@app.post("/")
def machine_reading_comprehension(context: str, question: str):
    response = mrc(context, question)

    return response

