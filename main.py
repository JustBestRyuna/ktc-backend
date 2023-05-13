from fastapi import FastAPI, UploadFile

from process_image.read_image import read_image

app = FastAPI()


@app.post("/read_image")
async def read_image_api(file: UploadFile):
    return read_image(file)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
