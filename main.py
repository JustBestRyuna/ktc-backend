from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from process_image.read_image import read_image, give_points

app = FastAPI()

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/read_image")
async def read_image_api(file: UploadFile):
    return read_image(file)


@app.post("/give_points")
async def give_points_api(file: UploadFile):
    return max(give_points(file))


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
