from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from app.analyzer import analyze_image

app = FastAPI(title="Image Analyzer API")

# Optional: CORS setup for future frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your frontend URL later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Image Analyzer API is running."}

@app.post("/analyze/")
async def analyze(file: UploadFile = File(...)):
    result = await analyze_image(file)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
