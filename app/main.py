from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.analyzer import analyze_image

app = FastAPI(title="Image Analyzer API")

# CORS setup (allow all for now, can be restricted later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Image Analyzer API is running."}

@app.post("/analyze/")
async def analyze(file: UploadFile = File(...)):
    try:
        print(f"📥 Received file: {file.filename}")
        result = await analyze_image(file)
        print("✅ Image analysis complete.")
        return JSONResponse(content=result)
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Optional testing endpoint to verify file upload alone
@app.post("/test/")
async def test_upload(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        print(f"📦 Test upload received: {file.filename}, size: {len(contents)} bytes")
        return {"filename": file.filename, "size": len(contents)}
    except Exception as e:
        print(f"❌ Error during test upload: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)

