from fastapi import FastAPI

app = FastAPI(
    title="Test Server", version="0.0.0", description="Testing crud operation"
)


@app.get("/")
async def root():
    return {"hello world"}
