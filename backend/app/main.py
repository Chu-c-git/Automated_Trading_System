from fastapi import FastAPI


app = FastAPI(title="ATS Backend API")


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "ATS backend is running"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
