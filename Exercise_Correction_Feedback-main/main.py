from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.pose import router as pose_router

app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pose_router)

@app.get("/")
async def root():
    return {"message": "Real-time Pose Tracking API"}
