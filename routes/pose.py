# from fastapi import APIRouter, WebSocket, WebSocketDisconnect, FastAPI
# import asyncio
# import json
# from services.pose_tracking import detect_pose  # Ensure this function returns a generator/async stream

# router = APIRouter()
# app = FastAPI()

# # Store active connections
# connections = {}

# @router.websocket("/ws/pose/{exercise}")
# async def websocket_exercise(websocket: WebSocket, exercise: str):
#     await websocket.accept()
#     print(f"Connected to WebSocket for {exercise}")
#     try:
#         # Run the pose detection asynchronously and send data
#         async for pose_data in detect_pose(exercise):
#             await websocket.send_text(json.dumps(pose_data))  # Send the data as JSON
#     except WebSocketDisconnect:
#         print(f"Connection closed for {exercise}")

# # WebSocket endpoint for general pose detection
# @router.websocket("/ws/pose")
# async def websocket_main(websocket: WebSocket):
#     await websocket.accept()
#     print("Connected to WebSocket for general pose detection")
#     try:
#         # Run the pose detection asynchronously and send data
#         async for pose_data in detect_pose("general"):
#             await websocket.send_text(json.dumps(pose_data))  # Send the data as JSON
#     except WebSocketDisconnect:
#         print("Connection closed for general pose detection")

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, FastAPI
import asyncio
import json
from services.pose_tracking import detect_pose  # Ensure this function returns a generator/async stream

router = APIRouter()
app = FastAPI()

# Store active connections
connections = {}

# @router.websocket("/ws/pose/{exercise}")
# async def websocket_exercise(websocket: WebSocket, exercise: str):
#     await websocket.accept()
#     print(f"Connected to WebSocket for {exercise}")
#     try:
#         # Run the pose detection asynchronously and send data
#         async for pose_data in detect_pose(exercise,path=None):
#             await websocket.send_text(json.dumps(pose_data))  # Send the data as JSON
#     except WebSocketDisconnect:
#         print(f"Connection closed for {exercise}")


@router.websocket("/ws/pose/{exercise}")
async def websocket_exercise(websocket: WebSocket, exercise: str):
    await websocket.accept()
    print(f"Connected to WebSocket for {exercise}")

    try:
        async for pose_data in detect_pose(exercise):  # ✅ Now works with async for!
            await websocket.send_text(json.dumps(pose_data))  
    except WebSocketDisconnect:
        print(f"Connection closed for {exercise}")



# WebSocket endpoint for general pose detection
@router.websocket("/ws/pose")
async def websocket_main(websocket: WebSocket):
    await websocket.accept()
    print("Connected to WebSocket for general pose detection")
    try:
        # Run the pose detection asynchronously and send data
        async for pose_data in detect_pose("general"):
            await websocket.send_text(json.dumps(pose_data))  # Send the data as JSON
    except WebSocketDisconnect:
        print("Connection closed for general pose detection")

app.include_router(router)
