from fastapi import APIRouter, WebSocket
import asyncio
import json
from services.pose_tracking import detect_pose  # Import pose detection function

router = APIRouter()

@router.websocket("/ws/pose")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        async for pose_data in detect_pose():
            await websocket.send_text(json.dumps(pose_data))
            await asyncio.sleep(0.05)
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        try:
            await websocket.close()
        except RuntimeError:
            pass  # Avoid double-close error
