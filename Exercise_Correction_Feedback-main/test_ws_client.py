import asyncio
import websockets

async def test_ws():
    uri = "ws://localhost:8000/ws/pose"
    async with websockets.connect(uri) as websocket:
        while True:
            response = await websocket.recv()
            print("Received:", response)

asyncio.run(test_ws())
