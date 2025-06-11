from pydantic import BaseModel

class PoseData(BaseModel):
    angle: float
    stage: str
    counter: int
