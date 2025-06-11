import React, { useEffect, useRef, useState } from "react";
import { useParams } from "react-router-dom";
import './Record.module.css'; // Import the custom CSS file

const Record = () => {
  const { id } = useParams<{ id: string }>();
  const [poseData, setPoseData] = useState({
    angle: "--",
    stage: "--",
    counter: "--",
    message: "Waiting...",
  });
  const [frame, setFrame] = useState<string | null>(null);
  const socketRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!id) return;

    const socketUrl = `ws://localhost:8000/ws/pose/${id}`;
    socketRef.current = new WebSocket(socketUrl);

    socketRef.current.onopen = () => {
      console.log(`✅ WebSocket Connected: ${socketUrl}`);
    };

    socketRef.current.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        setPoseData({
          angle: data.angle?.toFixed(2) || "--",
          stage: data.stage || "--",
          counter: data.counter || "--",
          message: data.message || "Waiting...",
        });

        if (data.frame) {
          setFrame(`data:image/jpeg;base64,${data.frame}`);
        }
      } catch (error) {
        console.error("❌ Error parsing WebSocket message:", error);
      }
    };

    socketRef.current.onerror = (error) => {
      console.error("🔥 WebSocket Error:", error);
    };

    socketRef.current.onclose = () => {
      console.log(`❌ WebSocket Disconnected: ${socketUrl}, reconnecting...`);
      setTimeout(() => {
        socketRef.current = new WebSocket(socketUrl);
      }, 2000);
    };

    return () => socketRef.current?.close();
  }, [id]);

  return (
    <div className="record-container">
      <h1 className="record-header">
        Live Pose Tracking <span>({id})</span>
      </h1>

      <div className="record-message">
        {poseData.message}
      </div>

      <div className="record-frame-container">
        {frame ? (
          <img
            src={frame}
            alt="Pose Frame"
            className="record-frame"
          />
        ) : (
          <p className="record-placeholder">Waiting for camera feed...</p>
        )}
      </div>

      <div className="record-data-container">
        <p><span className="record-data-title">Angle:</span> {poseData.angle}°</p>
        <p><span className="record-data-label">Stage:</span> {poseData.stage}</p>
        <p><span className="record-data-label">Count:</span> {poseData.counter}</p>
      </div>
    </div>
  );
};

export default Record;