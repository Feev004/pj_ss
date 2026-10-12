"use client";

import React, { useState, useEffect, useRef } from "react";

interface VideoItem {
  filename: string;
  size_mb: number;
  created_at: string;
}

interface SystemStatus {
  status: string;
  models: {
    accident_model: boolean;
    vehicle_model: boolean;
  };
  config: {
    enable_telegram: boolean;
    accident_conf_threshold: number;
    track_conf: number;
    process_every_n: number;
    clahe_enabled: boolean;
  };
}

interface IncidentLogItem {
  frame: number;
  time: number;
  box: number[];
  confidence: number;
  img: string;
  label: string;
  vehicle_counts?: Record<string, number>;
}

interface EvaluationMetrics {
  TP: number;
  FP: number;
  FN: number;
  TN: number;
  precision: number;
  recall: number;
  f1: number;
  total_gt: number;
  total_pred: number;
  iou_threshold: number;
  mAP: number;
  avg_inference_ms: number | null;
}

export default function Home() {
  const [activeVideo, setActiveVideo] = useState<string>("");
  const [videoList, setVideoList] = useState<VideoItem[]>([]);
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [isBackendOnline, setIsBackendOnline] = useState<boolean>(false);

  const [feedMode, setFeedMode] = useState<"stream" | "raw">("stream");
  const [detectedObjects, setDetectedObjects] = useState<string[]>([]);
  const [accidentLog, setAccidentLog] = useState<IncidentLogItem[]>([]);
  const [detectionFrames, setDetectionFrames] = useState<number[]>([]);
  const [metrics, setMetrics] = useState<EvaluationMetrics | null>(null);

  const [iouThreshold, setIouThreshold] = useState<number>(0.5);
  const [isLoadingMetrics, setIsLoadingMetrics] = useState<boolean>(false);

  // Upload modal & state
  const [showUploadModal, setShowUploadModal] = useState<boolean>(false);
  const [uploadTab, setUploadTab] = useState<"file" | "url">("file");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [videoUrlInput, setVideoUrlInput] = useState<string>("");
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadSuccess, setUploadSuccess] = useState<string | null>(null);

  // Snapshot modal
  const [activeSnapshot, setActiveSnapshot] = useState<IncidentLogItem | null>(null);

  // Audio alert state
  const [audioAlertEnabled, setAudioAlertEnabled] = useState<boolean>(false);
  const prevAccidentCountRef = useRef<number>(0);

  // Play audio alert safely using Web Audio API
  const playAlertTone = () => {
    try {
      const audioCtx = new (window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)();
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.type = "sawtooth";
      osc.frequency.setValueAtTime(880, audioCtx.currentTime); // A5 note
      osc.frequency.exponentialRampToValueAtTime(440, audioCtx.currentTime + 0.3);
      gain.gain.setValueAtTime(0.2, audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.3);
      osc.connect(gain);
      gain.connect(audioCtx.destination);
      osc.start();
      osc.stop(audioCtx.currentTime + 0.3);
    } catch {
      // Audio autoplay policy fallback
    }
  };

  // Fetch system status & video list on mount
  const checkBackendAndVideos = async () => {
    try {
      const statusRes = await fetch("/backend/api/status");
      if (statusRes.ok) {
        const statusData: SystemStatus = await statusRes.json();
        setSystemStatus(statusData);
        setIsBackendOnline(true);
      } else {
        setIsBackendOnline(false);
      }
    } catch {
      setIsBackendOnline(false);
    }

    try {
      const videosRes = await fetch("/backend/api/videos");
      if (videosRes.ok) {
        const videosData = await videosRes.json();
        if (videosData.success && Array.isArray(videosData.videos)) {
          setVideoList(videosData.videos);
          if (videosData.videos.length > 0 && !activeVideo) {
            setActiveVideo(videosData.videos[0].filename);
          }
        }
      }
    } catch {
      // ignore
    }
  };

  useEffect(() => {
    checkBackendAndVideos();
    const interval = setInterval(checkBackendAndVideos, 8000);
    return () => clearInterval(interval);
  }, [activeVideo]);

  // Polling detections, accident log, and detection frames when an active video is selected
  useEffect(() => {
    if (!activeVideo || !isBackendOnline) return;

    const pollData = async () => {
      // 1. Detected objects
      try {
        const res = await fetch(`/backend/detected_objects/${encodeURIComponent(activeVideo)}`);
        if (res.ok) {
          const data = await res.json();
          if (Array.isArray(data)) {
            setDetectedObjects(data);
          }
        }
      } catch {
        // ignore
      }

      // 2. Accident log
      try {
        const res = await fetch(`/backend/accident_log/${encodeURIComponent(activeVideo)}`);
        if (res.ok) {
          const data = await res.json();
          if (Array.isArray(data)) {
            setAccidentLog(data);
            if (audioAlertEnabled && data.length > prevAccidentCountRef.current) {
              playAlertTone();
            }
            prevAccidentCountRef.current = data.length;
          }
        }
      } catch {
        // ignore
      }

      // 3. Detection frames
      try {
        const res = await fetch(`/backend/detection_frames/${encodeURIComponent(activeVideo)}`);
        if (res.ok) {
          const data = await res.json();
          if (Array.isArray(data)) {
            setDetectionFrames(data);
          }
        }
      } catch {
        // ignore
      }
    };

    pollData();
    const timer = setInterval(pollData, 1200);
    return () => clearInterval(timer);
  }, [activeVideo, isBackendOnline, audioAlertEnabled]);

  // Fetch evaluation metrics
  const fetchEvaluation = async (iouVal: number) => {
    if (!activeVideo || !isBackendOnline) return;
    setIsLoadingMetrics(true);
    try {
      const res = await fetch(`/backend/evaluate/${encodeURIComponent(activeVideo)}?iou=${iouVal}`);
      if (res.ok) {
        const data = await res.json();
        setMetrics(data);
      }
    } catch {
      // ignore
    } finally {
      setIsLoadingMetrics(false);
    }
  };

  useEffect(() => {
    if (activeVideo && isBackendOnline) {
      fetchEvaluation(iouThreshold);
    }
  }, [activeVideo, iouThreshold, isBackendOnline]);

  // Handle Video Upload (File)
  const handleFileUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile) {
      setUploadError("กรุณาเลือกไฟล์วิดีโอก่อน");
      return;
    }

    setIsUploading(true);
    setUploadError(null);
    setUploadSuccess(null);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const res = await fetch("/backend/upload", {
        method: "POST",
        headers: {
          Accept: "application/json",
        },
        body: formData,
      });

      const data = await res.json();
      if (res.ok && data.success) {
        setUploadSuccess(`อัปโหลดสำเร็จ: ${data.filename}`);
        setActiveVideo(data.filename);
        setSelectedFile(null);
        await checkBackendAndVideos();
        setTimeout(() => setShowUploadModal(false), 1200);
      } else {
        setUploadError(data.error || "เกิดข้อผิดพลาดในการอัปโหลด");
      }
    } catch (err) {
      setUploadError(`การเชื่อมต่อล้มเหลว: ${String(err)}`);
    } finally {
      setIsUploading(false);
    }
  };

  // Handle Video Upload (URL)
  const handleUrlUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!videoUrlInput.trim()) {
      setUploadError("กรุณาใส่ URL วิดีโอ");
      return;
    }

    setIsUploading(true);
    setUploadError(null);
    setUploadSuccess(null);

    try {
      const res = await fetch("/backend/upload_url", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        body: JSON.stringify({ video_url: videoUrlInput }),
      });

      const data = await res.json();
      if (res.ok && data.success) {
        setUploadSuccess(`ดาวน์โหลดวิดีโอจาก URL สำเร็จ: ${data.filename}`);
        setActiveVideo(data.filename);
        setVideoUrlInput("");
        await checkBackendAndVideos();
        setTimeout(() => setShowUploadModal(false), 1200);
      } else {
        setUploadError(data.error || "ไม่สามารถดาวน์โหลดวิดีโอจาก URL นี้ได้");
      }
    } catch (err) {
      setUploadError(`การเชื่อมต่อล้มเหลว: ${String(err)}`);
    } finally {
      setIsUploading(false);
    }
  };

  const hasRecentAccident = accidentLog.length > 0;
  const latestAccident = hasRecentAccident ? accidentLog[accidentLog.length - 1] : null;

  return (
    <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column" }}>
      {/* 1. TOP COMMAND BAR */}
      <header
        style={{
          borderBottom: "1px solid var(--border-subtle)",
          background: "rgba(10, 14, 24, 0.85)",
          backdropFilter: "blur(20px)",
          position: "sticky",
          top: 0,
          zIndex: 100,
          padding: "14px 24px",
        }}
      >
        <div
          style={{
            maxWidth: "1600px",
            margin: "0 auto",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            flexWrap: "wrap",
            gap: "16px",
          }}
        >
          {/* Brand Logo & Title */}
          <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
            <div
              style={{
                width: "42px",
                height: "42px",
                borderRadius: "12px",
                background: "linear-gradient(135deg, #ef4444, #3b82f6)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: "20px",
                boxShadow: "0 0 20px rgba(59, 130, 246, 0.4)",
              }}
            >
              🚨
            </div>
            <div>
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <h1 style={{ fontSize: "1.25rem", fontWeight: "800", letterSpacing: "-0.02em" }}>
                  SafeVision AI
                </h1>
                <span className="badge badge-purple" style={{ fontSize: "0.65rem" }}>
                  Next.js Edition
                </span>
              </div>
              <p style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                Real-Time Accident Detection & Dual-Model Telemetry System
              </p>
            </div>
          </div>

          {/* System Telemetry Badges */}
          <div style={{ display: "flex", alignItems: "center", gap: "10px", flexWrap: "wrap" }}>
            <div className={`badge ${isBackendOnline ? "badge-online" : "badge-offline"}`}>
              <span className={`radar-dot ${!isBackendOnline ? "danger" : ""}`} />
              {isBackendOnline ? "Backend: Online" : "Backend: Disconnected"}
            </div>

            <div className="badge badge-blue">
              <span>🧠 Dual Model:</span>
              <span style={{ color: "#fff", fontWeight: 700 }}>YOLOv11 + Best</span>
            </div>

            <div className="badge badge-amber">
              <span>✨ CLAHE: Active</span>
            </div>

            <div
              className="badge"
              style={{
                background: "rgba(100, 116, 139, 0.15)",
                color: "#94a3b8",
                border: "1px solid rgba(100, 116, 139, 0.3)",
              }}
              title="Telegram alert has been safely disabled on backend"
            >
              <span>📱 Telegram: ปิดการส่ง</span>
            </div>

            {/* Audio Alert Toggle */}
            <button
              onClick={() => setAudioAlertEnabled(!audioAlertEnabled)}
              className="btn-ghost"
              style={{
                fontSize: "0.75rem",
                padding: "6px 12px",
                borderColor: audioAlertEnabled ? "var(--accent-crimson)" : "var(--border-subtle)",
              }}
              title="Toggle audio tone when accidents are detected"
            >
              {audioAlertEnabled ? "🔊 เสียงเตือน: เปิด" : "🔇 เสียงเตือน: ปิด"}
            </button>

            {/* Upload Button */}
            <button
              onClick={() => {
                setShowUploadModal(true);
                setUploadError(null);
                setUploadSuccess(null);
              }}
              className="btn-primary"
              style={{ fontSize: "0.8rem", padding: "8px 16px" }}
            >
              <span>+ อัปโหลดวิดีโอ</span>
            </button>
          </div>
        </div>
      </header>

      {/* 2. EMERGENCY HAZARD BANNER (WHEN ACCIDENT DETECTED) */}
      {hasRecentAccident && latestAccident && (
        <div
          style={{
            background: "linear-gradient(90deg, rgba(239, 68, 68, 0.25), rgba(185, 28, 28, 0.35))",
            borderBottom: "1px solid rgba(239, 68, 68, 0.6)",
            padding: "12px 24px",
            animation: "flashDanger 2.5s infinite ease-in-out",
          }}
        >
          <div
            style={{
              maxWidth: "1600px",
              margin: "0 auto",
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              flexWrap: "wrap",
              gap: "12px",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
              <span style={{ fontSize: "1.5rem" }}>⚠️</span>
              <div>
                <span style={{ fontWeight: "800", color: "#fca5a5", fontSize: "0.95rem" }}>
                  ตรวจพบอุบัติเหตุ! (Accident Detected)
                </span>
                <span
                  style={{
                    marginLeft: "10px",
                    fontSize: "0.85rem",
                    color: "#fecaca",
                  }}
                >
                  เวลา: {latestAccident.time} วินาที | เฟรม: #{latestAccident.frame} | ความมั่นใจ:{" "}
                  {(latestAccident.confidence * 100).toFixed(1)}%
                  {latestAccident.vehicle_counts && Object.keys(latestAccident.vehicle_counts).length > 0 && (
                    <span style={{ marginLeft: "8px", fontWeight: "600", color: "#6ee7b7" }}>
                      [ยานพาหนะ:{" "}
                      {Object.entries(latestAccident.vehicle_counts)
                        .map(([k, v]) => `${k} (${v})`)
                        .join(", ")}
                      ]
                    </span>
                  )}
                </span>
              </div>
            </div>

            <button
              onClick={() => setActiveSnapshot(latestAccident)}
              className="btn-ghost"
              style={{
                background: "rgba(239, 68, 68, 0.25)",
                borderColor: "rgba(239, 68, 68, 0.5)",
                color: "#fee2e2",
                fontWeight: "600",
                fontSize: "0.75rem",
              }}
            >
              🔍 ดูภาพหลักฐาน (View Snapshot)
            </button>
          </div>
        </div>
      )}

      {/* 3. MAIN DASHBOARD CONTENT */}
      <main style={{ flex: 1, padding: "24px", maxWidth: "1600px", width: "100%", margin: "0 auto" }}>
        {/* Active Video Selector Bar */}
        <div
          className="glass-panel"
          style={{
            padding: "16px 20px",
            marginBottom: "24px",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            flexWrap: "wrap",
            gap: "14px",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "14px", flexWrap: "wrap" }}>
            <span style={{ fontSize: "0.85rem", fontWeight: "600", color: "var(--text-muted)" }}>
              📹 วิดีโอที่กำลังวิเคราะห์:
            </span>

            {videoList.length > 0 ? (
              <select
                value={activeVideo}
                onChange={(e) => setActiveVideo(e.target.value)}
                className="input-control"
                style={{ width: "auto", minWidth: "260px", padding: "8px 12px" }}
              >
                {videoList.map((vid) => (
                  <option key={vid.filename} value={vid.filename}>
                    {vid.filename} ({vid.size_mb} MB)
                  </option>
                ))}
              </select>
            ) : (
              <span style={{ fontSize: "0.85rem", color: "var(--text-dim)" }}>
                ยังไม่มีวิดีโอในระบบ กรุณากดปุ่ม &quot;+ อัปโหลดวิดีโอ&quot;
              </span>
            )}
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
            <button
              onClick={() => setFeedMode("stream")}
              className={`btn-ghost ${feedMode === "stream" ? "active" : ""}`}
            >
              🔴 สตรีมการตรวจจับสด (AI Live Stream)
            </button>
            <button
              onClick={() => setFeedMode("raw")}
              className={`btn-ghost ${feedMode === "raw" ? "active" : ""}`}
            >
              🎬 วิดีโอต้นฉบับ (Original Video)
            </button>
            <button
              onClick={() => checkBackendAndVideos()}
              className="btn-ghost"
              title="Refresh connection & video list"
            >
              🔄 รีเฟรช
            </button>
          </div>
        </div>

        {/* 2-Column Responsive Surveillance Grid */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(480px, 1fr))",
            gap: "24px",
            alignItems: "start",
          }}
        >
          {/* LEFT COLUMN: LIVE FEED VIEWPORT & DETECTIONS */}
          <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
            {/* Live Video Card */}
            <div className="glass-panel" style={{ overflow: "hidden" }}>
              {/* Card Header */}
              <div
                style={{
                  padding: "16px 20px",
                  borderBottom: "1px solid var(--border-subtle)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                  <span className={`radar-dot ${feedMode === "stream" ? "" : "danger"}`} />
                  <span style={{ fontWeight: "700", fontSize: "0.95rem" }}>
                    {feedMode === "stream" ? "AI Live Inference Stream" : "Original Source Playback"}
                  </span>
                </div>
                <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
                  {activeVideo || "No active file"}
                </div>
              </div>

              {/* Viewport Box */}
              <div
                style={{
                  position: "relative",
                  background: "#030712",
                  minHeight: "420px",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                {activeVideo ? (
                  feedMode === "stream" ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img
                      src={`/backend/video_feed/${encodeURIComponent(activeVideo)}`}
                      alt="AI Detection Feed"
                      style={{
                        width: "100%",
                        height: "auto",
                        maxHeight: "560px",
                        objectFit: "contain",
                        display: "block",
                      }}
                      onError={(e) => {
                        (e.target as HTMLElement).style.display = "none";
                      }}
                    />
                  ) : (
                    <video
                      key={activeVideo}
                      controls
                      autoPlay
                      muted
                      loop
                      style={{
                        width: "100%",
                        height: "auto",
                        maxHeight: "560px",
                        objectFit: "contain",
                        display: "block",
                      }}
                    >
                      <source
                        src={`/backend/video/${encodeURIComponent(activeVideo)}`}
                        type="video/mp4"
                      />
                      เบราว์เซอร์ของคุณไม่รองรับการเล่นวิดีโอ
                    </video>
                  )
                ) : (
                  <div style={{ textAlign: "center", color: "var(--text-muted)", padding: "40px" }}>
                    <div style={{ fontSize: "48px", marginBottom: "12px" }}>📁</div>
                    <p style={{ fontWeight: "600" }}>ยังไม่ได้เลือกวิดีโอ</p>
                    <p style={{ fontSize: "0.85rem", color: "var(--text-dim)" }}>
                      กรุณาเลือกวิดีโอจากเมนูด้านบน หรืออัปโหลดวิดีโอใหม่
                    </p>
                  </div>
                )}

                {/* HUD Overlay Badge */}
                {activeVideo && feedMode === "stream" && (
                  <div
                    style={{
                      position: "absolute",
                      bottom: "12px",
                      left: "12px",
                      background: "rgba(0, 0, 0, 0.7)",
                      backdropFilter: "blur(6px)",
                      border: "1px solid rgba(255, 255, 255, 0.15)",
                      borderRadius: "6px",
                      padding: "4px 10px",
                      fontSize: "0.75rem",
                      fontFamily: "var(--font-mono)",
                      color: "#34d399",
                      display: "flex",
                      alignItems: "center",
                      gap: "6px",
                    }}
                  >
                    <span className="radar-dot" />
                    LIVE YOLO TRACKING
                  </div>
                )}
              </div>

              {/* Viewport Footer Telemetry */}
              <div
                style={{
                  padding: "14px 20px",
                  background: "rgba(0, 0, 0, 0.3)",
                  borderTop: "1px solid var(--border-subtle)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  flexWrap: "wrap",
                  gap: "10px",
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                  <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>วัตถุที่กำลังตรวจจับ:</span>
                  {detectedObjects.length > 0 ? (
                    <div style={{ display: "flex", gap: "6px", flexWrap: "wrap" }}>
                      {Array.from(new Set(detectedObjects)).map((label, idx) => (
                        <span
                          key={idx}
                          className="badge"
                          style={{
                            background:
                              label.toLowerCase() === "accident"
                                ? "rgba(239, 68, 68, 0.2)"
                                : "rgba(59, 130, 246, 0.2)",
                            color: label.toLowerCase() === "accident" ? "#fca5a5" : "#93c5fd",
                            border: `1px solid ${
                              label.toLowerCase() === "accident"
                                ? "rgba(239, 68, 68, 0.4)"
                                : "rgba(59, 130, 246, 0.4)"
                            }`,
                            fontSize: "0.7rem",
                          }}
                        >
                          {label}
                        </span>
                      ))}
                    </div>
                  ) : (
                    <span style={{ fontSize: "0.75rem", color: "var(--text-dim)" }}>
                      ไม่พบวัตถุในเฟรมปัจจุบัน
                    </span>
                  )}
                </div>

                <div style={{ fontSize: "0.75rem", color: "var(--text-dim)", fontFamily: "var(--font-mono)" }}>
                  CLAHE: Enhanced | Resolution: 1020x600
                </div>
              </div>
            </div>

            {/* Saved Key Detection Frames Filmstrip */}
            <div className="glass-panel" style={{ padding: "20px" }}>
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  marginBottom: "14px",
                }}
              >
                <h3 style={{ fontSize: "0.95rem", fontWeight: "700" }}>
                  🎞️ ภาพเฟรมที่มีการตรวจจับ (Key Detection Frames)
                </h3>
                <span className="badge badge-blue">{detectionFrames.length} Frames</span>
              </div>

              {detectionFrames.length > 0 ? (
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(auto-fill, minmax(130px, 1fr))",
                    gap: "12px",
                    maxHeight: "260px",
                    overflowY: "auto",
                    paddingRight: "6px",
                  }}
                >
                  {detectionFrames.slice(-16).map((fNum) => (
                    <div
                      key={fNum}
                      style={{
                        background: "rgba(0, 0, 0, 0.4)",
                        borderRadius: "10px",
                        border: "1px solid var(--border-subtle)",
                        padding: "6px",
                        textAlign: "center",
                        cursor: "pointer",
                        transition: "transform 0.2s ease, border-color 0.2s ease",
                      }}
                      onClick={() =>
                        setActiveSnapshot({
                          frame: fNum,
                          time: Math.floor(fNum / 30),
                          box: [0, 0, 0, 0],
                          confidence: 0.8,
                          img: `${activeVideo}_detframe_${fNum}.jpg`,
                          label: "accident",
                        })
                      }
                      onMouseEnter={(e) => {
                        (e.currentTarget as HTMLElement).style.borderColor = "var(--accent-blue)";
                        (e.currentTarget as HTMLElement).style.transform = "scale(1.03)";
                      }}
                      onMouseLeave={(e) => {
                        (e.currentTarget as HTMLElement).style.borderColor = "var(--border-subtle)";
                        (e.currentTarget as HTMLElement).style.transform = "scale(1)";
                      }}
                    >
                      <div
                        style={{
                          fontSize: "0.7rem",
                          color: "var(--text-muted)",
                          marginBottom: "4px",
                          fontFamily: "var(--font-mono)",
                        }}
                      >
                        Frame #{fNum}
                      </div>
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={`/backend/detection_frame/${encodeURIComponent(activeVideo)}?frame=${fNum}`}
                        alt={`Frame ${fNum}`}
                        style={{
                          width: "100%",
                          height: "75px",
                          objectFit: "cover",
                          borderRadius: "6px",
                        }}
                        onError={(e) => {
                          (e.target as HTMLElement).style.display = "none";
                        }}
                      />
                    </div>
                  ))}
                </div>
              ) : (
                <div
                  style={{
                    padding: "24px",
                    textAlign: "center",
                    color: "var(--text-dim)",
                    fontSize: "0.85rem",
                  }}
                >
                  ยังไม่มีเฟรมตรวจจับที่บันทึกไว้สำหรับวิดีโอนี้
                </div>
              )}
            </div>
          </div>

          {/* RIGHT COLUMN: AI EVALUATION & ACCIDENT LOG */}
          <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
            {/* AI Model Evaluation & Performance Analytics */}
            <div className="glass-panel" style={{ padding: "20px" }}>
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  marginBottom: "16px",
                }}
              >
                <div>
                  <h3 style={{ fontSize: "1rem", fontWeight: "700" }}>
                    📊 การประเมินผลโมเดล (AI Evaluation & Metrics)
                  </h3>
                  <p style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                    คำนวณ Precision, Recall, mAP และเวลาการประมวลผลเทียบกับ Ground Truth
                  </p>
                </div>

                <button
                  onClick={() => fetchEvaluation(iouThreshold)}
                  className="btn-ghost"
                  style={{ fontSize: "0.75rem", padding: "6px 12px" }}
                  disabled={isLoadingMetrics}
                >
                  {isLoadingMetrics ? "กำลังคำนวณ..." : "⚡ Recalculate"}
                </button>
              </div>

              {/* IoU Threshold Slider */}
              <div
                style={{
                  background: "rgba(0, 0, 0, 0.25)",
                  padding: "12px 16px",
                  borderRadius: "10px",
                  marginBottom: "16px",
                  border: "1px solid var(--border-subtle)",
                }}
              >
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    fontSize: "0.8rem",
                    marginBottom: "6px",
                  }}
                >
                  <span style={{ color: "var(--text-muted)" }}>เกณฑ์ IoU Threshold (สำหรับ mAP):</span>
                  <span style={{ fontWeight: "700", color: "var(--accent-cyan)", fontFamily: "var(--font-mono)" }}>
                    {iouThreshold.toFixed(2)}
                  </span>
                </div>
                <input
                  type="range"
                  min="0.1"
                  max="0.9"
                  step="0.05"
                  value={iouThreshold}
                  onChange={(e) => setIouThreshold(parseFloat(e.target.value))}
                  style={{ width: "100%", accentColor: "var(--accent-blue)" }}
                />
              </div>

              {/* Metrics Grid */}
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(2, 1fr)",
                  gap: "12px",
                  marginBottom: "16px",
                }}
              >
                {/* mAP Card */}
                <div
                  style={{
                    background: "rgba(16, 185, 129, 0.08)",
                    border: "1px solid rgba(16, 185, 129, 0.25)",
                    borderRadius: "12px",
                    padding: "14px",
                  }}
                >
                  <div style={{ fontSize: "0.75rem", color: "#34d399", fontWeight: "600" }}>mAP @ IoU</div>
                  <div
                    style={{
                      fontSize: "1.6rem",
                      fontWeight: "800",
                      color: "#10b981",
                      fontFamily: "var(--font-mono)",
                    }}
                  >
                    {metrics ? `${(metrics.mAP * 100).toFixed(1)}%` : "--"}
                  </div>
                  <div style={{ fontSize: "0.7rem", color: "var(--text-dim)" }}>
                    Mean Average Precision
                  </div>
                </div>

                {/* F1 Score Card */}
                <div
                  style={{
                    background: "rgba(59, 130, 246, 0.08)",
                    border: "1px solid rgba(59, 130, 246, 0.25)",
                    borderRadius: "12px",
                    padding: "14px",
                  }}
                >
                  <div style={{ fontSize: "0.75rem", color: "#60a5fa", fontWeight: "600" }}>F1-Score</div>
                  <div
                    style={{
                      fontSize: "1.6rem",
                      fontWeight: "800",
                      color: "#3b82f6",
                      fontFamily: "var(--font-mono)",
                    }}
                  >
                    {metrics ? `${(metrics.f1 * 100).toFixed(1)}%` : "--"}
                  </div>
                  <div style={{ fontSize: "0.7rem", color: "var(--text-dim)" }}>
                    Harmonic mean of P & R
                  </div>
                </div>

                {/* Precision */}
                <div
                  style={{
                    background: "rgba(255, 255, 255, 0.03)",
                    border: "1px solid var(--border-subtle)",
                    borderRadius: "12px",
                    padding: "12px",
                  }}
                >
                  <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Precision</div>
                  <div style={{ fontSize: "1.25rem", fontWeight: "700", fontFamily: "var(--font-mono)" }}>
                    {metrics ? `${(metrics.precision * 100).toFixed(1)}%` : "--"}
                  </div>
                </div>

                {/* Recall */}
                <div
                  style={{
                    background: "rgba(255, 255, 255, 0.03)",
                    border: "1px solid var(--border-subtle)",
                    borderRadius: "12px",
                    padding: "12px",
                  }}
                >
                  <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Recall</div>
                  <div style={{ fontSize: "1.25rem", fontWeight: "700", fontFamily: "var(--font-mono)" }}>
                    {metrics ? `${(metrics.recall * 100).toFixed(1)}%` : "--"}
                  </div>
                </div>
              </div>

              {/* Confusion Matrix Stats & Inference Latency */}
              <div
                style={{
                  background: "rgba(0, 0, 0, 0.3)",
                  borderRadius: "10px",
                  padding: "12px 16px",
                  fontSize: "0.75rem",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  flexWrap: "wrap",
                  gap: "10px",
                  fontFamily: "var(--font-mono)",
                }}
              >
                <div>
                  <span style={{ color: "#34d399" }}>TP: {metrics?.TP ?? 0}</span> |{" "}
                  <span style={{ color: "#f87171" }}>FP: {metrics?.FP ?? 0}</span> |{" "}
                  <span style={{ color: "#fbbf24" }}>FN: {metrics?.FN ?? 0}</span> |{" "}
                  <span style={{ color: "#94a3b8" }}>TN: {metrics?.TN ?? 0}</span>
                </div>

                <div style={{ color: "var(--accent-cyan)" }}>
                  ⚡ ความเร็วประมวลผล:{" "}
                  {metrics?.avg_inference_ms !== null && metrics?.avg_inference_ms !== undefined
                    ? `${metrics.avg_inference_ms.toFixed(1)} ms/frame`
                    : "กำลังวัดผล"}
                </div>
              </div>
            </div>

            {/* Incident History & Evidence Snapshots Log */}
            <div className="glass-panel" style={{ padding: "20px" }}>
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  marginBottom: "16px",
                }}
              >
                <div>
                  <h3 style={{ fontSize: "1rem", fontWeight: "700" }}>
                    🚨 ประวัติการเกิดอุบัติเหตุ (Incident Timeline)
                  </h3>
                  <p style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                    รายการอุบัติเหตุที่บันทึกได้พร้อมภาพหลักฐานและจำนวนยานพาหนะ
                  </p>
                </div>
                <span className="badge badge-live">{accidentLog.length} เหตุการณ์</span>
              </div>

              {accidentLog.length > 0 ? (
                <div
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    gap: "12px",
                    maxHeight: "420px",
                    overflowY: "auto",
                    paddingRight: "6px",
                  }}
                >
                  {accidentLog.map((item, idx) => (
                    <div
                      key={idx}
                      style={{
                        background: "rgba(239, 68, 68, 0.06)",
                        border: "1px solid rgba(239, 68, 68, 0.25)",
                        borderRadius: "12px",
                        padding: "12px",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "space-between",
                        gap: "12px",
                        cursor: "pointer",
                        transition: "all 0.2s ease",
                      }}
                      onClick={() => setActiveSnapshot(item)}
                      onMouseEnter={(e) => {
                        (e.currentTarget as HTMLElement).style.borderColor = "rgba(239, 68, 68, 0.6)";
                        (e.currentTarget as HTMLElement).style.background = "rgba(239, 68, 68, 0.12)";
                      }}
                      onMouseLeave={(e) => {
                        (e.currentTarget as HTMLElement).style.borderColor = "rgba(239, 68, 68, 0.25)";
                        (e.currentTarget as HTMLElement).style.background = "rgba(239, 68, 68, 0.06)";
                      }}
                    >
                      {/* Left: Thumbnail */}
                      <div style={{ flexShrink: 0 }}>
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img
                          src={`/backend/detection_frame/${encodeURIComponent(activeVideo)}?frame=${item.frame}`}
                          alt="Accident snapshot"
                          style={{
                            width: "70px",
                            height: "50px",
                            objectFit: "cover",
                            borderRadius: "6px",
                            border: "1px solid rgba(239, 68, 68, 0.3)",
                          }}
                          onError={(e) => {
                            (e.target as HTMLElement).style.display = "none";
                          }}
                        />
                      </div>

                      {/* Middle: Details */}
                      <div style={{ flex: 1 }}>
                        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                          <span style={{ fontWeight: "700", fontSize: "0.85rem", color: "#fca5a5" }}>
                            ⏱️ วินาทีที่ {item.time}s
                          </span>
                          <span
                            style={{
                              fontSize: "0.7rem",
                              color: "var(--text-muted)",
                              fontFamily: "var(--font-mono)",
                            }}
                          >
                            Frame #{item.frame}
                          </span>
                        </div>

                        <div style={{ fontSize: "0.75rem", marginTop: "2px", color: "var(--text-main)" }}>
                          ความมั่นใจ:{" "}
                          <strong style={{ color: "#f87171" }}>
                            {(item.confidence * 100).toFixed(1)}%
                          </strong>
                        </div>

                        {/* Vehicle Counts */}
                        {item.vehicle_counts && Object.keys(item.vehicle_counts).length > 0 && (
                          <div
                            style={{
                              fontSize: "0.7rem",
                              color: "#6ee7b7",
                              marginTop: "2px",
                            }}
                          >
                            🚗 ยานพาหนะในพื้นที่:{" "}
                            {Object.entries(item.vehicle_counts)
                              .map(([vName, vCount]) => `${vName} (${vCount})`)
                              .join(", ")}
                          </div>
                        )}
                      </div>

                      {/* Right: Action */}
                      <div style={{ flexShrink: 0 }}>
                        <span className="badge badge-blue" style={{ fontSize: "0.7rem" }}>
                          ดูภาพขยาย
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div
                  style={{
                    padding: "36px 20px",
                    textAlign: "center",
                    color: "var(--text-dim)",
                    fontSize: "0.85rem",
                  }}
                >
                  <div style={{ fontSize: "36px", marginBottom: "8px" }}>🛡️</div>
                  <p style={{ fontWeight: "600", color: "#34d399" }}>ยังไม่พบอุบัติเหตุในวิดีโอนี้</p>
                  <p style={{ fontSize: "0.75rem" }}>
                    ระบบกำลังเฝ้าระวังและวิเคราะห์อย่างต่อเนื่อง
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      </main>

      {/* 4. UPLOAD VIDEO MODAL */}
      {showUploadModal && (
        <div className="modal-overlay" onClick={() => setShowUploadModal(false)}>
          <div
            className="modal-content"
            style={{ maxWidth: "560px", padding: "28px" }}
            onClick={(e) => e.stopPropagation()}
          >
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                marginBottom: "20px",
              }}
            >
              <h2 style={{ fontSize: "1.2rem", fontWeight: "700" }}>📥 อัปโหลดวิดีโอเข้าสู่ระบบ</h2>
              <button
                onClick={() => setShowUploadModal(false)}
                style={{
                  background: "transparent",
                  border: "none",
                  color: "var(--text-muted)",
                  fontSize: "1.2rem",
                  cursor: "pointer",
                }}
              >
                ✕
              </button>
            </div>

            {/* Tab Switcher */}
            <div
              style={{
                display: "flex",
                background: "rgba(0, 0, 0, 0.4)",
                padding: "4px",
                borderRadius: "10px",
                marginBottom: "20px",
                gap: "4px",
              }}
            >
              <button
                onClick={() => {
                  setUploadTab("file");
                  setUploadError(null);
                }}
                className={`btn-ghost ${uploadTab === "file" ? "active" : ""}`}
                style={{ flex: 1, justifyContent: "center" }}
              >
                📁 อัปโหลดไฟล์วิดีโอ (MP4, AVI)
              </button>
              <button
                onClick={() => {
                  setUploadTab("url");
                  setUploadError(null);
                }}
                className={`btn-ghost ${uploadTab === "url" ? "active" : ""}`}
                style={{ flex: 1, justifyContent: "center" }}
              >
                🔗 ดาวน์โหลดจาก URL
              </button>
            </div>

            {/* Feedback Messages */}
            {uploadError && (
              <div
                style={{
                  background: "rgba(239, 68, 68, 0.15)",
                  border: "1px solid rgba(239, 68, 68, 0.4)",
                  color: "#fca5a5",
                  padding: "10px 14px",
                  borderRadius: "8px",
                  fontSize: "0.8rem",
                  marginBottom: "16px",
                }}
              >
                ⚠️ {uploadError}
              </div>
            )}

            {uploadSuccess && (
              <div
                style={{
                  background: "rgba(16, 185, 129, 0.15)",
                  border: "1px solid rgba(16, 185, 129, 0.4)",
                  color: "#6ee7b7",
                  padding: "10px 14px",
                  borderRadius: "8px",
                  fontSize: "0.8rem",
                  marginBottom: "16px",
                }}
              >
                ✅ {uploadSuccess}
              </div>
            )}

            {/* File Upload Form */}
            {uploadTab === "file" ? (
              <form onSubmit={handleFileUpload}>
                <div
                  style={{
                    border: "2px dashed var(--border-subtle)",
                    borderRadius: "12px",
                    padding: "30px 20px",
                    textAlign: "center",
                    cursor: "pointer",
                    background: "rgba(255, 255, 255, 0.02)",
                    marginBottom: "20px",
                  }}
                  onClick={() => document.getElementById("file-input-el")?.click()}
                >
                  <div style={{ fontSize: "36px", marginBottom: "8px" }}>🎬</div>
                  <p style={{ fontWeight: "600", fontSize: "0.9rem" }}>
                    {selectedFile ? selectedFile.name : "คลิกเลือกไฟล์วิดีโอ หรือลากไฟล์มาวางที่นี่"}
                  </p>
                  <p style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "4px" }}>
                    รองรับไฟล์: .mp4, .avi, .mov, .mkv, .webm (ขนาดไม่เกิน 500MB)
                  </p>
                  <input
                    id="file-input-el"
                    type="file"
                    accept=".mp4,.avi,.mov,.mkv,.webm"
                    style={{ display: "none" }}
                    onChange={(e) => {
                      if (e.target.files && e.target.files[0]) {
                        setSelectedFile(e.target.files[0]);
                        setUploadError(null);
                      }
                    }}
                  />
                </div>

                <button
                  type="submit"
                  className="btn-primary"
                  style={{ width: "100%" }}
                  disabled={!selectedFile || isUploading}
                >
                  {isUploading ? "กำลังอัปโหลดและตรวจสอบวิดีโอ..." : "อัปโหลดและเริ่มวิเคราะห์"}
                </button>
              </form>
            ) : (
              /* URL Upload Form */
              <form onSubmit={handleUrlUpload}>
                <div style={{ marginBottom: "18px" }}>
                  <label
                    style={{
                      display: "block",
                      fontSize: "0.8rem",
                      fontWeight: "600",
                      marginBottom: "6px",
                      color: "var(--text-muted)",
                    }}
                  >
                    ป้อน URL วิดีโอสาธารณะ:
                  </label>
                  <input
                    type="url"
                    className="input-control"
                    placeholder="https://example.com/video.mp4"
                    value={videoUrlInput}
                    onChange={(e) => setVideoUrlInput(e.target.value)}
                    required
                  />
                  <p style={{ fontSize: "0.7rem", color: "var(--text-dim)", marginTop: "6px" }}>
                    🛡️ มีระบบ SSRF Protection ป้องกันการเข้าถึงเครือข่ายภายใน และจำกัดขนาดไม่เกิน 250MB
                  </p>
                </div>

                <button
                  type="submit"
                  className="btn-primary"
                  style={{ width: "100%" }}
                  disabled={!videoUrlInput || isUploading}
                >
                  {isUploading ? "กำลังดาวน์โหลดวิดีโอจาก URL..." : "ดาวน์โหลดและเริ่มวิเคราะห์"}
                </button>
              </form>
            )}
          </div>
        </div>
      )}

      {/* 5. SNAPSHOT LIGHTBOX MODAL */}
      {activeSnapshot && (
        <div className="modal-overlay" onClick={() => setActiveSnapshot(null)}>
          <div
            className="modal-content"
            style={{ maxWidth: "880px", padding: "24px" }}
            onClick={(e) => e.stopPropagation()}
          >
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                marginBottom: "16px",
              }}
            >
              <div>
                <h3 style={{ fontSize: "1.1rem", fontWeight: "700", color: "#fca5a5" }}>
                  🚨 ภาพหลักฐานการเกิดอุบัติเหตุ (Incident Snapshot Evidence)
                </h3>
                <p style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                  เวลา: {activeSnapshot.time} วินาที | เฟรม: #{activeSnapshot.frame} | ความมั่นใจ:{" "}
                  {(activeSnapshot.confidence * 100).toFixed(1)}%
                </p>
              </div>
              <button
                onClick={() => setActiveSnapshot(null)}
                style={{
                  background: "transparent",
                  border: "none",
                  color: "var(--text-muted)",
                  fontSize: "1.3rem",
                  cursor: "pointer",
                }}
              >
                ✕
              </button>
            </div>

            {/* Large Snapshot Image */}
            <div
              style={{
                background: "#000",
                borderRadius: "12px",
                overflow: "hidden",
                border: "1px solid var(--border-subtle)",
                textAlign: "center",
                marginBottom: "16px",
              }}
            >
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={`/backend/detection_frame/${encodeURIComponent(activeVideo)}?frame=${activeSnapshot.frame}`}
                alt="Enlarged accident evidence"
                style={{
                  maxWidth: "100%",
                  maxHeight: "580px",
                  height: "auto",
                  display: "block",
                  margin: "0 auto",
                }}
              />
            </div>

            {/* ROI Vehicle Count Details */}
            {activeSnapshot.vehicle_counts && Object.keys(activeSnapshot.vehicle_counts).length > 0 && (
              <div
                style={{
                  background: "rgba(16, 185, 129, 0.08)",
                  border: "1px solid rgba(16, 185, 129, 0.2)",
                  borderRadius: "10px",
                  padding: "12px 16px",
                  fontSize: "0.85rem",
                  color: "#6ee7b7",
                }}
              >
                <strong>📊 การนับยานพาหนะในพื้นที่เกิดเหตุ:</strong>
                <div style={{ display: "flex", gap: "16px", marginTop: "6px" }}>
                  {Object.entries(activeSnapshot.vehicle_counts).map(([name, count]) => (
                    <span key={name} className="badge badge-online">
                      {name}: {count} คัน
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* FOOTER */}
      <footer
        style={{
          borderTop: "1px solid var(--border-subtle)",
          padding: "16px 24px",
          textAlign: "center",
          fontSize: "0.75rem",
          color: "var(--text-dim)",
          background: "rgba(10, 14, 24, 0.5)",
        }}
      >
        SafeVision AI • Car Accident Detection & Surveillance System • Next.js & Flask YOLO Engine
      </footer>
    </div>
  );
}
