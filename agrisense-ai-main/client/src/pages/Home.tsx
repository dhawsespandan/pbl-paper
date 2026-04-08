import { useState, useCallback } from "react";
import Header from "@/components/Header";
import ImageInput from "@/components/ImageInput";
import OutputBox from "@/components/OutputBox";
import type { ResultStatus, DetectionResult } from "@/types";

export default function Home() {
  const [dragOver, setDragOver] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [status, setStatus] = useState<ResultStatus>("idle");
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [errorType, setErrorType] = useState<string | null>(null);
  const [activeNav, setActiveNav] = useState("Detection");

  const processFile = useCallback(async (file: File) => {
    if (!file.type.startsWith("image/")) return;

    setPreviewUrl(URL.createObjectURL(file));
    setStatus("loading");
    setResult(null);
    setError(null);
    setErrorType(null);

    try {
      const formData = new FormData();
      formData.append("image", file);

      const res = await fetch("/api/analyze", {
        method: "POST",
        body: formData,
      });

      const text = await res.text();
      let payload: Record<string, unknown> = {};
      try {
        payload = text ? (JSON.parse(text) as Record<string, unknown>) : {};
      } catch {
        throw new Error(text?.slice(0, 200) || "Invalid response from server");
      }

      if (!res.ok) {
        const msg =
          (typeof payload.error === "string" && payload.error) ||
          (typeof payload.detail === "string" && payload.detail) ||
          `Analysis failed (${res.status})`;
        const eType = typeof payload.error_type === "string" ? payload.error_type : null;
        setErrorType(eType);
        throw new Error(msg);
      }

      setResult(payload as DetectionResult);
      setStatus("success");
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Something went wrong";
      setError(message);
      setStatus("error");
    }
  }, []);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) processFile(file);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) processFile(file);
  };

  const handleReset = () => {
    setPreviewUrl(null);
    setStatus("idle");
    setResult(null);
    setError(null);
    setErrorType(null);
  };

  return (
    <div
      className="min-h-screen flex flex-col"
      style={{ background: "linear-gradient(160deg, #f4f7f2 0%, #f0ede6 100%)" }}
    >
      <Header activeNav={activeNav} onNavChange={setActiveNav} />

      {/* Idle / Loading / Error — vertically centered */}
      {status !== "success" && (
        <main className="flex-1 flex flex-col items-center justify-center px-8 gap-8 w-full">
          <div className="w-full max-w-2xl flex flex-col items-center gap-8">
            <div className="text-center">
              <h1 className="text-[26px] font-bold text-[#111] tracking-tight leading-tight">
                Diagnose Your Apple Crop
              </h1>
              <p className="text-[14px] text-[#888] mt-2.5 max-w-md leading-relaxed">
                Upload a clear image of an apple leaf, fruit, or flower cluster. Our AI will detect diseases and provide actionable guidance.
              </p>
            </div>

            <ImageInput
              status={status}
              previewUrl={previewUrl}
              dragOver={dragOver}
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={() => setDragOver(false)}
              onFileChange={handleFileChange}
              onReset={handleReset}
              onBrowseClick={() => {}}
            />

            {/* Error state */}
            {status === "error" && error && (
              errorType === "unrecognized_image" ? (
                <div
                  className="w-full rounded-2xl overflow-hidden"
                  style={{
                    border: "1px solid #e8d9c0",
                    boxShadow: "0 1px 2px rgba(0,0,0,0.04), 0 8px 24px rgba(0,0,0,0.06)",
                  }}
                >
                  <div
                    className="px-5 py-4 flex items-start gap-4"
                    style={{ background: "linear-gradient(to right, #fdf6ec, #fefaf4)" }}
                  >
                    <div
                      className="w-10 h-10 rounded-xl flex items-center justify-center shrink-0 mt-0.5"
                      style={{ background: "linear-gradient(135deg, #e0a020, #c47a10)", boxShadow: "0 4px 10px rgba(180,100,20,0.25)" }}
                    >
                      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                        <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
                        <line x1="11" y1="8" x2="11" y2="14"/><line x1="11" y1="16" x2="11.01" y2="16"/>
                      </svg>
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-[11px] font-bold uppercase tracking-widest text-[#c47a10] mb-1">Image Not Recognised</p>
                      <p className="text-[14px] font-semibold text-[#333] leading-snug">Not a leaf, flower, or fruit image</p>
                      <p className="text-[13px] text-[#888] mt-1.5 leading-relaxed">
                        Please upload a clear photo of an apple leaf, fruit, or flower cluster for accurate disease detection.
                      </p>
                    </div>
                  </div>
                </div>
              ) : (
                <div
                  className="w-full rounded-2xl px-5 py-4 text-[13px] text-[#c0392b] font-medium"
                  style={{
                    background: "linear-gradient(135deg, #fff5f5, #fff0f0)",
                    border: "1px solid #f5c6c6",
                  }}
                >
                  ⚠ {error}
                </div>
              )
            )}
          </div>
        </main>
      )}

      {/* Success — top-anchored two-column */}
      {status === "success" && result && previewUrl && (
        <main className="flex-1 flex flex-col items-center px-8 py-12 w-full">
          <div className="w-full max-w-5xl">
            <OutputBox result={result} previewUrl={previewUrl} onReset={handleReset} />
          </div>
        </main>
      )}
    </div>
  );
}