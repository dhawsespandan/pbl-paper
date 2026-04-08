import type { DetectionResult } from "@/types";

interface OutputBoxProps {
  result: DetectionResult;
  previewUrl: string;
  onReset: () => void;
}

export default function OutputBox({ result, previewUrl, onReset }: OutputBoxProps) {
  const isHealthy = result.disease.toLowerCase().includes("healthy");

  return (
    <div className="w-full flex gap-6 items-start">
      {/* Left — image + reset */}
      <div className="w-[40%] shrink-0 flex flex-col gap-4">
        <div
          className="w-full rounded-3xl overflow-hidden"
          style={{
            boxShadow: "0 1px 2px rgba(0,0,0,0.04), 0 8px 32px rgba(0,0,0,0.08)",
            border: "1px solid rgba(220,215,207,0.8)",
          }}
        >
          <img src={previewUrl} alt="Uploaded crop" className="w-full h-72 object-cover" />
          <div className="bg-white px-4 py-3 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-[#4caf50]" />
              <span className="text-[12px] text-[#666] font-medium">Analysis complete</span>
            </div>
            <span className="text-[11px] text-[#bbb]">AgriSense AI</span>
          </div>
        </div>
        <button
          onClick={onReset}
          className="w-full rounded-2xl border border-[#ddd] bg-white text-[13px] font-semibold text-[#555] py-3 hover:bg-[#f7f5f0] hover:border-[#ccc] transition-all"
          style={{ boxShadow: "0 1px 3px rgba(0,0,0,0.04)" }}
        >
          Upload Again
        </button>
      </div>

      {/* Right — result card */}
      <div
        className="flex-1 min-w-0 rounded-3xl bg-white overflow-hidden"
        style={{
          boxShadow: "0 1px 2px rgba(0,0,0,0.04), 0 8px 32px rgba(0,0,0,0.07)",
          border: "1px solid rgba(220,215,207,0.8)",
        }}
      >
        {/* Header band */}
        <div
          className="px-6 py-5 flex items-start gap-4"
          style={{
            background: isHealthy
              ? "linear-gradient(to right, #e8f5e5, #f3faf2)"
              : "linear-gradient(to right, #fff4e3, #fff9f2)",
            borderBottom: `1px solid ${isHealthy ? "#d0e8ca" : "#f0dab8"}`,
          }}
        >
          <div
            className="w-11 h-11 rounded-2xl flex items-center justify-center shrink-0"
            style={{
              background: isHealthy
                ? "linear-gradient(135deg, #3d7a34, #1f4a18)"
                : "linear-gradient(135deg, #e08c20, #b86c10)",
              boxShadow: isHealthy
                ? "0 4px 12px rgba(45,90,39,0.3)"
                : "0 4px 12px rgba(180,100,20,0.3)",
            }}
          >
            {isHealthy ? (
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="20 6 9 17 4 12" />
              </svg>
            ) : (
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="10" />
                <line x1="12" y1="8" x2="12" y2="12" />
                <line x1="12" y1="16" x2="12.01" y2="16" />
              </svg>
            )}
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-[11px] font-bold uppercase tracking-widest text-[#aaa] mb-1">Detection Result</p>
            <p className="text-[17px] font-bold text-[#111] leading-snug">{result.disease}</p>
          </div>
          <div className="shrink-0 text-right">
            <p className="text-[10px] text-[#bbb] uppercase tracking-wider mb-1">Confidence</p>
            <span className={`text-[17px] font-bold ${isHealthy ? "text-[#2d5a27]" : "text-[#c47a10]"}`}>
              {result.confidence}
            </span>
          </div>
        </div>

        {/* Body */}
        <div className="px-6 py-5 space-y-5">
          {result.image_type === "fruit" && (
            <>
              <div>
                <p className="text-[10px] font-bold uppercase tracking-widest text-[#bbb] mb-1.5">Severity</p>
                <p className="text-[14px] font-medium text-[#333]">{result.severity}</p>
              </div>

              <div className="h-px bg-linear-to-r from-transparent via-[#ece8e2] to-transparent" />
            </>
          )}

          {result.image_type === "flower_cluster" && result.flower_count !== undefined && (
            <>
              <div className="flex items-center gap-4">
                <div
                  className="flex flex-col items-center justify-center rounded-2xl px-5 py-3 shrink-0"
                  style={{
                    background: isHealthy
                      ? "linear-gradient(135deg, #eaf5e7, #f3faf1)"
                      : "linear-gradient(135deg, #fff4e3, #fff9f3)",
                    border: `1px solid ${isHealthy ? "#c8e0c3" : "#efd4a0"}`,
                  }}
                >
                  <span className={`text-[28px] font-bold leading-none ${isHealthy ? "text-[#2d5a27]" : "text-[#c47a10]"}`}>
                    {result.flower_count}
                  </span>
                  <span className="text-[10px] font-bold uppercase tracking-widest text-[#aaa] mt-1">Flowers</span>
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-[10px] font-bold uppercase tracking-widest text-[#bbb] mb-1.5">Flower Count</p>
                  <p className="text-[14px] font-medium text-[#333]">
                    {result.flower_count === 0
                      ? "No individual flowers detected in this image."
                      : `${result.flower_count} individual flower${result.flower_count === 1 ? "" : "s"} detected by the model.`}
                  </p>
                  <p className="text-[11px] text-[#aaa] mt-1.5 leading-snug">
                    Partially open buds and tightly overlapping flowers in dense clusters may not be counted.
                  </p>
                </div>
              </div>

              <div className="h-px bg-linear-to-r from-transparent via-[#ece8e2] to-transparent" />
            </>
          )}

          <div>
            <p className="text-[10px] font-bold uppercase tracking-widest text-[#bbb] mb-1.5">Findings</p>
            <p className="text-[14px] text-[#555] leading-[1.7]">{result.details}</p>
          </div>

          <div className="h-px bg-linear-to-r from-transparent via-[#ece8e2] to-transparent" />

          <div>
            <p className="text-[10px] font-bold uppercase tracking-widest text-[#bbb] mb-2">Recommended Action</p>
            <div
              className="rounded-2xl px-5 py-4"
              style={{
                background: isHealthy
                  ? "linear-gradient(135deg, #eaf5e7, #f3faf1)"
                  : "linear-gradient(135deg, #fff4e3, #fff9f3)",
                border: `1px solid ${isHealthy ? "#c8e0c3" : "#efd4a0"}`,
              }}
            >
              <p className="text-[14px] text-[#333] leading-[1.7]">{result.recommendation}</p>
            </div>
          </div>
        </div>

        <div className="px-6 pb-5">
          <p className="text-[11px] text-[#ccc] text-center">
            AI-generated · Always verify with a qualified agronomist
          </p>
        </div>
      </div>
    </div>
  );
}