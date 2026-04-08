import { useRef } from "react";
import LoadingIndicator from "./LoadingIndicator";
import { CAPABILITY_TAGS } from "@/constants";
import type { ResultStatus } from "@/types";

interface ImageInputProps {
  status: ResultStatus;
  previewUrl: string | null;
  dragOver: boolean;
  onDrop: (e: React.DragEvent) => void;
  onDragOver: (e: React.DragEvent) => void;
  onDragLeave: () => void;
  onFileChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onReset: () => void;
  onBrowseClick: () => void;
}

export default function ImageInput({
  status,
  previewUrl,
  dragOver,
  onDrop,
  onDragOver,
  onDragLeave,
  onFileChange,
  onReset,
  onBrowseClick,
}: ImageInputProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleBrowse = () => {
    fileInputRef.current?.click();
    onBrowseClick();
  };

  return (
    <div className="w-full max-w-2xl flex flex-col gap-6">
      <div
        className="w-full bg-white rounded-3xl overflow-hidden"
        style={{
          boxShadow: "0 1px 2px rgba(0,0,0,0.04), 0 8px 32px rgba(0,0,0,0.08)",
          border: dragOver ? "1.5px dashed #2d5a27" : "1px solid rgba(220,215,207,0.8)",
          transition: "border 0.2s",
        }}
      >
        {!previewUrl ? (
          <div
            className="flex flex-col items-center justify-center gap-5 py-16 px-8 cursor-pointer"
            style={{ background: dragOver ? "rgba(45,90,39,0.03)" : "transparent", transition: "background 0.2s" }}
            onDrop={onDrop}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onClick={handleBrowse}
          >
            <div
              className="w-16 h-16 rounded-2xl flex items-center justify-center"
              style={{
                background: dragOver
                  ? "linear-gradient(135deg, #3d7a34, #1f4a18)"
                  : "linear-gradient(135deg, #f0f5ef, #e8f0e6)",
                boxShadow: dragOver ? "0 4px 16px rgba(45,90,39,0.25)" : "none",
                transition: "all 0.2s",
              }}
            >
              <svg
                width="28"
                height="28"
                viewBox="0 0 24 24"
                fill="none"
                stroke={dragOver ? "white" : "#2d5a27"}
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <rect x="3" y="3" width="18" height="18" rx="3" />
                <circle cx="8.5" cy="8.5" r="1.5" />
                <polyline points="21 15 16 10 5 21" />
              </svg>
            </div>
            <div className="text-center">
              <p className="text-[16px] font-semibold text-[#222]">
                {dragOver ? "Drop image here" : "Upload leaf or plant image"}
              </p>
              <p className="text-[13px] text-[#aaa] mt-1.5">
                Drag & drop or{" "}
                <span className="text-[#2d5a27] font-semibold hover:underline">browse files</span>
                <span className="mx-2 text-[#ddd]">·</span>
                JPG, PNG, WEBP
              </p>
            </div>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={onFileChange}
            />
          </div>
        ) : (
          <div className="relative">
            <img src={previewUrl} alt="Uploaded crop" className="w-full h-64 object-cover" />
            {status === "loading" && <LoadingIndicator />}
            <button
              onClick={(e) => { e.stopPropagation(); onReset(); }}
              className="absolute top-4 right-4 w-8 h-8 rounded-full bg-white border border-[#e0dbd3] flex items-center justify-center hover:bg-[#f5f3f0] transition-colors"
              style={{ boxShadow: "0 2px 8px rgba(0,0,0,0.12)" }}
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#555" strokeWidth="2.5" strokeLinecap="round">
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>
          </div>
        )}
      </div>

      {status === "idle" && (
        <div className="flex items-center justify-center gap-2 flex-wrap">
          {CAPABILITY_TAGS.map((tag) => (
            <span
              key={tag}
              className="text-[11px] text-[#999] bg-white/70 border border-[#e2ddd6] px-3 py-1 rounded-full font-medium"
            >
              {tag}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}