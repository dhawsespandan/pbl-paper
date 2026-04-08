import { NAV_ITEMS } from "@/constants";

interface HeaderProps {
  activeNav: string;
  onNavChange: (item: string) => void;
}

export default function Header({ activeNav, onNavChange }: HeaderProps) {
  return (
    <header
      className="bg-white/80 backdrop-blur-md border-b border-[#e2ddd6]/80 px-8 flex items-center gap-4 h-15 sticky top-0 z-10"
      style={{ boxShadow: "0 1px 3px rgba(0,0,0,0.06), 0 4px 12px rgba(0,0,0,0.03)" }}
    >
      {/* Logo */}
      <div className="flex items-center gap-3">
        <div className="relative w-9 h-9 shrink-0">
          <div
            className="absolute inset-0 rounded-xl bg-linear-to-br from-[#3d7a34] to-[#1f4a18]"
            style={{ boxShadow: "0 2px 8px rgba(45,90,39,0.35)" }}
          />
          <div className="relative w-full h-full flex items-center justify-center">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
              <path
                d="M12 2C9.5 2 7 4 7 7c0 1.5.5 2.8 1.3 3.8C6.5 12 5 14.3 5 17c0 3.3 2.7 6 7 6s7-2.7 7-6c0-2.7-1.5-5-3.3-6.2C16.5 9.8 17 8.5 17 7c0-3-2.5-5-5-5z"
                fill="white"
              />
            </svg>
          </div>
        </div>
        <div>
          <p className="text-[14px] font-bold text-[#111] tracking-tight leading-none">AgriSense AI</p>
          <p className="text-[11px] text-[#999] mt-0.5 leading-none">Disease Detection & Advisory</p>
        </div>
      </div>

      <div className="h-5 w-px bg-[#e2ddd6] mx-2" />

      {/* Nav */}
      <nav className="flex items-center gap-1">
        {NAV_ITEMS.map((item) => (
          <button
            key={item}
            onClick={() => onNavChange(item)}
            className={`px-3 py-1.5 rounded-lg text-[13px] transition-colors ${
              activeNav === item
                ? "bg-[#f0f5ef] text-[#2d5a27] font-semibold"
                : "text-[#888] hover:text-[#333] hover:bg-[#f7f5f0] font-medium"
            }`}
          >
            {item}
          </button>
        ))}
      </nav>

      {/* Right */}
      <div className="ml-auto flex items-center gap-3">
        <div
          className="flex items-center gap-2 border border-[#d4e8d0] px-3.5 py-1.5 rounded-full"
          style={{
            background: "linear-gradient(to right, #eaf3e7, #f0f7ee)",
            boxShadow: "0 1px 3px rgba(45,90,39,0.1)",
          }}
        >
          <div className="w-2 h-2 rounded-full bg-[#4caf50] animate-pulse" />
          <span className="text-[12px] text-[#2d5a27] font-semibold">Apple · Season 2026</span>
        </div>
        <div
          className="w-8 h-8 rounded-full flex items-center justify-center border border-[#ccc]"
          style={{
            background: "linear-gradient(135deg, #e8e4de, #d8d3cc)",
            boxShadow: "0 1px 3px rgba(0,0,0,0.1)",
          }}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#666" strokeWidth="2" strokeLinecap="round">
            <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
            <circle cx="12" cy="7" r="4" />
          </svg>
        </div>
      </div>
    </header>
  );
}