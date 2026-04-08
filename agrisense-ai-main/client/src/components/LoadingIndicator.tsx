export default function LoadingIndicator() {
  return (
    <div className="absolute inset-0 bg-white/75 backdrop-blur-sm flex flex-col items-center justify-center gap-4">
      <div className="w-10 h-10 border-[2.5px] border-[#2d5a27] border-t-transparent rounded-full animate-spin" />
      <p className="text-[14px] text-[#444] font-semibold">Analyzing image…</p>
      <p className="text-[12px] text-[#aaa]">This usually takes a few seconds</p>
    </div>
  );
}