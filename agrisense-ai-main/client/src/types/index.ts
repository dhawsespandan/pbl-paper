export type ResultStatus = "idle" | "loading" | "success" | "error";

export interface DetectionResult {
  image_type: "fruit" | "leaf" | "flower_cluster";
  disease: string;
  severity: string;
  confidence: string;
  recommendation: string;
  details: string;
  flower_count?: number;
}