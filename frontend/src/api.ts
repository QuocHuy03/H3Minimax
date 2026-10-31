// Thin API client for the FastAPI backend.

export interface ServerConfig {
  backend: "comfy" | "diffusers";
  mock: boolean;
  fps: number;
  min_frames: number;
  max_frames: number;
  // comfy backend
  comfy_url?: string;
  comfy_online?: boolean;
  // diffusers backend
  model_id?: string;
  device_mode?: string;
  device?: string;
}

export type JobStatus = "queued" | "running" | "done" | "error";

export interface Job {
  id: string;
  mode: "t2v" | "i2v";
  status: JobStatus;
  params: Record<string, unknown>;
  created_at: number;
  started_at: number | null;
  finished_at: number | null;
  logs: string[];
  error: string | null;
  output_file: string | null;
  video_url: string | null;
}

export interface GenerateInput {
  mode: "t2v" | "i2v";
  prompt: string;
  num_frames: number;
  num_inference_steps: number;
  seed?: number | null;
  width?: number | null;
  height?: number | null;
  image?: File | null;
  last_image?: File | null;
}

export async function getConfig(): Promise<ServerConfig> {
  const r = await fetch("/api/config");
  if (!r.ok) throw new Error("config failed");
  return r.json();
}

export async function submitJob(input: GenerateInput): Promise<Job> {
  const fd = new FormData();
  fd.set("mode", input.mode);
  fd.set("prompt", input.prompt);
  fd.set("num_frames", String(input.num_frames));
  fd.set("num_inference_steps", String(input.num_inference_steps));
  if (input.seed != null && input.seed !== undefined) fd.set("seed", String(input.seed));
  if (input.width) fd.set("width", String(input.width));
  if (input.height) fd.set("height", String(input.height));
  if (input.image) fd.set("image", input.image);
  if (input.last_image) fd.set("last_image", input.last_image);

  const r = await fetch("/api/generate", { method: "POST", body: fd });
  if (!r.ok) {
    const txt = await r.text();
    throw new Error(txt || `generate failed (${r.status})`);
  }
  return r.json();
}

export async function getJob(id: string): Promise<Job> {
  const r = await fetch(`/api/jobs/${id}`);
  if (!r.ok) throw new Error("job fetch failed");
  return r.json();
}

export async function listJobs(): Promise<Job[]> {
  const r = await fetch("/api/jobs");
  if (!r.ok) throw new Error("jobs fetch failed");
  return (await r.json()).jobs;
}
