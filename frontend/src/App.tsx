import { useCallback, useEffect, useRef, useState } from "react";
import {
  Clapperboard,
  Download,
  ImagePlus,
  Loader2,
  AlertCircle,
  X,
} from "lucide-react";
import {
  getConfig,
  getJob,
  listJobs,
  submitJob,
  type Job,
  type ServerConfig,
} from "@/api";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

const FPS = 24;

const RESOLUTIONS: Record<string, { native: [number, number]; fast: [number, number] }> = {
  "16:9": { native: [1344, 768], fast: [960, 544] },
  "9:16": { native: [768, 1344], fast: [544, 960] },
  "1:1": { native: [768, 768], fast: [544, 544] },
  "4:3": { native: [1024, 768], fast: [736, 544] },
  "3:4": { native: [768, 1024], fast: [544, 736] },
};

function snapFrames(frames: number): number {
  const clamped = Math.max(120, Math.min(360, Math.round(frames)));
  let n = Math.ceil((clamped - 5) / 17);
  let snapped = 17 * n + 5;
  while (snapped > 360) snapped = 17 * --n + 5;
  return snapped;
}

type Mode = "t2v" | "i2v";

export default function App() {
  const [cfg, setCfg] = useState<ServerConfig | null>(null);
  const [mode, setMode] = useState<Mode>("t2v");

  const [prompt, setPrompt] = useState("");
  const [seconds, setSeconds] = useState(5);
  const [steps, setSteps] = useState(50);
  const [aspect, setAspect] = useState<keyof typeof RESOLUTIONS>("16:9");
  const [quality, setQuality] = useState<"native" | "fast">("fast");
  const [seed, setSeed] = useState<string>("");
  const [firstFrame, setFirstFrame] = useState<File | null>(null);
  const [lastFrame, setLastFrame] = useState<File | null>(null);

  const [jobs, setJobs] = useState<Job[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const pollRef = useRef<number | null>(null);

  useEffect(() => {
    getConfig().then(setCfg).catch(() => setError("Cannot reach the backend on :8000."));
    refreshJobs();
  }, []);

  const refreshJobs = useCallback(async () => {
    try {
      setJobs(await listJobs());
    } catch {
      /* ignore */
    }
  }, []);

  useEffect(() => {
    if (!activeId) return;
    const tick = async () => {
      try {
        const j = await getJob(activeId);
        setJobs((prev) => {
          const rest = prev.filter((p) => p.id !== j.id);
          return [j, ...rest].sort((a, b) => b.created_at - a.created_at);
        });
        if (j.status === "done" || j.status === "error") {
          if (pollRef.current) window.clearInterval(pollRef.current);
          pollRef.current = null;
        }
      } catch {
        /* ignore */
      }
    };
    tick();
    pollRef.current = window.setInterval(tick, 1500);
    return () => {
      if (pollRef.current) window.clearInterval(pollRef.current);
    };
  }, [activeId]);

  const frames = snapFrames(seconds * FPS);
  const realSeconds = (frames / FPS).toFixed(1);
  const [w, h] = RESOLUTIONS[aspect][quality];

  async function onSubmit() {
    setError(null);
    if (!prompt.trim()) {
      setError("Enter a prompt first.");
      return;
    }
    if (mode === "i2v" && !firstFrame && !lastFrame) {
      setError("Image-to-Video needs a first and/or last frame.");
      return;
    }
    setBusy(true);
    try {
      const job = await submitJob({
        mode,
        prompt,
        num_frames: frames,
        num_inference_steps: steps,
        seed: seed.trim() === "" ? null : Number(seed),
        width: w,
        height: h,
        image: mode === "i2v" ? firstFrame : null,
        last_image: mode === "i2v" ? lastFrame : null,
      });
      setActiveId(job.id);
      setJobs((prev) => [job, ...prev]);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  const active = jobs.find((j) => j.id === activeId) ?? jobs[0] ?? null;

  return (
    <div className="min-h-screen bg-background">
      <div className="mx-auto max-w-6xl px-4 py-6 sm:px-6 lg:py-8">
        {/* Header */}
        <header className="mb-6 flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-primary">
              <Clapperboard className="h-5 w-5 text-primary-foreground" />
            </div>
            <div>
              <h1 className="text-lg font-semibold tracking-tight">MiniMax-H3 Video Studio</h1>
              <p className="text-sm text-muted-foreground">
                Open-weight text / image to video with native audio
              </p>
            </div>
          </div>
          {cfg && <StatusBadge cfg={cfg} />}
        </header>

        <div className="grid gap-5 lg:grid-cols-2">
          {/* Controls */}
          <Card>
            <CardContent className="space-y-5 p-5">
              <Tabs value={mode} onValueChange={(v) => setMode(v as Mode)}>
                <TabsList className="grid w-full grid-cols-2">
                  <TabsTrigger value="t2v">Text to Video</TabsTrigger>
                  <TabsTrigger value="i2v">Image to Video</TabsTrigger>
                </TabsList>
              </Tabs>

              <div className="space-y-2">
                <Label htmlFor="prompt">Prompt</Label>
                <Textarea
                  id="prompt"
                  rows={4}
                  value={prompt}
                  placeholder="A red fox trotting through a snowy pine forest, snow crunching underfoot"
                  onChange={(e) => setPrompt(e.target.value)}
                />
              </div>

              {mode === "i2v" && (
                <div className="grid grid-cols-2 gap-4">
                  <ImageField label="First frame" file={firstFrame} onChange={setFirstFrame} />
                  <ImageField label="Last frame (optional)" file={lastFrame} onChange={setLastFrame} />
                </div>
              )}

              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label>Duration</Label>
                  <span className="text-sm text-muted-foreground">
                    <span className="font-medium text-foreground">{realSeconds}s</span> · {frames} frames @ {FPS}fps
                  </span>
                </div>
                <Slider
                  min={5}
                  max={15}
                  step={1}
                  value={[seconds]}
                  onValueChange={(v) => setSeconds(v[0])}
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>Aspect ratio</Label>
                  <Select value={aspect} onValueChange={(v) => setAspect(v as keyof typeof RESOLUTIONS)}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {Object.keys(RESOLUTIONS).map((a) => (
                        <SelectItem key={a} value={a}>{a}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label>Quality</Label>
                  <Select value={quality} onValueChange={(v) => setQuality(v as "native" | "fast")}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="fast">Fast · 544</SelectItem>
                      <SelectItem value="native">Native · 768</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
              <p className="text-xs text-muted-foreground">Canvas {w} x {h} px</p>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label>Steps</Label>
                    <span className="text-sm font-medium">{steps}</span>
                  </div>
                  <Slider
                    min={10}
                    max={60}
                    step={1}
                    value={[steps]}
                    onValueChange={(v) => setSteps(v[0])}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="seed">Seed</Label>
                  <Input
                    id="seed"
                    type="number"
                    value={seed}
                    placeholder="Random"
                    onChange={(e) => setSeed(e.target.value)}
                  />
                </div>
              </div>

              {error && (
                <div className="flex items-start gap-2 rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
                  <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
                  <span>{error}</span>
                </div>
              )}

              <Button className="w-full" disabled={busy} onClick={onSubmit}>
                {busy && <Loader2 className="animate-spin" />}
                {busy ? "Submitting" : "Generate"}
              </Button>
              <p className="text-xs text-muted-foreground">
                No negative prompt or guidance scale — MiniMax-H3 is guidance-distilled. Output carries native stereo audio.
              </p>
            </CardContent>
          </Card>

          {/* Result + history */}
          <div className="space-y-5">
            <Card>
              <CardHeader className="p-5 pb-0">
                <CardTitle>Result</CardTitle>
              </CardHeader>
              <CardContent className="p-5">
                <Preview job={active} />
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="p-5 pb-0">
                <CardTitle>History</CardTitle>
              </CardHeader>
              <CardContent className="p-5">
                {jobs.length === 0 ? (
                  <p className="text-sm text-muted-foreground">No jobs yet.</p>
                ) : (
                  <div className="space-y-2">
                    {jobs.map((j) => (
                      <button
                        key={j.id}
                        onClick={() => setActiveId(j.id)}
                        className={`flex w-full items-center gap-3 rounded-md border px-3 py-2 text-left transition-colors hover:bg-accent ${
                          j.id === active?.id ? "border-primary" : "border-border"
                        }`}
                      >
                        <StatusDot status={j.status} />
                        <span className="w-10 shrink-0 text-xs font-semibold uppercase text-primary">
                          {j.mode}
                        </span>
                        <span className="flex-1 truncate text-sm text-muted-foreground">
                          {String(j.params.prompt ?? "")}
                        </span>
                        <span className="shrink-0 text-xs text-muted-foreground">{j.status}</span>
                      </button>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}

function StatusBadge({ cfg }: { cfg: ServerConfig }) {
  if (cfg.backend === "comfy") {
    return cfg.comfy_online ? (
      <Badge variant="success">ComfyUI · connected</Badge>
    ) : (
      <Badge variant="warning">ComfyUI · offline</Badge>
    );
  }
  if (cfg.mock) return <Badge variant="warning">Mock mode</Badge>;
  return <Badge variant="success">Live · {cfg.device_mode}</Badge>;
}

function ImageField({
  label,
  file,
  onChange,
}: {
  label: string;
  file: File | null;
  onChange: (f: File | null) => void;
}) {
  const url = file ? URL.createObjectURL(file) : null;
  return (
    <div className="space-y-2">
      <Label>{label}</Label>
      <label className="relative flex h-28 cursor-pointer items-center justify-center overflow-hidden rounded-md border border-dashed border-input bg-background transition-colors hover:bg-accent">
        {url ? (
          <img src={url} alt={label} className="h-full w-full object-cover" />
        ) : (
          <span className="flex flex-col items-center gap-1 text-xs text-muted-foreground">
            <ImagePlus className="h-5 w-5" />
            Upload
          </span>
        )}
        <input
          type="file"
          accept="image/png,image/jpeg,image/webp"
          className="absolute inset-0 cursor-pointer opacity-0"
          onChange={(e) => onChange(e.target.files?.[0] ?? null)}
        />
      </label>
      {file && (
        <button
          type="button"
          onClick={() => onChange(null)}
          className="flex items-center gap-1 text-xs text-destructive hover:underline"
        >
          <X className="h-3 w-3" /> Remove
        </button>
      )}
    </div>
  );
}

function StatusDot({ status }: { status: Job["status"] }) {
  const color =
    status === "done"
      ? "bg-emerald-500"
      : status === "error"
        ? "bg-destructive"
        : status === "running"
          ? "bg-amber-500 animate-pulse"
          : "bg-muted-foreground";
  return <span className={`h-2.5 w-2.5 shrink-0 rounded-full ${color}`} />;
}

function Preview({ job }: { job: Job | null }) {
  if (!job) {
    return (
      <div className="flex min-h-56 items-center justify-center rounded-md border border-dashed bg-muted/30 text-sm text-muted-foreground">
        Your video will appear here.
      </div>
    );
  }

  if (job.status === "done" && job.video_url) {
    return (
      <div className="space-y-3">
        <video
          src={job.video_url}
          controls
          autoPlay
          loop
          playsInline
          className="w-full rounded-md bg-black"
        />
        <Button asChild variant="secondary" className="w-full">
          <a href={job.video_url} download>
            <Download /> Download mp4
          </a>
        </Button>
      </div>
    );
  }

  if (job.status === "error") {
    return (
      <div className="space-y-3">
        <div className="flex items-center gap-2 text-sm font-medium text-destructive">
          <AlertCircle className="h-4 w-4" /> Generation failed
        </div>
        <LogView logs={job.logs} />
      </div>
    );
  }

  return (
    <div className="flex min-h-56 flex-col items-center justify-center gap-3 rounded-md border border-dashed bg-muted/30">
      <Loader2 className="h-7 w-7 animate-spin text-primary" />
      <p className="text-sm text-muted-foreground">
        {job.status === "queued" ? "Queued" : "Rendering"}
      </p>
      <div className="w-full px-4">
        <LogView logs={job.logs} />
      </div>
    </div>
  );
}

function LogView({ logs }: { logs: string[] }) {
  if (!logs?.length) return null;
  return (
    <pre className="max-h-44 overflow-auto whitespace-pre-wrap break-words rounded-md border bg-muted/50 p-3 text-xs text-muted-foreground">
      {logs.slice(-12).join("\n")}
    </pre>
  );
}
