#!/usr/bin/env python3
"""
VLANeXt training scheduler — concurrent, fault-tolerant GPU job orchestration.

Replaces scripts/autostart_trainings.sh. Features:
  * CONCURRENCY: packs multiple 2-GPU trainings onto the 4 cards at once (fills idle GPUs).
  * GPU LEASING: tracks GPUs handed out to our running jobs so two of our jobs never collide.
  * FAILURE DETECTION + RETRY: a job whose process dies without checkpoint_final.pt is retried
    up to MAX_RETRIES (with GPU-zombie cleanup) instead of being silently skipped.
  * STATE PERSISTENCE: atomic-writes .claude/scheduler_state.json so a restart resumes.
  * DUAL RESOURCE GATE: real GPU free MiB + cgroup RAM headroom (the OOM fix).
  * NON-BLOCKING launches: each poll launches all that fit, then returns to the loop (no long
    in-loop sleeps) so STOP and crash-reaping stay responsive. A freshly launched job's GPUs are
    held via a short "warmup" grace before its memory is trusted as allocated.
  * STALL ALERT: warns if nothing progresses for STALL_WARN_SEC.

Background: launched inside tmux session `vlanext-sched` (user prefers tmux over nohup).
  tmux new-session -d -s vlanext-sched -c <repo>
  tmux send-keys -t vlanext-sched "<venv>/python scripts/scheduler.py 2>&1 | tee logs/scheduler.log" Enter
Stop: touch .scheduler_stop  (graceful). Status: python3 scripts/scheduler.py --status
"""
import os, sys, json, time, subprocess, glob

REPO = "/mnt/afs-h200/yuyangcheng/workplace/VLANeXt"
VENV = "/mnt/afs-h200/yuyangcheng/venvs/fla_triton32/bin"
os.chdir(REPO)

STATE_FILE = os.path.join(REPO, ".claude", "scheduler_state.json")
STOP_FILE = os.path.join(REPO, ".scheduler_stop")
LOGDIR = os.path.join(REPO, "logs")
ABLATION_DIR = "VLANeXt_ablation_wm"

# ---- resource policy ----
DEFAULT_GPUS_PER_JOB = 2        # a train job's GPU count if its QUEUE entry doesn't set "gpus"
MIN_GPU_FREE_MIB = 30000        # a WM training needs ~25GB/card; require 30GB headroom
RAM_PER_GPU_GB = 42             # ~42GB host RSS per DDP rank (proc); a job reserves gpus * this
RAM_SAFETY_GB = 20              # keep this much headroom on top of jobs
CGROUP_LIMIT = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
CGROUP_USAGE = "/sys/fs/cgroup/memory/memory.usage_in_bytes"

POLL_SEC = 120                  # resource poll cadence
WARMUP_SEC = 300                # after launch, treat a job's GPUs as leased+RAM-reserved without
                                # trusting nvidia-smi free (the proc hasn't allocated yet)
MAX_RETRIES = 2
STALL_WARN_SEC = 3600

# ---- job queue (priority order) ----
# Two kinds of jobs:
#   train: {kind:"train", tag, cfg, save_dir, gpus?} -> gpus-GPU torchrun DDP (default 2);
#          done = checkpoint_final.pt; reserves gpus*RAM_PER_GPU_GB host RAM.
#   eval : {kind:"eval", tag, prefix, nshards, result_dir, shard_glob} -> N 1-GPU sharded workers,
#          done = all shards produced an *_SR* result dir under ABLATION_DIR/<result_dir>/
# NOTE all train configs MUST have distributed:true (else torchrun procs collide on GPU0 — fixed 2026-06-10).
QUEUE = [
    # ttt64 env-gen eval is the #1 priority — give it workers first.
    {"kind": "eval", "tag": "ttt64_envgen", "prefix": "ttt64", "nshards": 8,
     "result_dir": "ttt_chunk64_libero_spatial", "shard_glob": "envgen_ttt64_shard"},
    # swa_ttt uses spare GPUs after eval — 4-GPU DDP for speed (restarts from scratch; no usable mid-ckpt).
    {"kind": "train", "tag": "swa_ttt", "cfg": "ablation_wm_swa_ttt", "save_dir": "swa_ttt_libero_spatial", "gpus": 4},
    {"kind": "train", "tag": "hp_object", "cfg": "ablation_wm_ttt_chunk256_libero_object", "save_dir": "ttt_chunk256_libero_object", "gpus": 2},
    {"kind": "train", "tag": "hp_goal",   "cfg": "ablation_wm_ttt_chunk256_libero_goal",   "save_dir": "ttt_chunk256_libero_goal",   "gpus": 2},
    {"kind": "train", "tag": "hp_long",   "cfg": "ablation_wm_ttt_chunk256_libero_long",   "save_dir": "ttt_chunk256_libero_long",   "gpus": 2},
]
# eval resource model: each shard worker = 1 GPU + ~22GB RAM, can pack many per card.
EVAL_RAM_PER_WORKER_GB = 22
EVAL_MAX_WORKERS = 8            # cap (RAM-bound); scheduler launches as many as RAM allows up to this
PLUS_DIR = "third_party/LIBERO-plus"

ENV = dict(os.environ,
           WANDB_API_KEY="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",
           TORCHDYNAMO_DISABLE="1",
           PYTHONPATH=REPO)


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


# ---------- resource probing ----------
def gpu_free_mib():
    out = sh("nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits")
    res = {}
    for ln in out.splitlines():
        try:
            i, f = ln.split(",")
            res[int(i)] = int(f)
        except ValueError:
            continue
    return res

def ram_free_gb():
    """Free RAM for NEW allocations = limit - real anonymous RSS.
    memory.usage_in_bytes counts page cache + kernel slab (dentry/inode) that are RECLAIMABLE
    under pressure — after a training run that did heavy ckpt IO, usage can read ~320GB while
    real process RSS is ~3GB (the rest evicts on demand). Reading usage_in_bytes here starved
    4-GPU training. Use total_rss (true non-reclaimable anon mem) instead; the per-job RAM
    reservations + RAM_SAFETY_GB provide headroom for cache that hasn't evicted yet."""
    try:
        lim = int(open(CGROUP_LIMIT).read())
        rss = 0
        for ln in open("/sys/fs/cgroup/memory/memory.stat"):
            k, v = ln.split()[:2]
            if k in ("total_rss", "rss"):
                rss = max(rss, int(v))   # total_rss includes children; prefer it
        if rss == 0:                     # fallback if stat unreadable
            return (lim - int(open(CGROUP_USAGE).read())) / 1e9
        return (lim - rss) / 1e9
    except Exception:
        return 0.0


# ---------- experiment status ----------
def exp_done(save_dir):
    """Exact path (no fuzzy glob): ABLATION_DIR/<save_dir>/checkpoint_final.pt."""
    return os.path.exists(os.path.join(ABLATION_DIR, save_dir, "checkpoint_final.pt"))

def eval_done(job):
    """An eval job is done when every shard has produced an *_SR* result dir."""
    base = os.path.join(ABLATION_DIR, job["result_dir"])
    n_sr = len(glob.glob(f"{base}/*{job['shard_glob']}*_SR*"))
    return n_sr >= job["nshards"]

def eval_shard_done(job, s):
    """Has THIS shard finished? (its result dir got the *_SR<score>* suffix)"""
    base = os.path.join(ABLATION_DIR, job["result_dir"])
    return len(glob.glob(f"{base}/*{job['shard_glob']}{s}*_SR*")) > 0

def eval_shard_running(job, s):
    """Is this exact shard's worker process alive? Match the unique config filename."""
    out = sh(f"pgrep -af 'envgen_{job['prefix']}_shard{s}\\.yaml'")
    return any(l.strip() for l in out.splitlines())

def eval_shards_running(job):
    """Count this eval job's shard worker processes still alive."""
    out = sh(f"pgrep -af 'libero_plus_envgen_{job['prefix']}_shard'")
    return len([l for l in out.splitlines() if l.strip()])

def launch_eval_shard(job, shard, gpu):
    """Launch ONE eval shard worker on one GPU (1 GPU + ~22GB RAM each)."""
    cfg = f"libero_plus_envgen_{job['prefix']}_shard{shard}"
    logf = os.path.join(LOGDIR, f"plus_envgen_{job['prefix']}_shard{shard}.log")
    env = dict(ENV, CUDA_VISIBLE_DEVICES=str(gpu),
               MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa",
               LIBERO_CONFIG_PATH="/tmp/libero_plus_cfg",
               PYTHONPATH=f"{REPO}/{PLUS_DIR}:{REPO}", WANDB_MODE="disabled")
    cmd = (f"cd {REPO}/{PLUS_DIR} && {VENV}/python {REPO}/scripts/libero_plus_bench_eval.py "
           f"--config {REPO}/config/{cfg}.yaml")
    f = open(logf, "w")
    p = subprocess.Popen(cmd, shell=True, env=env, stdout=f, stderr=subprocess.STDOUT,
                         preexec_fn=os.setsid)
    log(f"  EVAL-SHARD {job['tag']} shard{shard} on GPU{gpu} pid{p.pid}")
    return p.pid

def proc_alive(pid):
    if pid is None:
        return False
    try:
        os.kill(pid, 0); return True
    except (OSError, TypeError):
        return False

def log_has_fatal(cfg):
    lg = os.path.join(LOGDIR, f"{cfg}.log")
    if not os.path.exists(lg):
        return ""
    return sh(f"grep -hoE 'OutOfMemoryError|ChildFailedError|No module named|No space left' "
              f"'{lg}' | tail -1")

def kill_job_zombies(cfg):
    """A torchrun job that OOM'd can leave child procs holding GPU memory. Kill any process
    still referencing this job's --config so its GPUs free up before we retry."""
    pids = sh(f"pgrep -f 'config/{cfg}.yaml'")
    n = 0
    for pid in pids.split():
        try:
            os.kill(int(pid), 9); n += 1
        except (OSError, ValueError):
            pass
    if n:
        log(f"  cleaned {n} lingering proc(s) for {cfg}")
        time.sleep(5)  # let the driver reclaim VRAM


# ---------- state (atomic write) ----------
def load_state():
    if os.path.exists(STATE_FILE):
        try:
            return json.load(open(STATE_FILE))
        except Exception:
            pass
    return {"jobs": {}}   # tag -> {status, gpus, pid, port, retries, started}

def save_state(st):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f, indent=2)
    os.replace(tmp, STATE_FILE)   # atomic: --status never reads a half-written file


# ---------- launching ----------
def launch(tag, cfg, gpus, port):
    logf = os.path.join(LOGDIR, f"{cfg}.log")
    cmd = (f"CUDA_VISIBLE_DEVICES={','.join(map(str,gpus))} "
           f"{VENV}/torchrun --nproc_per_node={len(gpus)} --master_port={port} "
           f"scripts/train.py --config config/{cfg}.yaml")
    f = open(logf, "w")
    p = subprocess.Popen(cmd, shell=True, env=ENV, stdout=f, stderr=subprocess.STDOUT,
                         preexec_fn=os.setsid)
    log(f"LAUNCH {tag} on GPU{gpus} port{port} pid{p.pid} -> {logf}")
    return p.pid


def main():
    if "--status" in sys.argv:
        print(json.dumps(load_state(), indent=2)); return

    log("scheduler start. queue: " + " ".join(j["tag"] for j in QUEUE))
    st = load_state()
    used_ports = [j.get("port", 0) for j in st["jobs"].values() if j.get("port")]
    port = max([29569] + used_ports) + 1
    last_progress = time.time()

    def is_done(job):
        return eval_done(job) if job["kind"] == "eval" else exp_done(job["save_dir"])

    # mark already-finished up front
    for job in QUEUE:
        if is_done(job):
            st["jobs"].setdefault(job["tag"], {})["status"] = "done"
    save_state(st)

    while True:
        if os.path.exists(STOP_FILE):
            log("stop file found, exiting (running jobs keep going)."); os.remove(STOP_FILE); break
        now = time.time()

        # 1. reap running jobs
        for job in QUEUE:
            tag = job["tag"]; j = st["jobs"].get(tag, {})
            if j.get("status") != "running":
                continue
            if is_done(job):
                j.update(status="done", gpus=[]); log(f"DONE {tag}"); last_progress = now
            elif job["kind"] == "train":
                if not proc_alive(j.get("pid")):
                    fatal = log_has_fatal(job["cfg"]) or "no-final-ckpt"
                    kill_job_zombies(job["cfg"])
                    r = j.get("retries", 0)
                    if r < MAX_RETRIES:
                        j.update(status="queued", gpus=[], pid=None, retries=r + 1)
                        log(f"FAILED {tag} (cause={fatal}); retry {r+1}/{MAX_RETRIES}")
                    else:
                        j.update(status="failed", gpus=[], pid=None)
                        log(f"GIVEUP {tag}")
                    last_progress = now
            else:  # eval: re-entrant top-up happens in the launch loop below; here just give up
                # if NOTHING is progressing (no workers, not done) for too long it'll be caught by
                # the STALL warning. We deliberately don't requeue here — the launch loop is the
                # single, idempotent source of truth for (re)launching missing shards.
                pass
            st["jobs"][tag] = j
        save_state(st)

        # 2. leased GPUs + reserved RAM from running jobs.
        #    Only TRAINING jobs LEASE GPUs (they need a whole card). EVAL workers CO-RESIDE
        #    (~10GB/card) and never lease, so eval never starves the 4-GPU training.
        leased = set(); reserved_ram = 0.0
        for tag, j in st["jobs"].items():
            if j.get("status") == "running":
                if j.get("kind") != "eval":
                    leased.update(j.get("gpus", []))
                if now - j.get("started", 0) < WARMUP_SEC:
                    reserved_ram += j.get("ram_gb", DEFAULT_GPUS_PER_JOB * RAM_PER_GPU_GB)

        # 3. launch what fits (priority order, non-blocking)
        free = gpu_free_mib()
        avail = [g for g, mib in free.items() if mib >= MIN_GPU_FREE_MIB and g not in leased]
        ram = ram_free_gb() - reserved_ram

        for job in QUEUE:
            tag = job["tag"]; j = st["jobs"].get(tag, {"status": "queued", "retries": 0})
            if j.get("status") in ("done", "failed"):
                continue
            if is_done(job):
                if j.get("status") != "done":
                    log(f"DONE {tag}"); last_progress = time.time()
                st["jobs"][tag] = {**j, "status": "done", "gpus": []}; save_state(st); continue

            if job["kind"] == "train":
                if j.get("status") == "running":
                    continue                      # already live; reaper owns it
                ngpu = job.get("gpus", DEFAULT_GPUS_PER_JOB)
                need_ram = ngpu * RAM_PER_GPU_GB
                if len(avail) >= ngpu and ram >= (need_ram + RAM_SAFETY_GB):
                    gpus = avail[:ngpu]; avail = avail[ngpu:]
                    pid = launch(tag, job["cfg"], gpus, port)
                    j.update(kind="train", status="running", gpus=gpus, pid=pid, port=port,
                             started=time.time(), retries=j.get("retries", 0),
                             ram_gb=need_ram)
                    st["jobs"][tag] = j; save_state(st); port += 1
                    ram -= need_ram; leased.update(gpus); last_progress = time.time()

            else:  # eval: RE-ENTRANT top-up — every poll, launch shards that are neither done
                   # nor running, up to EVAL_MAX_WORKERS concurrent, bounded by RAM. CO-RESIDES
                   # with training (does NOT lease cards). Elastic: grows as RAM frees up.
                pending = [s for s in range(job["nshards"])
                           if not eval_shard_done(job, s) and not eval_shard_running(job, s)]
                running_now = sum(1 for s in range(job["nshards"]) if eval_shard_running(job, s))
                if not pending:                   # all shards done-or-running: just track to completion
                    j.update(kind="eval", status="running",
                             started=j.get("started", time.time()), retries=j.get("retries", 0))
                    st["jobs"][tag] = j; save_state(st); continue
                gpu_pool = [g for g, mib in free.items() if mib >= 12000]   # eval needs ~10GB
                if not gpu_pool:
                    continue
                headroom = EVAL_MAX_WORKERS - running_now
                n_by_ram = int((ram - RAM_SAFETY_GB) // EVAL_RAM_PER_WORKER_GB)
                n_new = min(len(pending), max(0, headroom), max(0, n_by_ram))
                if n_new <= 0:
                    continue
                log(f"LAUNCH eval {tag}: +{n_new} worker(s) [running {running_now}, "
                    f"pending {len(pending)}, RAM {ram:.0f}GB, pool GPU{gpu_pool}]")
                for i in range(n_new):
                    s = pending[i]
                    g = gpu_pool[i % len(gpu_pool)]
                    launch_eval_shard(job, s, g)
                    time.sleep(40)                # stagger model loads (RAM/GPU spike protection)
                j.update(kind="eval", status="running",
                         started=j.get("started", time.time()), retries=j.get("retries", 0),
                         ram_gb=(running_now + n_new) * EVAL_RAM_PER_WORKER_GB)
                st["jobs"][tag] = j; save_state(st)
                ram -= n_new * EVAL_RAM_PER_WORKER_GB
                last_progress = time.time()       # NOTE: eval does NOT lease GPUs (co-resides)

        # 4. exit when all terminal
        statuses = [st["jobs"].get(job["tag"], {}).get("status", "queued") for job in QUEUE]
        if all(s in ("done", "failed") for s in statuses):
            log("all jobs terminal: " + ", ".join(
                f"{job['tag']}={st['jobs'].get(job['tag'],{}).get('status')}" for job in QUEUE))
            break

        # 5. stall warning
        if time.time() - last_progress > STALL_WARN_SEC:
            qd = [job["tag"] for job in QUEUE
                  if st["jobs"].get(job["tag"], {}).get("status") not in ("done", "running", "failed")]
            log(f"STALL {STALL_WARN_SEC/60:.0f}min: waiting={qd} | GPUfree={free} "
                f"RAM={ram:.0f}GB(reserved {reserved_ram:.0f}) leased={sorted(leased)}")
            last_progress = time.time()

        time.sleep(POLL_SEC)

    log("scheduler exit.")


if __name__ == "__main__":
    main()
