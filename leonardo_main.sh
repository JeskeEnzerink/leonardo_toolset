#!/usr/bin/env bash
# Single entry point for submission, GPU array workers, and finalizer.
# Put this file in the project root and launch: bash leonardo_main.sh
# Slurm jobs reinvoke this same script with --worker or --finalize.
set -euo pipefail

# ================================================================
# EDIT SETTINGS HERE
# ================================================================
# File locations
# By default, use the folder containing this script. Override with
# LEONARDO_PROJECT_ROOT if you store the script somewhere else.
PROJECT_ROOT="${LEONARDO_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# Support both an ordinary Git checkout (leonardo_toolset/destripe) and
# an existing project containing a nested checkout (leonardo_toolset/
# leonardo_toolset/destripe).
PYTHON_DIR="$PROJECT_ROOT/leonardo_toolset/destripe"
if [[ ! -f "$PYTHON_DIR/leonardo_pipeline.py" ]]; then
    PYTHON_DIR="$PROJECT_ROOT/leonardo_toolset/leonardo_toolset/destripe"
fi
MAIN_SCRIPT="$PROJECT_ROOT/leonardo_main.sh"
SETTINGS_JSON="$PYTHON_DIR/leonardo_settings.json"
TRAINING_PY="$PYTHON_DIR/leonardo_training.py"
PIPELINE_PY="$PYTHON_DIR/leonardo_pipeline.py"
OUTPUT_DIR="$PROJECT_ROOT/desired_output"
LOG_DIR="$PROJECT_ROOT/logs"
JAX_CACHE_DIR="$PROJECT_ROOT/jax_cache"

# Python environment and JAX GPU allocation
ANACONDA_MODULE="anaconda3"
CONDA_ENV="leonardo"
JAX_PREALLOCATE="true"
JAX_MEMORY_FRACTION="0.95"

# GPU resources (four simultaneous array tasks PER channel; channels sequential)
GPU_JOB_NAME="leo_main"
GPU_PARTITION="YOUR_GPU_PARTITION"
GPU_QOS="normal"
GPU_TASKS_PER_CHANNEL=4
GPU_GRES="gpu:1"
GPU_CPUS=18
GPU_MEMORY="100G"
GPU_TIME="2-00:00:00"
GPU_NICE=1000

# Finalizer resources (validates logs and optionally merges; no GPU requested)
FINAL_JOB_NAME="leo_finish"
FINAL_PARTITION="YOUR_FINALIZER_PARTITION"
FINAL_QOS="normal"
FINAL_CPUS=2
FINAL_MEMORY="24G"
FINAL_TIME="02:00:00"

# Notifications and output log names
MAIL_USER=""                    # optional; leave empty to use cluster defaults
GPU_MAIL_EVENTS="FAIL"
FINAL_MAIL_EVENTS="END,FAIL"
TASK_LOG_NAME="task.log"
COMBINED_LOG_NAME="all_tasks.log"
FINAL_LOG_NAME="finalize.log"

# Training, crop, postprocessing and merge true/false are edited in
# SETTINGS_JSON (leonardo_settings.json), not in this shell script.
# ================================================================

# NEW runs place task_0, task_1, ... inside RUN_DIR/tasks/.
# A resumed legacy run keeps its existing RUN_DIR/task_0 layout. Never move
# in-progress task directories; doing so breaks SLURM logs and recovery.
task_root_for_run() {
    local run_dir="$1"
    if [[ -d "$run_dir/task_0" && -d "$run_dir/tasks/task_0" ]]; then
        echo "AMBIGUOUS TASK LAYOUT in $run_dir: both legacy and nested tasks" >&2
        return 1
    fi
    if [[ -d "$run_dir/task_0" ]]; then
        printf '%s\n' "$run_dir"
    else
        printf '%s\n' "$run_dir/tasks"
    fi
}

submit_main() {

ROOT="$PROJECT_ROOT"
D="$PYTHON_DIR"
SOURCE="$SETTINGS_JSON"
MAIN="$MAIN_SCRIPT"
PIPE="$PIPELINE_PY"

for required in "$SOURCE" "$MAIN" "$PIPE" "$TRAINING_PY"; do
    [[ -f "$required" ]] || { echo "MISSING REQUIRED FILE: $required" >&2; exit 2; }
done
command -v sbatch >/dev/null || { echo 'SLURM sbatch is unavailable' >&2; exit 3; }
python3 - "$SOURCE" <<'CHECKJSON'
import json, sys
from pathlib import Path
cfg=json.loads(Path(sys.argv[1]).read_text())
required=('input', 'output', 'destripe', 'training', 'warm_start', 'analysis', 'automation')
assert all(k in cfg for k in required), f'Missing required section: {required}'
assert cfg['training']['backend']=='jax', 'Expected JAX backend'
# Training choices (epochs, warm start, resampling) belong to the JSON.
assert isinstance(cfg['automation'].get('merge_after_success'), bool), 'merge_after_success must be true or false'
print('Source leonardo_settings.json valid; auto merge:', cfg['automation']['merge_after_success'], flush=True)
CHECKJSON
python3 "$PIPE" plan self-test
python3 "$PIPE" plan validate --source "$SOURCE"
CHANNEL_COUNT="$(python3 "$PIPE" plan count --source "$SOURCE")"
[[ "$CHANNEL_COUNT" =~ ^[1-4]$ ]] || { echo "Unexpected channel count: $CHANNEL_COUNT" >&2; exit 5; }
TOTAL_TASKS=$((CHANNEL_COUNT * GPU_TASKS_PER_CHANNEL))

# A named RUN_TAG can be reused to resume previously validated blocks.
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%d_%H%M%S)_$$}"
[[ "$RUN_TAG" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo 'Invalid RUN_TAG' >&2; exit 4; }
RUN_DIR="$OUTPUT_DIR/leonardo_main_${RUN_TAG}"
mkdir -p "$RUN_DIR"
if [[ -e "$RUN_DIR/leonardo_settings.json" ]]; then
    cmp -s "$SOURCE" "$RUN_DIR/leonardo_settings.json" || {
        echo "Existing run uses different leonardo_settings.json. Use a new RUN_TAG: $RUN_DIR" >&2
        exit 5
    }
else
    cp -- "$SOURCE" "$RUN_DIR/leonardo_settings.json"
fi
TASK_ROOT="$(task_root_for_run "$RUN_DIR")"
for ((t=0;t<TOTAL_TASKS;t++)); do mkdir -p "$TASK_ROOT/task_${t}"; done

# SLURM expands %a to each array-task index. Directing logs here requires
# submission-time --output, because #SBATCH does not expand shell variables.
# Use nonoverlapping GLOBAL task IDs. One array (four GPUs) per channel;
# each next array waits until the previous channel's entire array finishes.
PREV_JOBID=""
for ((channel=0;channel<CHANNEL_COUNT;channel++)); do
    first=$((channel * GPU_TASKS_PER_CHANNEL))
    last=$((first + GPU_TASKS_PER_CHANNEL - 1))
    options=(
        --parsable
        --job-name="${GPU_JOB_NAME}_c${channel}"
        --partition="$GPU_PARTITION"
        --qos="$GPU_QOS"
        --array="${first}-${last}%${GPU_TASKS_PER_CHANNEL}"
        --gres="$GPU_GRES"
        --cpus-per-task="$GPU_CPUS"
        --mem="$GPU_MEMORY"
        --time="$GPU_TIME"
        --nice="$GPU_NICE"
        --mail-type="$GPU_MAIL_EVENTS"
        --export="ALL,RUN_TAG=${RUN_TAG}"
        --output="$TASK_ROOT/task_%a/$TASK_LOG_NAME"
        --error="$TASK_ROOT/task_%a/$TASK_LOG_NAME"
        --open-mode=append
    )
    if [[ -n "$MAIL_USER" ]]; then options+=(--mail-user="$MAIL_USER"); fi
    if [[ -n "$PREV_JOBID" ]]; then
        options+=(--dependency="afterany:${PREV_JOBID}")
    fi
    if ! JOBID="$(sbatch "${options[@]}" "$MAIN" --worker)"; then
        echo "FAILED to submit channel $channel; preceding arrays remain submitted." >&2
        echo "Run directory: $RUN_DIR" >&2
        exit 6
    fi
    JOBID="${JOBID%%;*}"
    [[ "$JOBID" =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch job ID: $JOBID" >&2; exit 6; }
    echo "Submitted channel $((channel+1))/$CHANNEL_COUNT: job $JOBID, array ${first}-${last}, four GPUs"
    PREV_JOBID="$JOBID"
done

# afterany means the finalizer always writes the combined log, even if a GPU
# task failed. It only merges after validating ALL expected TIFFs.
final_options=(
    --parsable
    --job-name="$FINAL_JOB_NAME"
    --partition="$FINAL_PARTITION"
    --qos="$FINAL_QOS"
    --cpus-per-task="$FINAL_CPUS"
    --mem="$FINAL_MEMORY"
    --time="$FINAL_TIME"
    --mail-type="$FINAL_MAIL_EVENTS"
    --dependency="afterany:${PREV_JOBID}"
    --export="ALL,RUN_TAG=${RUN_TAG},ARRAY_JOBID=${PREV_JOBID}"
    --output="$RUN_DIR/$FINAL_LOG_NAME"
    --error="$RUN_DIR/$FINAL_LOG_NAME"
)
if [[ -n "$MAIL_USER" ]]; then final_options+=(--mail-user="$MAIL_USER"); fi
if FINAL_JOBID="$(sbatch "${final_options[@]}" "$MAIN" --finalize)"; then
    FINAL_JOBID="${FINAL_JOBID%%;*}"
    echo "Submitted $TOTAL_TASKS GPU tasks in $CHANNEL_COUNT sequential channel array(s)."
    echo "Submitted dependent finalizer: $FINAL_JOBID"
else
    echo "WARNING: GPU arrays were submitted, but automatic finalizer submission failed." >&2
    echo "Submit the finalizer manually with --dependency=afterany:${PREV_JOBID}" >&2
    exit 7
fi
echo "Run directory: $RUN_DIR"
echo "Task directory: $TASK_ROOT"
echo "Each GPU task log: $TASK_ROOT/task_0/$TASK_LOG_NAME through task_$((TOTAL_TASKS-1))/$TASK_LOG_NAME"
echo "After all tasks finish: $COMBINED_LOG_NAME and $FINAL_LOG_NAME"
echo 'No GPU memory CSV is generated.'
}

worker_main() {
module load "$ANACONDA_MODULE"
source activate "$CONDA_ENV"

ROOT="$PROJECT_ROOT"
D="$PYTHON_DIR"
PREC=fp32
TASK="${SLURM_ARRAY_TASK_ID:?Submit as sbatch array task}"
JOBID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-unknown}}"
RUN_TAG="${RUN_TAG:-job_${JOBID}}"
# Global task IDs keep each of the per-channel workers in a separate task_N directory.
[[ "$RUN_TAG" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "Invalid RUN_TAG; use letters/numbers/._- only" >&2; exit 2; }
OUT="$OUTPUT_DIR/leonardo_main_${RUN_TAG}"
TASK_ROOT="$(task_root_for_run "$OUT")"
TASKDIR="$TASK_ROOT/task_${TASK}"
CHUNKS="$TASKDIR/chunks"
CONFIGS="$TASKDIR/configs"
SOURCE="$OUT/leonardo_settings.json"
KEEP_RAW="$(python3 -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["output"].get("save_before_postprocessing", False)).lower())' "$SOURCE")"

cleanup_raw_after_success() {
    if [[ "$KEEP_RAW" != "true" ]]; then
        rm -f -- "$RAW"
    fi
    rm -f -- "${RAW%.tif}.json" "${RAW%.tif}__work.npy" "${RAW%.tif}__done.npy"
}
CORE="$TRAINING_PY"
PIPE="$PIPELINE_PY"

export PYTHONPATH="$ROOT:$ROOT/leonardo_toolset:$D${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE="$JAX_PREALLOCATE"
export XLA_PYTHON_CLIENT_MEM_FRACTION="$JAX_MEMORY_FRACTION"
export MPLBACKEND=Agg
export JAX_COMPILATION_CACHE_DIR="$JAX_CACHE_DIR"
mkdir -p "$JAX_COMPILATION_CACHE_DIR" "$CHUNKS" "$CONFIGS"
for required in "$SOURCE" "$CORE" "$PIPE" \
  "$D/leonardo_alpha1_full256.py" "$D/loss_term_jax.py" \
  "$D/guided_filter_upsample_optimized.py"; do
    [[ -f "$required" ]] || { echo "MISSING: $required" >&2; exit 3; }
done
# Protect against simultaneous jobs attempting same chunk output.
exec 9>"$TASKDIR/.worker.lock"
flock -n 9 || { echo "Another worker already owns $TASKDIR" >&2; exit 4; }

PLANNED="$(python3 -u "$PIPE" plan prepare --source "$SOURCE" --task "$TASK" \
    --task-dir "$TASKDIR" --precision "$PREC")"
mapfile -t PARTS <<< "$PLANNED"
FIRST="${PARTS[0]}"; LASTEXCL="${PARTS[1]}"
HEIGHT="${PARTS[2]}"; WIDTH="${PARTS[3]}"
BLOCKS="${PARTS[4]}"; INPUT_STEM="${PARTS[5]}"
TAG_FIRST="$(printf '%04d' "$FIRST")"
TAG_LAST="$(printf '%04d' "$((LASTEXCL-1))")"
FINAL="$TASKDIR/${INPUT_STEM}_z${TAG_FIRST}_${TAG_LAST}_alpha1_fftwidth5_uint16.tif"

if (( LASTEXCL == FIRST )); then
    echo "TASK $TASK assigned zero planes for its channel; done"
    exit 0
fi

# The core and loss import must work with GPU JAX; never silently use torch.
python3 - "$CORE" "$SOURCE" "$FIRST" "$LASTEXCL" "$HEIGHT" "$WIDTH" <<'PREFLIGHT'
import importlib.util,json,sys,jax
from leonardo_toolset.destripe.loss_term_jax import Loss_jax
from guided_filter_upsample_optimized import GuidedUpsample
core,source,first,last,h,w=sys.argv[1],sys.argv[2],*map(int,sys.argv[3:])
c=json.load(open(source))
assert c['training']['backend']=='jax'
assert jax.default_backend()=='gpu',f'JAX default backend is not GPU: {jax.default_backend()}'
spec=importlib.util.spec_from_file_location('leo_dynamic_core_check',core)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
assert m.jax_flag==1,'JAX unavailable; torch fallback forbidden'
print(f'PREFLIGHT OK: JAX GPU, original loss, epochs=150, fresh initialization, '
      f'Z[{first}:{last}], YX={h}x{w}',flush=True)
PREFLIGHT

if [[ -e "$FINAL" ]]; then
    python3 - "$FINAL" "$((LASTEXCL-FIRST))" "$HEIGHT" "$WIDTH" <<'CHECKFINAL'
import sys
from leonardo_pipeline import check_tiff
check_tiff(sys.argv[1],int(sys.argv[2]),(int(sys.argv[3]),int(sys.argv[4])))
print('Existing full TIFF validated; skipping completed task',flush=True)
CHECKFINAL
    exit 0
fi

echo "TASK=$TASK Z[$FIRST:$LASTEXCL] shape=($HEIGHT,$WIDTH), $BLOCKS chunks of <=8 planes, precision=$PREC"
echo "RUN ROOT: $OUT"
for ((b=0;b<BLOCKS;b++)); do
    A=$((FIRST+b*8))
    Z=$((A+8))
    (( Z <= LASTEXCL )) || Z="$LASTEXCL"
    TAG="z$(printf '%04d' "$A")_$(printf '%04d' "$((Z-1))")"
    CONFIG="$CONFIGS/$TAG.json"
    RESULT="$CHUNKS/alpha1w5_${TAG}.tif"
    DONE="$CHUNKS/${TAG}.validated.json"
    RAW="$(python3 -u "$PIPE" plan chunk-config --manifest "$TASKDIR/manifest.json" \
             --start "$A" --end "$Z" --config "$CONFIG" --chunks "$CHUNKS")"
    if [[ -f "$DONE" ]]; then
        python3 - "$RESULT" "$DONE" "$A" "$Z" "$HEIGHT" "$WIDTH" <<'CHECKCHUNK'
import json,sys
from leonardo_pipeline import check_tiff
res,done=sys.argv[1:3]
a,z,h,w=map(int,sys.argv[3:])
assert json.load(open(done))=={'start':a,'end':z,'alpha':1.0,'width':5.0}
check_tiff(res,z-a,(h,w))
CHECKCHUNK
        echo "REUSED validated chunk $TAG"
        continue
    fi
    if [[ -f "$RESULT" ]]; then
        python3 - "$RESULT" "$A" "$Z" "$HEIGHT" "$WIDTH" <<'RECOVER'
import sys
from leonardo_pipeline import check_tiff
p=sys.argv[1];a,z,h,w=map(int,sys.argv[2:])
check_tiff(p,z-a,(h,w))
print('Validated corrected chunk from interrupted marker step',flush=True)
RECOVER
        python3 - "$DONE" "$A" "$Z" <<'MARKRECOVER'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);a,z=map(int,sys.argv[2:]);p.write_text(json.dumps({'start':a,'end':z,'alpha':1.0,'width':5.0})+'\n')
MARKRECOVER
        cleanup_raw_after_success
        continue
    fi
    # Reuse a complete GPU-training output if the previous attempt failed
    # only during FFT postprocessing. Never delete a valid raw TIFF first.
    if [[ -f "$RAW" ]]; then
        echo "RECOVERY: found previous raw TIFF for $TAG; validating"
        if python3 - "$RAW" "$A" "$Z" "$HEIGHT" "$WIDTH" <<'CHECKRAW'
import sys
from leonardo_pipeline import check_tiff
path=sys.argv[1]
a,z,h,w=map(int,sys.argv[2:])
check_tiff(path,z-a,(h,w))
print('Previous raw TIFF validated',flush=True)
CHECKRAW
        then
            if python3 -u "$PIPE" postprocess \
                --raw "$RAW" --output "$RESULT" --config "$CONFIG"; then
                python3 - "$DONE" "$A" "$Z" <<'MARKRECOVEREDRAW'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);a,z=map(int,sys.argv[2:])
p.write_text(json.dumps({'start':a,'end':z,'alpha':1.0,'width':5.0})+'\n')
MARKRECOVEREDRAW
                cleanup_raw_after_success
                echo "RECOVERED $TAG from raw TIFF without GPU retraining"
                continue
            else
                echo "RECOVERY postprocessing failed for $TAG; raw TIFF retained" >&2
                exit 11
            fi
        else
            echo "Previous raw TIFF for $TAG is invalid; it must be regenerated" >&2
        fi
    fi
    export LEONARDO_TEST_CONFIG="$CONFIG"
    for ((attempt=1;attempt<=3;attempt++)); do
        echo "=== TASK $TASK BLOCK $b/$BLOCKS $TAG ATTEMPT $attempt START $(date -Is) ==="
        rm -f -- "$RAW" "${RAW%.tif}.json" "${RAW%.tif}__work.npy" \
          "${RAW%.tif}__done.npy" "$RESULT.partial"
        if python3 -u "$CORE"; then
            # If training succeeded, a deterministic postprocessing error is
            # not a reason to train the same eight planes three times.
            if ! python3 -u "$PIPE" postprocess \
                --raw "$RAW" --output "$RESULT" --config "$CONFIG"; then
                echo "POSTPROCESS FAILED for $TAG; completed raw TIFF retained" >&2
                exit 11
            fi
            python3 - "$DONE" "$A" "$Z" <<'MARK'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);a,z=map(int,sys.argv[2:])
p.write_text(json.dumps({'start':a,'end':z,'alpha':1.0,'width':5.0})+'\n')
MARK
            cleanup_raw_after_success
            echo "COMPLETED $TAG"
            break
        else
            code=$?
            echo "TRAINING FAILED $TAG attempt $attempt: code $code" >&2
            if ((attempt==3)); then
                echo "FAILED after 3 attempts; earlier corrected chunks retained" >&2
                exit 10
            fi
            sleep 5
        fi
    done
done
python3 -u "$PIPE" merge-task --task-dir "$TASKDIR" --output "$FINAL"
echo "FINISHED precision=$PREC task=$TASK output=$FINAL"
}

finalize_main() {

ROOT="$PROJECT_ROOT"
D="$PYTHON_DIR"
RUN_TAG="${RUN_TAG:?The launcher must set RUN_TAG}"
RUN_DIR="$OUTPUT_DIR/leonardo_main_${RUN_TAG}"
TASK_ROOT="$(task_root_for_run "$RUN_DIR")"
mkdir -p "$LOG_DIR"
ALL_LOG="$LOG_DIR/leonardo_main_${RUN_TAG}.log"
CHANNEL_COUNT="$(python3 "$PIPELINE_PY" plan count --source "$RUN_DIR/leonardo_settings.json")"
TOTAL_TASKS=$((CHANNEL_COUNT * GPU_TASKS_PER_CHANNEL))

# The tasks ran concurrently: group their output by task to keep traceback and
# progress context readable; the resulting file is NOT a time-sorted trace.
{
    printf 'LEONARDO RUN: %s\n' "$RUN_DIR"
    printf 'Last GPU array job: %s\n' "${ARRAY_JOBID:-unknown}"
    printf '%s sequential channels, four parallel GPU tasks each.\n' "$CHANNEL_COUNT"
    printf 'Each task section is grouped; this is not a chronological log.\n'
    for ((t=0;t<TOTAL_TASKS;t++)); do
        printf '\n========== TASK %d ==========\n' "$t"
        if [[ -f "$TASK_ROOT/task_${t}/$TASK_LOG_NAME" ]]; then
            sed 's/\r/\n/g' "$TASK_ROOT/task_${t}/$TASK_LOG_NAME"
        else
            printf 'NO TASK LOG FOUND for task_%d\n' "$t"
        fi
    done
} > "$ALL_LOG"
echo "Combined GPU task logs: $ALL_LOG"
module load "$ANACONDA_MODULE"
source activate "$CONDA_ENV"
export PYTHONPATH="$ROOT:$ROOT/leonardo_toolset:$D${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg

MERGE="$(python3 - "$RUN_DIR/leonardo_settings.json" <<'READOPTION'
import json,sys
cfg=json.load(open(sys.argv[1]))
value=cfg['automation']['merge_after_success']
if not isinstance(value,bool):
    raise SystemExit('leonardo_settings.json: automation.merge_after_success must be a JSON boolean')
print('1' if value else '0')
READOPTION
)"

echo "Checking all $TOTAL_TASKS task manifests and finished uint16 TIFFs, per enabled channel..."
if ! python3 -u "$D/leonardo_pipeline.py" merge-all "$RUN_DIR" --dry-run 2>&1 | tee -a "$ALL_LOG"; then
    echo 'At least one task TIFF is incomplete/invalid. Merge SKIPPED; all data preserved.' | tee -a "$ALL_LOG" >&2
    exit 10
fi

if [[ "$MERGE" == 1 ]]; then
    echo 'Auto-merge enabled. Merging Z within each channel; different channels stay separate.' | tee -a "$ALL_LOG"
    python3 -u "$D/leonardo_pipeline.py" merge-all "$RUN_DIR" 2>&1 | tee -a "$ALL_LOG"
    echo 'AUTO-MERGE FINISHED. All original task TIFFs remain untouched.' | tee -a "$ALL_LOG"
else
    echo "Auto-merge disabled. All $TOTAL_TASKS task TIFFs validated and kept separately by channel." | tee -a "$ALL_LOG"
fi
}

case "${1:-}" in
    "") submit_main ;;
    --worker) worker_main ;;
    --finalize) finalize_main ;;
    *) printf 'Usage: bash leonardo_main.sh\n' >&2; exit 2 ;;
esac

