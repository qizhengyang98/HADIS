import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# Config (edit as needed)
# -------------------------
# QPS for the 4 levels (phase 1..4). You can set these manually.
qps1 = 2
qps2 = 20
qps3 = 40
qps4 = 60

# Each phase lasts ~50 seconds; total = 7 * phase_len_s
phase_len_s = 50

# Request ID range for labeling synthetic requests
MAX_REQUEST_ID = 5000

# Output files
invocations_csv = 'step_invocations.csv'
invocations_plot = 'step_invocations.png'
trace_dir = 'trace'
trace_path = os.path.join(trace_dir, 'trace_step.txt')

# Optional: set a seed for reproducibility (comment out to randomize)
# np.random.seed(42)

# -------------------------
# Build the step-wise QPS series (per second)
# Order: 1-2-3-4-3-2-1
# -------------------------
phase_order = [qps1, qps2, qps3, qps4, qps3, qps2, qps1]
invocations_per_second = []
for qps in phase_order:
    invocations_per_second.extend([qps] * phase_len_s)

total_seconds = len(invocations_per_second)  # should be 7 * phase_len_s (default 350)
assert total_seconds == 7 * phase_len_s, "Unexpected total_seconds; check phase_len_s."

# -------------------------
# Save per-second invocations and plot
# -------------------------
df = pd.DataFrame({
    'second': np.arange(total_seconds),
    'invocations': invocations_per_second
})
df.to_csv(invocations_csv, index=False)

plt.figure(figsize=(9, 3))
plt.step(df['second'], df['invocations'], where='post')
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Demands (QPS)', fontsize=14)
plt.grid(True, linewidth=0.5)
plt.tight_layout()
plt.savefig(invocations_plot, dpi=150)
plt.close()

# -------------------------
# Generate a Poisson-arrival trace from the per-second QPS
# Within each second, we sample inter-arrivals ~ Exponential(scale=1000/requests) in ms
# -------------------------
os.makedirs(trace_dir, exist_ok=True)

trace_ms = []  # absolute arrival times in milliseconds
offset_ms = 0  # start of current second in ms

for sec_idx, qps in enumerate(invocations_per_second):
    requests = int(round(qps))

    if requests <= 0:
        # If you truly want to forbid zeros, you can raise as before.
        # raise Exception(f'No requests at second: {sec_idx}')
        offset_ms += 1000
        continue

    # Inter-arrival times in *milliseconds* within this second.
    # Exponential mean = 1000 / requests (so we expect 'requests' arrivals in 1s)
    inter_arrivals = np.random.exponential(scale=1000.0 / requests, size=requests)

    # Convert to integer ms (like your prior code); sort cumulative times
    inter_arrivals = np.rint(inter_arrivals).astype(int)

    current_ms = 0
    for delta in inter_arrivals:
        current_ms += int(delta)
        # Keep only arrivals that stay within the 1s window
        if current_ms < 1000:
            trace_ms.append(offset_ms + current_ms)

    offset_ms += 1000

# Write (timestamp_ms, request_id)
with open(trace_path, 'w') as wf:
    for t in trace_ms:
        wf.write(f"{t},{np.random.randint(1, MAX_REQUEST_ID)}\n")

print(f"Done. Wrote {invocations_csv}, {invocations_plot}, and {trace_path}.")
