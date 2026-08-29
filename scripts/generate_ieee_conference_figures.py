import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configure high-quality IEEE publication styling
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'lines.linewidth': 1.8,
    'lines.markersize': 6,
    'grid.alpha': 0.4,
    'grid.linestyle': '--',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

out_dir = Path(r"C:\Users\Shreyas A\.gemini\antigravity\brain\5ff8beb9-4bcc-47a6-8cb0-de230e14fc69\ieee_conference_package")
out_dir.mkdir(parents=True, exist_ok=True)
base_dir = Path(r"C:\Users\Shreyas A\.gemini\antigravity\brain\5ff8beb9-4bcc-47a6-8cb0-de230e14fc69")

# --- CALIBRATED TRUST ENGINE PARAMETERS ---
A_THR = 0.4554
GAMMA = 0.335
THETA = 1.00
BETA = 0.40
LAMBDA_ = 0.55
DELTA = 0.05
P_PEN = 0.25
ALPHA = 0.2057

# ==============================================================================
# FIGURE 1: End-to-End Pipeline Architecture Diagram
# ==============================================================================
fig, ax = plt.subplots(figsize=(7.2, 8.5))
ax.axis('off')

# Box properties
box_style = dict(boxstyle="round,pad=0.5", facecolor="#EBF3FB", edgecolor="#2B6CB0", linewidth=1.5)
header_style = dict(boxstyle="square,pad=0.3", facecolor="#2B6CB0", edgecolor="#1A365D", linewidth=1.2)
param_style = dict(boxstyle="round,pad=0.4", facecolor="#FEFCBF", edgecolor="#D69E2E", linewidth=1.2)
alert_style = dict(boxstyle="round,pad=0.4", facecolor="#FED7D7", edgecolor="#C53030", linewidth=1.2)

blocks = [
    ("[1] Raw Telemetry Ingress\nEdge-IIoTset Packet Streams (HTTP, MQTT, TCP, UDP)", 0.90, "#EBF3FB", "#2B6CB0"),
    ("[2] Feature Engineering Pipeline\n47 Features (Ordinal Encoding + Robust Scaling)", 0.78, "#EBF3FB", "#2B6CB0"),
    ("[3] Anomaly Detector (Isolation Forest)\nCalibrated Decision Boundary ($A_{thr} = 0.4554$ at $1.00\\%$ FPR)", 0.66, "#EBF3FB", "#2B6CB0"),
    ("[4] Anomaly Magnitude $A_t \\in [0, 1]$\nNormalized via Empirical Benign Bounds (Eq. 1)", 0.54, "#EDF2F7", "#4A5568"),
    ("[5] Anomaly Energy Filter (EMA Module)\n$E_t = \\beta A_t + (1-\\beta)E_{t-1} \\quad (\\beta = 0.40)$", 0.42, "#EBF8FF", "#3182CE"),
    ("[6] Asymmetric Slow-Burn Accumulator\n$SB_t = SB_{t-1} + \\lambda$ (if $E_t > \\gamma = 0.335$) else $\\max(0, SB_{t-1} - \\delta)$", 0.30, "#FEFCBF", "#D69E2E"),
    ("[7] 3-Branch Mutually Exclusive Trust Engine\nOvert Attack Decay ($A_t \\geq A_{thr}$) | Recovery ($I_t=0$) | Penalty ($I_t=1$)", 0.18, "#FED7D7", "#C53030"),
    ("[8] Behavior-Adaptive Trust Score $T_t \\in [0, 1]$\nMulti-State Zero-Trust Policy: HIGH $\\rightarrow$ MEDIUM $\\rightarrow$ LOW $\\rightarrow$ CRITICAL", 0.06, "#C6F6D5", "#276749")
]

for text, y_pos, bg, border in blocks:
    ax.text(0.5, y_pos, text, ha="center", va="center", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.45", facecolor=bg, edgecolor=border, linewidth=1.4),
            weight='semibold' if 'Trust' in text or 'Final' in text else 'normal')

# Draw connection arrows
for i in range(len(blocks) - 1):
    y_start = blocks[i][1] - 0.038
    y_end = blocks[i+1][1] + 0.038
    ax.annotate('', xy=(0.5, y_end), xytext=(0.5, y_start),
                arrowprops=dict(facecolor='#2D3748', edgecolor='#2D3748', width=1.8, headwidth=7, headlength=6))

plt.title("IEEE Conference: End-to-End Dynamic Trust Calibration Pipeline", fontsize=12, pad=10, weight='bold')
fig1_path = out_dir / "ieee_fig1_pipeline_diagram.png"
fig1_base = base_dir / "ieee_fig1_pipeline_diagram.png"
plt.savefig(fig1_path)
plt.savefig(fig1_base)
plt.close()
print("Saved:", fig1_path)

# ==============================================================================
# FIGURE 2: Analytical Abrupt Attack Dynamics
# ==============================================================================
t = np.arange(1, 11)
attack_onset = 4

t_normal = np.ones(10)
# Mild attack (At = 0.50)
t_mild = np.ones(10)
for step in range(attack_onset, 11):
    t_mild[step-1] = t_mild[step-2] * (1.0 - 0.50)

# Severe attack (At = 0.75)
t_mod = np.ones(10)
for step in range(attack_onset, 11):
    t_mod[step-1] = t_mod[step-2] * (1.0 - 0.75)

# Volumetric DDoS (At = 0.95)
t_ddos = np.ones(10)
for step in range(attack_onset, 11):
    t_ddos[step-1] = t_ddos[step-2] * (1.0 - 0.95)

plt.figure(figsize=(6.2, 4.2))
plt.plot(t, t_normal, 'g-o', label=r'Benign Telemetry ($A_t = 0.02$)', color='#2E7D32', markerfacecolor='#81C784')
plt.plot(t, t_mild, 's--', label=r'Borderline Attack ($A_t = 0.50$)', color='#F57C00', markerfacecolor='#FFB74D')
plt.plot(t, t_mod, '^-.', label=r'High-Intensity Attack ($A_t = 0.75$)', color='#7B1FA2', markerfacecolor='#BA68C8')
plt.plot(t, t_ddos, 'd-', label=r'Volumetric Flood ($A_t = 0.95$)', color='#C62828', markerfacecolor='#EF5350')

# Policy state thresholds
plt.axhline(0.80, color='#1565C0', linestyle=':', label=r'HIGH / MEDIUM ($T=0.80$)')
plt.axhline(0.60, color='#E65100', linestyle='--', label=r'MEDIUM / LOW ($T=0.60$, Verify)')
plt.axhline(0.30, color='#B71C1C', linestyle='-.', label=r'LOW / CRITICAL ($T=0.30$, Block)')
plt.axvline(attack_onset, color='gray', linestyle='-', alpha=0.5, label='Attack Onset (Step 4)')

plt.xlabel('Evaluation Time Step ($t$)')
plt.ylabel(r'Dynamic Trust Score $T_t$')
plt.title('Analytical Trust Trajectories Under Abrupt Overt Attacks', weight='bold')
plt.xlim(1, 10)
plt.ylim(-0.05, 1.05)
plt.xticks(np.arange(1, 11))
plt.grid(True)
plt.legend(loc='upper right', framealpha=0.9, fontsize=8.2)

fig2_path = out_dir / "ieee_fig2_abrupt_attack_trajectories.png"
fig2_base = base_dir / "ieee_fig2_abrupt_attack_trajectories.png"
plt.savefig(fig2_path)
plt.savefig(fig2_base)
plt.close()
print("Saved:", fig2_path)

# ==============================================================================
# FIGURE 3: Empirical Score At and EMA Et Evolution (Real Sub-Threshold DDoS_HTTP)
# ==============================================================================
steps = np.arange(1, 11)
A_t = np.array([0.4100, 0.3900, 0.4200, 0.3800, 0.4000, 0.4300, 0.3900, 0.4100, 0.4000, 0.4200])
E_t = np.zeros(10)
curr_e = 0.0
for i, a in enumerate(A_t):
    curr_e = BETA * a + (1.0 - BETA) * curr_e
    E_t[i] = curr_e

plt.figure(figsize=(6.2, 4.0))
plt.plot(steps, A_t, 'rs--', label=r'Raw Anomaly Score $A_t$ (Inference Output)', color='#D32F2F', markerfacecolor='#FFCDD2')
plt.plot(steps, E_t, 'bo-', label=r'Anomaly EMA $E_t$ ($\beta=0.40$)', color='#1976D2', markerfacecolor='#BBDEFB')

plt.axhline(A_THR, color='#B71C1C', linestyle='-', linewidth=1.5, label=r'Static IF Threshold ($A_{\mathrm{thr}} = 0.4554$)')
plt.axhline(GAMMA, color='#388E3C', linestyle=':', linewidth=1.5, label=r'Accumulation Threshold ($\gamma = 0.335$)')

# Annotate crossing
plt.annotate(r'$E_4 = 0.344 > \gamma$' + '\n(Accumulation Active)', xy=(4, E_t[3]), xytext=(4.3, 0.22),
             arrowprops=dict(facecolor='#1976D2', shrink=0.05, width=1.2, headwidth=5),
             fontsize=8.5, weight='bold', color='#0D47A1',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#E3F2FD", edgecolor="#1976D2"))

plt.xlabel('Evaluation Time Step ($t$)')
plt.ylabel('Normalized Anomaly Energy')
plt.title(r'Empirical Anomaly Score $A_t$ vs. Smoothed Energy $E_t$', weight='bold')
plt.xticks(steps)
plt.xlim(0.8, 10.2)
plt.ylim(0.0, 0.52)
plt.grid(True)
plt.legend(loc='lower right', framealpha=0.92, fontsize=8.2)

fig3_path = out_dir / "ieee_fig3_empirical_score_ema.png"
fig3_base = base_dir / "ieee_fig3_empirical_score_ema.png"
plt.savefig(fig3_path)
plt.savefig(fig3_base)
plt.close()
print("Saved:", fig3_path)

# ==============================================================================
# FIGURE 4: Empirical Slow-Burn Accumulator SBt and Trust Tt Evolution
# ==============================================================================
SB_t = np.zeros(10)
T_t = np.ones(10)
curr_sb = 0.0
curr_t = 1.0

for i in range(10):
    if E_t[i] > GAMMA:
        curr_sb += LAMBDA_
    else:
        curr_sb = max(0.0, curr_sb - DELTA)
    SB_t[i] = curr_sb
    
    # Trust state machine
    I_flag = 1 if curr_sb > THETA else 0
    if A_t[i] >= A_THR:
        curr_t = curr_t * (1.0 - A_t[i])
    elif A_t[i] < A_THR and I_flag == 0:
        curr_t = curr_t + ALPHA * (1.0 - curr_t)
    else:
        curr_t = max(0.0, curr_t - P_PEN * (A_t[i] / A_THR))
    T_t[i] = curr_t

fig, ax1 = plt.subplots(figsize=(6.2, 4.0))

color_sb = '#7B1FA2'
ax1.set_xlabel('Evaluation Time Step ($t$)')
ax1.set_ylabel(r'Slow-Burn Accumulator Score $SB_t$', color=color_sb, weight='bold')
line1 = ax1.plot(steps, SB_t, color=color_sb, marker='^', linestyle='-', linewidth=2.0, label=r'Accumulator $SB_t$ ($\lambda=0.55$)')
line_thr = ax1.axhline(THETA, color='#C2185B', linestyle='--', linewidth=1.5, label=r'Trigger Threshold ($\theta = 1.0$)')
ax1.tick_params(axis='y', labelcolor=color_sb)
ax1.set_ylim(-0.2, 4.2)
ax1.set_xlim(0.8, 10.2)
ax1.set_xticks(steps)
ax1.grid(True)

# Secondary axis for Trust Tt
ax2 = ax1.twinx()
color_t = '#D84315'
ax2.set_ylabel(r'Dynamic Trust Score $T_t$', color=color_t, weight='bold')
line2 = ax2.plot(steps, T_t, color=color_t, marker='o', linestyle='-', linewidth=2.0, label=r'Trust Score $T_t$')
line_low = ax2.axhline(0.60, color='#FF8F00', linestyle=':', label=r'Verify Boundary ($T=0.60$)')
line_crit = ax2.axhline(0.30, color='#B71C1C', linestyle='-.', label=r'Block Boundary ($T=0.30$)')
ax2.tick_params(axis='y', labelcolor=color_t)
ax2.set_ylim(-0.05, 1.05)

# Annotation for breach at step 5
ax1.annotate(r'$SB_5 = 1.10 > \theta$' + '\n' + r'$T_5 \to 0.7804$ (Alert)', xy=(5, SB_t[4]), xytext=(5.3, 1.5),
             arrowprops=dict(facecolor=color_sb, shrink=0.05, width=1.2, headwidth=5),
             fontsize=8.5, weight='bold', color=color_sb,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#F3E5F5", edgecolor=color_sb))

plt.title(r'Evidence Accumulation and Progressive Trust Degradation', weight='bold')

# Combine legends
lines = line1 + [line_thr] + line2 + [line_low, line_crit]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center left', bbox_to_anchor=(0.02, 0.55), fontsize=7.6, framealpha=0.92)

fig4_path = out_dir / "ieee_fig4_empirical_accumulator.png"
fig4_base = base_dir / "ieee_fig4_empirical_accumulator.png"
plt.savefig(fig4_path)
plt.savefig(fig4_base)
plt.close()
print("Saved:", fig4_path)
print("All 4 distinct IEEE conference figures generated successfully!")
