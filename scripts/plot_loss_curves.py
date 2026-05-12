"""
Three plots in one figure:
- Left: V4 only (buggy run, very different scale)
- Middle: V5+V6 combined (corrected runs)
- Right: V5 vs V6 zoom-in last 20 epochs
"""
import pandas as pd
import matplotlib.pyplot as plt

v4 = pd.read_csv("v4_run/train_log_v4.csv")
v5 = pd.read_csv("v5_run/train_log_v5.csv")
v6 = pd.read_csv("v6_run/train_log_v6.csv")

v4_ep = v4.dropna(subset=['val_loss'])
v5_ep = v5.dropna(subset=['val_loss'])
v6_ep = v6.dropna(subset=['val_loss'])

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: V4 standalone (buggy, low values)
ax1.plot(v4_ep['epoch'], v4_ep['train_loss'], 'o-', label='Train', color='C0', markersize=4)
ax1.plot(v4_ep['epoch'], v4_ep['val_loss'], 's-', label='Val', color='C1', markersize=4)
ax1.set_title('V4 (ResNet18, with bugs)\n[mAP=0 hidden by buggy loss]')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: V5+V6 full training
ax2.plot(v5_ep['epoch'], v5_ep['train_loss'], 'o-', label='V5 Train', color='C2', markersize=4)
ax2.plot(v5_ep['epoch'], v5_ep['val_loss'], 's-', label='V5 Val', color='C3', markersize=4)
v6_cont = v6_ep[v6_ep['epoch'] > 50]
ax2.plot(v6_cont['epoch'], v6_cont['train_loss'], 'o-', label='V6 Train', color='C4', markersize=5)
ax2.plot(v6_cont['epoch'], v6_cont['val_loss'], 's-', label='V6 Val', color='C5', markersize=5)
ax2.axvline(x=50, color='gray', linestyle='--', alpha=0.5, label='V5 end / V6 start')
ax2.set_title('V5 + V6 (bugs fixed)\n[mAP=0.04 achieved]')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: zoom-in last epochs
combined = pd.concat([v5_ep, v6_ep[v6_ep['epoch'] > 50]]).sort_values('epoch')
last_20 = combined[combined['epoch'] >= 40]
ax3.plot(last_20['epoch'], last_20['train_loss'], 'o-', label='Train', color='C2', markersize=5)
ax3.plot(last_20['epoch'], last_20['val_loss'], 's-', label='Val', color='C3', markersize=5)
ax3.axvline(x=50, color='gray', linestyle='--', alpha=0.5)
ax3.set_title('Zoom: last 20 epochs\n[Showing plateau]')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Loss')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('artefacts/loss_comparison_v3.png', dpi=150, bbox_inches='tight')
print("Saved: artefacts/loss_comparison_v3.png")
plt.show()