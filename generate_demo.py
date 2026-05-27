import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ── Load & clean data ──────────────────────────────────────────────────────────
df = pd.read_csv("file.csv")

df["Education"] = df["Education"].astype(str).str.strip().replace(
    {"Phd": "PhD", "Ph.D": "PhD", "phd": "PhD", "Masters": "Master"}
)
df = df[df["Education"] != "X"]

# ── Encode for model ───────────────────────────────────────────────────────────
le = LabelEncoder()
df_enc = df.copy()
for col in ["Status", "Age_Group", "Education", "Industry", "Location", "Ai_Risk"]:
    df_enc[col] = le.fit_transform(df_enc[col].astype(str))

df_enc["Years_Of_Experience"] = pd.to_numeric(df_enc["Years_Of_Experience"], errors="coerce")
df_enc["Monthly_Salary_(Inr)"] = pd.to_numeric(df_enc["Monthly_Salary_(Inr)"], errors="coerce")
df_enc = df_enc.dropna()

X = df_enc.drop(["Status", "Monthly_Salary_(Inr)"], axis=1)
y = df_enc["Status"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

models = {
    "Decision\nTree": DecisionTreeClassifier(random_state=42),
    "Logistic\nRegression": LogisticRegression(max_iter=1000, random_state=42),
    "Random\nForest": RandomForestClassifier(random_state=42),
}
accuracies = {}
for name, m in models.items():
    m.fit(X_train, y_train)
    accuracies[name] = accuracy_score(y_test, m.predict(X_test)) * 100

# ── Figure layout ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 11), facecolor="#0f1117")
fig.suptitle(
    "Employment Classification Project  —  Demo Dashboard",
    fontsize=20, fontweight="bold", color="white", y=0.98,
)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35,
                       left=0.06, right=0.96, top=0.90, bottom=0.08)

DARK  = "#0f1117"
PANEL = "#1c1f2b"
ACCENT = ["#4fc3f7", "#81c784", "#ffb74d", "#f06292", "#ce93d8", "#80deea", "#ffcc80"]
TEXT  = "#e0e0e0"
GRID  = "#2a2d3a"

def style_ax(ax, title):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)

# ── 1. Employment Status ───────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
status_counts = df["Status"].value_counts()
wedge_props = {"linewidth": 2, "edgecolor": DARK}
ax1.pie(
    status_counts.values,
    labels=status_counts.index,
    autopct="%1.1f%%",
    colors=[ACCENT[0], ACCENT[1]],
    wedgeprops=wedge_props,
    textprops={"color": TEXT, "fontsize": 10},
    startangle=90,
)
ax1.set_facecolor(PANEL)
ax1.set_title("Employment Status", fontsize=12, fontweight="bold", color=TEXT, pad=10)
ax1.patch.set_facecolor(PANEL)

# ── 2. AI Risk Distribution ────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
risk_order = ["Low", "Moderate", "Medium", "High"]
risk_counts = df["Ai_Risk"].value_counts().reindex(risk_order).dropna()
bars = ax2.bar(risk_counts.index, risk_counts.values,
               color=ACCENT[:len(risk_counts)], edgecolor=DARK, linewidth=0.8)
ax2.set_ylabel("Count", color=TEXT)
ax2.yaxis.set_tick_params(labelcolor=TEXT)
ax2.xaxis.set_tick_params(labelcolor=TEXT)
ax2.set_facecolor(PANEL)
ax2.yaxis.grid(True, color=GRID, linestyle="--", alpha=0.6)
ax2.set_axisbelow(True)
for spine in ax2.spines.values():
    spine.set_edgecolor(GRID)
for bar, val in zip(bars, risk_counts.values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
             str(int(val)), ha="center", va="bottom", color=TEXT, fontsize=9)
ax2.set_title("AI Risk Distribution", fontsize=12, fontweight="bold", color=TEXT, pad=10)

# ── 3. Education Distribution ──────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
edu_counts = df["Education"].value_counts().head(6)
bars3 = ax3.barh(edu_counts.index[::-1], edu_counts.values[::-1],
                 color=ACCENT[2], edgecolor=DARK, linewidth=0.8)
ax3.set_xlabel("Count", color=TEXT)
ax3.tick_params(colors=TEXT, labelsize=9)
ax3.set_facecolor(PANEL)
ax3.xaxis.grid(True, color=GRID, linestyle="--", alpha=0.6)
ax3.set_axisbelow(True)
for spine in ax3.spines.values():
    spine.set_edgecolor(GRID)
ax3.set_title("Education Distribution", fontsize=12, fontweight="bold", color=TEXT, pad=10)

# ── 4. Avg Salary by Industry (top 8) ─────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
df["Monthly_Salary_(Inr)"] = pd.to_numeric(df["Monthly_Salary_(Inr)"], errors="coerce")
ind_salary = (
    df.dropna(subset=["Industry", "Monthly_Salary_(Inr)"])
    .groupby("Industry")["Monthly_Salary_(Inr)"]
    .mean()
    .sort_values(ascending=False)
    .head(8)
)
bars4 = ax4.bar(range(len(ind_salary)), ind_salary.values / 1000,
                color=ACCENT[4], edgecolor=DARK, linewidth=0.8)
ax4.set_xticks(range(len(ind_salary)))
ax4.set_xticklabels(ind_salary.index, rotation=35, ha="right", fontsize=8, color=TEXT)
ax4.set_ylabel("Avg Salary (₹ thousands)", color=TEXT)
ax4.tick_params(colors=TEXT)
ax4.set_facecolor(PANEL)
ax4.yaxis.grid(True, color=GRID, linestyle="--", alpha=0.6)
ax4.set_axisbelow(True)
for spine in ax4.spines.values():
    spine.set_edgecolor(GRID)
ax4.set_title("Avg Salary by Industry", fontsize=12, fontweight="bold", color=TEXT, pad=10)

# ── 5. Salary by Employment Status (box plot) ──────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
emp_sal  = df[df["Status"] == "Employed"]["Monthly_Salary_(Inr)"].dropna()
unemp_sal = df[df["Status"] == "Unemployed"]["Monthly_Salary_(Inr)"].dropna()

bp = ax5.boxplot(
    [emp_sal, unemp_sal],
    patch_artist=True,
    medianprops={"color": "white", "linewidth": 2},
    whiskerprops={"color": TEXT},
    capprops={"color": TEXT},
    flierprops={"marker": "o", "markerfacecolor": ACCENT[3], "markersize": 3, "alpha": 0.5},
)
for patch, color in zip(bp["boxes"], [ACCENT[0], ACCENT[1]]):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)

ax5.set_xticks([1, 2])
ax5.set_xticklabels(["Employed", "Unemployed"], color=TEXT, fontsize=10)
ax5.set_ylabel("Monthly Salary (₹)", color=TEXT)
ax5.tick_params(colors=TEXT)
ax5.set_facecolor(PANEL)
ax5.yaxis.grid(True, color=GRID, linestyle="--", alpha=0.6)
ax5.set_axisbelow(True)
for spine in ax5.spines.values():
    spine.set_edgecolor(GRID)
ax5.set_title("Salary by Employment Status", fontsize=12, fontweight="bold", color=TEXT, pad=10)

# ── 6. Model Accuracy Comparison ───────────────────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
model_names = list(accuracies.keys())
model_vals  = list(accuracies.values())
bar_colors  = [ACCENT[0], ACCENT[2], ACCENT[1]]
bars6 = ax6.bar(model_names, model_vals, color=bar_colors, edgecolor=DARK, linewidth=0.8, width=0.5)
ax6.set_ylim(0, 100)
ax6.axhline(50, color=ACCENT[3], linestyle="--", linewidth=1.2, alpha=0.7, label="50% baseline")
ax6.set_ylabel("Accuracy (%)", color=TEXT)
ax6.tick_params(colors=TEXT, labelsize=9)
ax6.set_facecolor(PANEL)
ax6.yaxis.grid(True, color=GRID, linestyle="--", alpha=0.6)
ax6.set_axisbelow(True)
for spine in ax6.spines.values():
    spine.set_edgecolor(GRID)
for bar, val in zip(bars6, model_vals):
    ax6.text(bar.get_x() + bar.get_width()/2, val + 1,
             f"{val:.1f}%", ha="center", va="bottom", color=TEXT, fontsize=9, fontweight="bold")
ax6.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=8)
ax6.set_title("Model Accuracy Comparison", fontsize=12, fontweight="bold", color=TEXT, pad=10)

# ── Save ───────────────────────────────────────────────────────────────────────
out = "demo_dashboard.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK)
print(f"Saved → {out}")
plt.show()
