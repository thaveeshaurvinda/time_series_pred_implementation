import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def run_comprehensive_eda(file_path, output_dir = "reports"):
    # Ensure reporting directory exists
    os.makedirs(output_dir, exist_ok = True)

    print("Loading dataset")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}. Please check your path structure.")

    df = pd.read_csv(file_path)
    df["date"] = pd.to_datetime(df["date"])
    df["hour"] = df["date"].dt.hour

    print(f"Dataset shape: {df.shape}")
    print(f"Columns detected: {list(df.columns)}\n")

    # Target variable skewness and outliers
    print("Generating target distribution")
    plt.figure(figsize = (12, 6))
    sns.set_theme(style = "whitegrid")

    # Create boxplot matching project specifications
    ax = sns.boxplot(x = df["Appliances"], color = "#3498db", flierprops = {"marker": "x", "markeredgecolor": "#e74c3c", "alpha": 0.5},)

    # Statistical markers
    median_val = df["Appliances"].median()
    q1 = df["Appliances"].quantile(0.25)
    q3 = df["Appliances"].quantile(0.75)
    iqr = q3 - q1
    upper_whisker = q3 + (1.5 * iqr)
    max_spike = df["Appliances"].max()

    plt.text(median_val, -0.2, f"Median: {median_val:.0f} Wh", color = "black", fontweight = "bold", ha = "center",)
    plt.title("Distribution Profile of Raw Appliance Energy Consumption", fontsize = 16, fontweight = "bold", pad = 15,)
    plt.xlabel("Energy Consumption (Wh)", fontsize = 12, labelpad = 10)

    props = dict(boxstyle = "square,pad = 0.5", facecolor = "white", edgecolor = "#e74c3c")
    ax.text(upper_whisker + 20, 0.1, f"Outlier Threshold (Upper Whisker): {upper_whisker:.0f} Wh\nMax Spike: {max_spike:.0f} Wh",color = "#e74c3c",fontweight = "bold", bbox = props,)

    plt.tight_layout()
    plt.savefig( os.path.join(output_dir, "outlier_boxplot.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Diurnal human behavior
    print("Generating diurnal hourly consumption")
    plt.figure(figsize = (12, 6))

    # Calculate hourly statistics
    hourly_stats = (df.groupby("hour")["Appliances"].agg(["mean", "median"]).reset_index())

    sns.lineplot(data=df, x = "hour", y = "Appliances", color = "#2c3e50", label = "Mean Consumption with 95% CI", linewidth = 2,)
    plt.plot(hourly_stats["hour"], hourly_stats["median"], color = "#e67e22", linestyle = "--", linewidth = 2, label = "Median Standby Baseline",)

    plt.title("Diurnal Hourly Trend Profile: Human Activity Cycles", fontsize = 16, fontweight = "bold", pad = 15,)
    plt.xlabel("Hour of Day (24h Scale)", fontsize = 12)
    plt.ylabel("Energy Consumption (Wh)", fontsize = 12)
    plt.xticks(range(0, 24))
    plt.legend(loc = "upper left", frameon=True)

    # Structural annotations for report documentation
    plt.axvspan(0, 6, color = "gray", alpha = 0.1)
    plt.text(3, df["Appliances"].mean() * 1.5, "Sleep Cycle\n(Standby)", color = "gray", ha = "center", style = "italic",)
    plt.axvspan(17, 21, color="red", alpha=0.05)
    plt.text(19, df["Appliances"].mean() * 1.5, "Peak Surge\n(Appliances Active)", color = "#c0392b", ha = "center", style = "italic",)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "diurnal_hourly_trends.png"), dpi = 300, bbox_inches = "tight",)
    plt.close()

    # Multivariate thermodynamic cross correlation
    print("Compiling Environmental Feature Cross-Correlation Matrix")
    plt.figure(figsize = (14, 10))

    target_features = [
        "Appliances",
        "T1",
        "RH_1",  # Kitchen
        "T2",
        "RH_2",  # Living Room
        "T3",
        "RH_3",  # Laundry
        "T_out",
        "Press_mm_hg",
        "Windspeed",
        "Tdewpoint",
    ]

    # Handle data boundaries cleanly if alternative datasets modify naming formats
    available_features = [col for col in target_features if col in df.columns]
    corr_matrix = df[available_features].corr()

    # Mask upper triangle for modern aesthetic finish
    mask = np.triu(np.ones_like(corr_matrix, dtype = bool))

    sns.heatmap( corr_matrix, mask = mask, annot = True, fmt = ".2f", cmap = "coolwarm", vmin = -1, vmax = 1, linewidths = 0.5, cbar_kws = {"label": "Pearson Correlation Coefficient ($r$)"},)

    plt.title("Thermodynamic Multi-Variable Interaction Matrix", fontsize = 16, fontweight = "bold", pad = 15,)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_correlation_heatmap.png"), dpi = 300, bbox_inches = "tight",)
    plt.close()

    # Microclimate internal vs external shifts
    print("Mapping Microclimate Ambient Envelope Tracking...")
    plt.figure(figsize = (12, 6))

    # Sample a clean 3-day continuous visual subset to inspect thermodynamic lag properties
    sample_sub = df.head(432)  # 432 steps * 10 mins = 3 days

    plt.plot(sample_sub["date"], sample_sub["T1"], color = "#e74c3c", label = "Indoor Kitchen Temp (T1)", linewidth = 1.5,)
    plt.plot(sample_sub["date"], sample_sub["T2"], color = "#e67e22", label = "Indoor Living Room Temp (T2)", linewidth = 1.5,)
    plt.plot(sample_sub["date"], sample_sub["T_out"], color = "#3498db", label = "Outdoor Ambient Temp (T_out)",linewidth = 2, linestyle = ":",)

    plt.title("Thermal Lag Diagnostics: Residential Envelope Response Profile", fontsize = 16, fontweight = "bold", pad = 15,)
    plt.xlabel("Chronological Sequence Timeline", fontsize = 12)
    plt.ylabel("Temperature (°C)", fontsize = 12)
    plt.legend(loc = "upper right", frameon = True)
    plt.xticks(rotation = 15)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "thermal_envelope_tracking.png"), dpi = 300, bbox_inches = "tight",)
    plt.close()
    print(f"\nEDA visualizations exported\n")


if __name__ == "__main__":
    DATA_PATH = os.path.join("../data", "energy_data_set.csv")
    run_comprehensive_eda(DATA_PATH)

