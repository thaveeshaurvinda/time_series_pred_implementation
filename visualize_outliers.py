import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def generate_outlier_boxplot(filepath = "data/energy_data_set.csv"):
    """
    1. Loads the dataset
    2. Creates a boxplot of appliance energy usage
    3. Calculates important statistics
    4. Displays outlier information
    5. Saves the graph in the reports folder
    """
    os.makedirs("reports", exist_ok = True)
    print("Loading raw dataset for outlier analysis...")
    data = pd.read_csv(filepath)
    plt.figure(figsize = (10, 5))

    sns.set_theme(style = "whitegrid")

    boxplot = sns.boxplot(
        # Target column
        x = data["Appliances"],
        color = "dodgerblue",
        flierprops = {
            "marker": "x",
            "markerfacecolor": "crimson",
            "markeredgecolor": "crimson",
            "alpha": 0.5
        },
        linewidth = 2,
        notch = True
    )

    plt.title("Distribution of raw appliance energy consumption", fontsize = 14, fontweight = "bold", pad = 15)
    plt.xlabel("Energy consumption (Wh)", fontsize = 12, labelpad = 10)

    # Calculate important statistics
    # First Quartile (25%)
    q1 = data["Appliances"].quantile(0.25)
    # Third Quartile (75%)
    q3 = data["Appliances"].quantile(0.75)
    # Median (50%)
    median = data["Appliances"].median()
    # Interquartile Range
    iqr = q3 - q1
    # Upper Outlier Boundary
    upper_whisker = q3 + (1.5 * iqr)
    # Maximum value in dataset
    maximum_value = data["Appliances"].max()
    plt.text(
        upper_whisker + 30,
        0.2,
        f"Outlier Threshold (Upper Whisker): "
        f"{upper_whisker:.0f} Wh\n"
        f"Max Spike: {maximum_value} Wh",
        color = "crimson",
        weight = "semibold",
        bbox = dict(
            facecolor = "white",
            alpha = 0.8,
            edgecolor = "crimson"
        )
    )

    # Add median label
    plt.text(
        median,
        -0.25,
        f"Median: {median:.0f} Wh",
        color = "black",
        weight = "bold",
        horizontalalignment = "center"
    )

    output_file = ("reports/outlier_boxplot.png")
    plt.savefig(output_file, bbox_inches = "tight", dpi = 300)
    plt.close()
    print(f"Success! Outlier boxplot saved to: {output_file}")

if __name__ == "__main__":
    generate_outlier_boxplot()

