import plotly.express as px
import pandas as pd

def plot_histogram(expert_counts: dict):
    if not isinstance(expert_counts, dict):
        raise ValueError("expert_counts must be a dictionary")
    data = []
    for layer_idx, counts in expert_counts.items():
        for expert in counts:
            data.append({
                "layer_idx": layer_idx,
                "expert": expert,
                "count": counts[expert]
            })
    df = pd.DataFrame(data)

    max_layer_idx = int(df["layer_idx"].max())
    max_expert = int(df["expert"].max())

    fig = px.density_heatmap(df, x="expert", y="layer_idx", z="count", nbinsx=max_expert+1, nbinsy=max_layer_idx+1, histfunc="sum")
    return fig
