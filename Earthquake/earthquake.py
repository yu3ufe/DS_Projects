import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Load the earthquake data
df = pd.read_csv("earthquake_data.csv")

# Convert the time column to datetime
df['time'] = pd.to_datetime(df['time'])

# Sort the DataFrame by time
df = df.sort_values('time')

def get_color(depth):
    """Maps earthquake depth to a color."""
    if depth < 70:
        return f'rgb(255, {int(255 - depth*3.64)}, 0)'
    elif depth < 300:
        return f'rgb({int(255 - (depth-70)*1.1)}, 0, 0)'
    else:
        return 'rgb(0, 0, 0)'

# Create the figure
fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scattergeo'}]])

# Add initial earthquake markers (start with the first 100 for performance)
initial_df = df.head(100)
fig.add_trace(go.Scattergeo(
    lon=initial_df['longitude'],
    lat=initial_df['latitude'],
    mode='markers',
    marker = dict(
        size = df['magnitude'] * 5,
        color = df['depth'],
        colorscale = 'Viridis',
        opacity = 0.8,
        colorbar = dict(title = 'Depth (km)')
    ),
    text=[f"Magnitude: {m}<br>Depth: {d} km<br>Time: {t}"
          for m, d, t in zip(initial_df['magnitude'], initial_df['depth'], initial_df['time'])],
    hoverinfo='text',
    name='Earthquakes'
))

fig.update_geos(
    projection_type="orthographic",
    showland=True,
    landcolor="LightGreen",
    showocean=True,
    oceancolor="LightBlue",
    showcountries=True,
    countrycolor="Black",
    showcoastlines=True,
    coastlinecolor="Black"
)

fig.update_layout(height=800, width=800, title_text="Global Earthquake Visualization")

# Create frames for animation
frames = []
for i in range(100, len(df), 100):  # Update frames every 100 data points
    frame_df = df.iloc[:i]
    frames.append(go.Frame(data=[go.Scattergeo(
        lon=frame_df['longitude'],
        lat=frame_df['latitude'],
        mode='markers',
        marker=dict(
            size=frame_df['magnitude'] * 4,  # Adjust scaling as needed
            color=[get_color(depth) for depth in frame_df['depth']],
            opacity=[max(0, min(1, (1 - (frame_df['time'].iloc[i-1] - t).total_seconds() / (30 * 24 * 60 * 60)))) for t in frame_df['time']],
            sizemode='diameter'
        ),
        text=[f"Magnitude: {m}<br>Depth: {d} km<br>Time: {t}"
              for m, d, t in zip(frame_df['magnitude'], frame_df['depth'], frame_df['time'])],
        hoverinfo='text'
    )]))

fig.frames = frames

fig.update_layout(
    updatemenus=[dict(
        type="buttons",
        buttons=[dict(label="Play",
                      method="animate",
                      args=[None, {"frame": {"duration": 100, "redraw": True},
                                   "fromcurrent": True,
                                   "transition": {"duration": 0}}]),
                 dict(label="Pause",
                      method="animate",
                      args=[[None], {"frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": 0}}])],
        direction="left",
        pad={"r": 10, "t": 87},
        showactive=False,
        x=0.1,
        xanchor="right",
        y=0,
        yanchor="top"
    )]
)

fig.show()
