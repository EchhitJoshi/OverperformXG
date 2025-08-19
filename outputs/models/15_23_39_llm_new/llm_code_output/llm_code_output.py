

# --- New generated code block ---
import plotly.express as px

# Filter data for Oliver Boscagli
player_data = dat[dat["player_name"] == "Oliver Boscagli"]

# Create the KDE plot for games_rating over year_e
fig = px.density_contour(
    player_data, 
    x="year_e", 
    y="games_rating", 
    title="KDE Plot of Games Rating by Year for Oliver Boscagli", 
    labels={"year_e": "Year", "games_rating": "Games Rating"},
    color_continuous_scale="Viridis"
)

fig.update_traces(contours_coloring="fill", showlabels=True)
fig.show()
# --- End of generated code block ---


# --- New generated code block ---
import plotly.express as px
import pandas as pd

# Filter the dataset for the player "Oliver Boscagli"
filtered_data = dat[dat['player_name'] == 'Oliver Boscagli']

# Ensure 'year_e' is treated as a numerical column, if needed
filtered_data['year_e'] = pd.to_numeric(filtered_data['year_e'], errors='coerce')

# Plot the KDE plot using plotly
fig = px.histogram(
    filtered_data,
    x="games_rating",
    color="year_e",
    marginal="violin",  # Letting values explored nicely 
 graph for interpretation 
 )
fig.update_layout else ..!
# --- End of generated code block ---


# --- New generated code block ---
To create a Kernel Density Estimation (KDE) plot for the `games_rating` column by year (filtered for "Oliver Boscagli"), I will use Plotly's `kdeplot` via `px.density_contour` with a single variable for KDE.

Here is the code:

```python
import plotly.express as px

# Filter data for Oliver Boscagli
filtered_data = dat[dat['player_name'] == 'Oliver Boscagli']

# Create the KDE plot
fig = px.density_contour(
    data_frame=filtered_data,
    x="year_e",
    y="games_rating",
    marginal_x=None,
    marginal_y=None,
    color="keyapor viensiryoring modifying Outco,"
)
But Note axs,p will better non axy<>


But
# --- End of generated code block ---


# --- New generated code block ---
import plotly.figure_factory as ff
import pandas as pd

# Filter the dataset for Oliver Boscagli
filtered_dat = dat[dat['player_name'] == 'Oliver Boscagli']

# Extract the relevant data
games_ratings = filtered_dat['games_rating']
years = filtered_dat['year_e']

# Create the KDE plot using Plotly
fig = ff.create_2d_density(
    x=years,
    y=games_ratings,
    colorscale="Viridis",
    hist_color="rgba(0,0,255,0.1)",
    point_size=3,
)

# Update the plot layout
fig.update_layout(
    title="KDE Plot of Games Rating by Year for Oliver Boscagli",
    xaxis_title="Year",
    yaxis_title="Games Rating",
    template="plotly_white"
)

# Show the plot
fig.show()
# --- End of generated code block ---
