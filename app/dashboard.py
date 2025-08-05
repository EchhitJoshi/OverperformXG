import dash
from dash import html, dcc, Input, Output, callback
import dash_bootstrap_components as dbc

from datetime import datetime,timedelta

from visualizations import *
from models import *

import arviz as az
import pymc as pm
import yaml


#Configs
px.defaults.template = "plotly_dark"


with open("config.yaml","r") as file:
    config = yaml.safe_load(file)

home_dir = config['HOME_DIRECTORY']


#### Data Reads ####




#### Main App ####
app = dash.Dash(__name__, external_stylesheets = [dbc.themes.CYBORG])

app.layout = dbc.Container([

    html.H1("Football Scout Report",className = 'text-center text-dark my-4'),

    html.Br(),
    html.Br(),

    dbc.Container(dcc.Dropdown(id = 'player_select', placeholder = 'Please select a player for analysis'))

    



],fluid = True)



#### Callbacks ####







