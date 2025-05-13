
import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output
import plotly.express as px

# Load main data and recommender files
df = pd.read_csv("lda_labeled_reviews.csv")
df = df.rename(columns={'lda_dominant_topic': 'dominant_topic'})
cosine_sim = np.load("cosine_sim.npy")
indices = pd.read_pickle("place_indices.pkl")

# Recommendation function
def recommend_places(place_name, cosine_sim=cosine_sim, indices=indices, df=df):
    if place_name not in indices:
        return ["(No similar bars found)"]
    idx = indices[place_name]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    place_indices = [i[0] for i in sim_scores]
    return df['place_name'].drop_duplicates().iloc[place_indices].tolist()

# Dash App
import dash_bootstrap_components as dbc

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    html.H1("🍸 Manhattan Bars Review Dashboard", className="text-center my-4"),

    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                id='place-dropdown',
                options=[{'label': name, 'value': name} for name in sorted(df['place_name'].dropna().unique())],
                placeholder="🔍 Select a Bar",
                className="mb-4"
            )
        ])
    ]),

    dbc.Row([
        dbc.Col(dcc.Graph(id='topic-counts'), md=4),
        dbc.Col(dcc.Graph(id='avg-rating-topic'), md=4),
        dbc.Col(dcc.Graph(id='rating-vs-topic'), md=4),
    ], className="mb-4"),

    dbc.Row([
        dbc.Col(html.Div(id='recommender-output', className="alert alert-info", style={'fontSize': 16}))
    ])
], fluid=True)

