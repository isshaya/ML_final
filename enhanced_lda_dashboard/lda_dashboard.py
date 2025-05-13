import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import dash_bootstrap_components as dbc

# Load data
df = pd.read_csv("lda_labeled_reviews.csv")
df = df.rename(columns={'lda_dominant_topic': 'dominant_topic'})
cosine_sim = np.load("cosine_sim.npy")
indices = pd.read_pickle("place_indices.pkl")

# Optional: fake coordinates for demo (if not present)
if 'latitude' not in df.columns or 'longitude' not in df.columns:
    coords = {
        'Bar A': (40.7128, -74.0060),
        'Bar B': (40.7150, -74.0110),
        'Bar C': (40.7100, -74.0010)
    }
    df['latitude'] = df['place_name'].map(lambda x: coords.get(x, (40.713, -74.006))[0])
    df['longitude'] = df['place_name'].map(lambda x: coords.get(x, (40.713, -74.006))[1])

# Recommender function
def recommend_places(place_name, cosine_sim=cosine_sim, indices=indices, df=df):
    if place_name not in indices:
        return ["(No similar bars found)"]
    idx = indices[place_name]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    place_indices = [i[0] for i in sim_scores]
    return df['place_name'].drop_duplicates().iloc[place_indices].tolist()

# Initialize Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    dbc.Container([
        html.H1("🍸 Manhattan Bars Review Dashboard", className="text-center my-4"),

        dbc.Row([
            dbc.Col(dcc.Dropdown(
                id='place-dropdown',
                options=[{'label': name, 'value': name} for name in sorted(df['place_name'].dropna().unique())],
                placeholder="🔍 Select a Bar",
                className="mb-4"
            ))
        ]),

        dcc.Tabs([
            dcc.Tab(label="📊 Topic Distribution", children=[
                html.Br(),
                dcc.Graph(id='topic-counts')
            ]),
            dcc.Tab(label="⭐ Average Ratings", children=[
                html.Br(),
                dcc.Graph(id='avg-rating-topic')
            ]),
            dcc.Tab(label="🎯 Rating vs Topic", children=[
                html.Br(),
                dcc.Graph(id='rating-vs-topic')
            ]),
            dcc.Tab(label="🗺 Map of Bars", children=[
                html.Br(),
                dcc.Graph(id='bar-map')
            ]),
            dcc.Tab(label="🤝 Similar Bars", children=[
                html.Br(),
                html.Div(id='recommender-output', className="alert alert-info", style={'fontSize': 16})
            ])
        ])
    ], style={'paddingLeft': '40px', 'paddingRight': '40px', 'paddingBottom': '60px'})
])

@app.callback(
    Output('topic-counts', 'figure'),
    Output('avg-rating-topic', 'figure'),
    Output('rating-vs-topic', 'figure'),
    Output('bar-map', 'figure'),
    Output('recommender-output', 'children'),
    Input('place-dropdown', 'value')
)
def update_graphs(selected_place):
    filtered = df if selected_place is None else df[df['place_name'] == selected_place]
    if filtered.empty:
        fig = px.bar(title="No Data Available")
        return fig, fig, fig, fig, "No recommendations available."

    topic_counts = filtered['dominant_topic'].value_counts().sort_index().reset_index()
    topic_counts.columns = ['dominant_topic', 'count']

    fig1 = px.bar(
        topic_counts,
        x='dominant_topic', y='count',
        title='Review Count by Topic'
    )
    fig1.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=40, b=30))

    avg_rating = filtered.groupby('dominant_topic')['rating_x'].mean().reset_index()
    fig2 = px.bar(
        avg_rating,
        x='dominant_topic', y='rating_x',
        title='Average Rating by Topic'
    )
    fig2.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=40, b=30))

    fig3 = px.scatter(
        filtered, x='dominant_topic', y='rating_x', color='place_name',
        hover_data=['review_text', 'topic_keywords'],
        title='Rating vs Topic'
    )
    fig3.update_layout(template="plotly_white", height=350, margin=dict(l=10, r=10, t=40, b=30))

    fig4 = px.scatter_mapbox(
        filtered.drop_duplicates(subset='place_name'),
        lat='latitude', lon='longitude', hover_name='place_name',
        hover_data=['rating_x'], color='place_name', zoom=13, height=450
    )
    fig4.update_layout(mapbox_style="open-street-map", margin=dict(t=30, b=10, l=10, r=10))

    if selected_place:
        recs = recommend_places(selected_place)
        rec_text = "### Recommended bars similar to **{}**:\n- ".format(selected_place) + "\\n- ".join(recs)
    else:
        rec_text = "Select a bar from the dropdown to see similar places."

    return fig1, fig2, fig3, fig4, dcc.Markdown(rec_text)

if __name__ == '__main__':
    print("Launching dashboard on http://127.0.0.1:8050 ...")
    app.run(debug=True)
