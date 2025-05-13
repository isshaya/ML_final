import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import dash_bootstrap_components as dbc

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

# Initialize Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    html.H1("🍸 Manhattan Bars Review Dashboard", className="text-center my-4"),

    dbc.Row([
        dbc.Col(dcc.Dropdown(
            id='place-dropdown',
            options=[{'label': name, 'value': name} for name in sorted(df['place_name'].dropna().unique())],
            placeholder="🔍 Select a Bar",
            className="mb-4"
        ))
    ]),

    html.Hr(),

    html.Div([
        html.H3("📊 Review Distribution by Topic", className="mt-4"),
        dcc.Graph(id='topic-counts')
    ], className="mb-5"),

    html.Div([
        html.H3("⭐ Average Rating by Topic", className="mt-4"),
        dcc.Graph(id='avg-rating-topic')
    ], className="mb-5"),

    html.Div([
        html.H3("🎯 Rating vs Topic Index", className="mt-4"),
        dcc.Graph(id='rating-vs-topic')
    ], className="mb-5"),

    html.Div([
        html.H3("🤝 Recommended Similar Bars", className="mt-4"),
        html.Div(id='recommender-output', className="alert alert-info", style={'fontSize': 16})
    ], className="mb-5")

], fluid=True)

@app.callback(
    Output('topic-counts', 'figure'),
    Output('avg-rating-topic', 'figure'),
    Output('rating-vs-topic', 'figure'),
    Output('recommender-output', 'children'),
    Input('place-dropdown', 'value')
)
def update_graphs(selected_place):
    filtered = df if selected_place is None else df[df['place_name'] == selected_place]
    if filtered.empty:
        fig = px.bar(title="No Data Available")
        return fig, fig, fig, "No recommendations available."

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

    if selected_place:
        recs = recommend_places(selected_place)
        rec_text = "### Recommended bars similar to **{}**:\n- ".format(selected_place) + "\\n- ".join(recs)
    else:
        rec_text = "Select a bar from the dropdown to see similar places."

    return fig1, fig2, fig3, dcc.Markdown(rec_text)

if __name__ == '__main__':
    print("Launching dashboard on http://127.0.0.1:8050 ...")
    app.run(debug=True)
