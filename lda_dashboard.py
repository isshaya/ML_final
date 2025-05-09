
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
app = Dash(__name__)

app.layout = html.Div([
    html.H2("Manhattan Bars Review Dashboard", style={'textAlign': 'center'}),

    dcc.Dropdown(
        id='place-dropdown',
        options=[{'label': name, 'value': name} for name in sorted(df['place_name'].dropna().unique())],
        placeholder="Select a Bar"
    ),

    dcc.Graph(id='topic-counts'),
    dcc.Graph(id='avg-rating-topic'),
    dcc.Graph(id='rating-vs-topic'),

    html.Div(id='recommender-output', style={'marginTop': 30, 'fontSize': 16})
])

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

    fig2 = px.bar(
        filtered.groupby('dominant_topic')['rating_x'].mean().reset_index(),
        x='dominant_topic', y='rating_x',
        title='Average Rating by Topic'
    )

    fig3 = px.scatter(
        filtered, x='dominant_topic', y='rating_x', color='place_name',
        hover_data=['review_text', 'topic_keywords'],
        title='Rating vs Topic'
    )

    if selected_place:
        recs = recommend_places(selected_place)
        rec_text = "Top recommended similar bars to **{}**:\n- ".format(selected_place) + "\n- ".join(recs)

    else:
        rec_text = "Select a bar to see recommendations."

    return fig1, fig2, fig3, dcc.Markdown(rec_text)

if __name__ == '__main__':
    app.run(debug=True)
