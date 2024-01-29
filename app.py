from flask import Flask, render_template, request
import pandas as pd
import plotly.express as px
import imdb
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# Function to calculate cosine similarity
def calculate_cosine_similarity(vector_A, vector_B):
    vector_A = vector_A.reshape(1, -1)
    vector_B = vector_B.reshape(1, -1)
    similarity_score = cosine_similarity(vector_A, vector_B)[0, 0]
    return similarity_score

# Function to get movie data with filters
def get_filtered_movie_data(emotion, min_year=None, max_year=None, min_rating=None):
    ia = imdb.IMDb()

    # Search for movies based on emotion (genre)
    movies = ia.search_movie(emotion)

    movie_data = []
    for movie in movies:
        # Fetch the movie details
        ia.update(movie)

        movie_info = {
            'Title': movie.get('title', 'N/A'),
            'Year': movie.get('year', 'N/A'),
            'Rating': movie.get('rating', 'N/A'),
            'Genres': ', '.join(movie.get('genres', []))
        }

        # Apply filters
        if (min_year is None or movie_info['Year'] >= min_year) and \
           (max_year is None or movie_info['Year'] <= max_year) and \
           (min_rating is None or (movie_info['Rating'] is not None and movie_info['Rating'] >= min_rating)):
            movie_data.append(movie_info)

    return pd.DataFrame(movie_data)

# Function to recommend movies with similarity
def recommend_movies_with_similarity(emotion, user_vector, num_recommendations=5):
    ia = imdb.IMDb()

    # Search for movies based on emotion (genre)
    movies = ia.search_movie(emotion)

    recommendations = []
    for movie in movies:
        # Get additional details (e.g., rating) for each movie
        ia.update(movie, info=['main', 'ratings'])

        movie_vector = np.array([
            1 if emotion in movie['genres'] else 0,  # Check if the emotion matches the movie genre
            movie.get('rating', 0) / 10.0  # Normalize the rating to be between 0 and 1
        ])

        # Calculate cosine similarity with user's emotion vector
        similarity_score = calculate_cosine_similarity(user_vector, movie_vector)

        recommendation_info = {
            'title': movie['title'],
            'year': movie['year'],
            'rating': movie.get('rating', 'N/A'),
            'similarity_score': similarity_score
        }
        recommendations.append(recommendation_info)

    # Sort recommendations based on similarity score (descending order)
    recommendations = sorted(recommendations, key=lambda x: x['similarity_score'], reverse=True)[:num_recommendations]

    return recommendations

# Flask route for the index page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        emotion = request.form['emotion']
        min_year = int(request.form.get('min_year', 0) or 0)
        max_year = int(request.form.get('max_year', float('inf')) or float('inf'))
        min_rating = float(request.form.get('min_rating', 0) or 0)
        user_rating = float(request.form.get('user_rating', 0) or 0)

        # User's emotion vector
        user_vector = np.array([1, user_rating / 10.0])

        # Get filtered movie data
        df = get_filtered_movie_data(emotion, min_year, max_year, min_rating)

        # Scatter plot: Ratings vs. Release Years
        fig_scatter = px.scatter(df, x='Year', y='Rating', color='Genres',
                                size_max=10, title=f'Scatter Plot for {emotion}',
                                labels={'Rating': 'IMDb Rating', 'Year': 'Release Year'})
        fig_scatter.update_traces(marker=dict(size=8, opacity=0.6))

        # Bar chart: Number of Movies in Each Genre
        fig_bar = px.bar(df, x='Genres', title=f'Number of Movies in Each Genre for {emotion}',
                        labels={'Genres': 'Movie Genres', 'count': 'Number of Movies'})
        fig_bar.update_traces(marker=dict(color='skyblue', line=dict(color='black', width=2)))

        # Get movie recommendations with similarity
        recommendations = recommend_movies_with_similarity(emotion, user_vector)

        return render_template('index.html', scatter_plot=fig_scatter.to_html(full_html=False),
                               bar_chart=fig_bar.to_html(full_html=False),
                               recommendations=recommendations)

    return render_template('index.html', scatter_plot=None, bar_chart=None, recommendations=None)

if __name__ == '__main__':
    app.run(debug=True)
