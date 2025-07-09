from flask import Flask, render_template, request
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained data
df = pickle.load(open('df.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

# Recommendation function (same as in training)
def recommend(song_name):
    try:
        idx = df[df['song'].str.lower() == song_name.lower()].index[0]
        distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])

        recommended_songs = []
        for i in distances[1:6]:  # Skip the first one (it’s the same song)
            recommended_songs.append(df.iloc[i[0]].song)
        return recommended_songs

    except IndexError:
        return None  # Song not found

# Home route
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Recommendation route
@app.route('/recommend', methods=['POST'])
def recommend_songs():
    song = request.form['song']
    recommendations = recommend(song)

    if recommendations:
        return render_template('index.html', recommendations=recommendations)
    else:
        error_message = "❌ Song not found in our dataset. Please try another."
        return render_template('index.html', error=error_message)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
