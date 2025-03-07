import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Dropout, ReLU, Concatenate
from tensorflow.keras.optimizers import Adam
import kaiwu as kw
import seaborn as sns
import matplotlib.pyplot as plt

data_path = "C:/Users/zcy28/Documents/data_set/ml-100k"  # Dataset path

def load_movielens_data():
    # Load the ratings data from MovieLens
    ratings_df = pd.read_csv(f'{data_path}/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
    # Load movie information
    movies_df = pd.read_csv(f'{data_path}/u.item', sep='|', encoding='latin-1', header=None, usecols=[0, 1],
                            names=['movie_id', 'movie_title'])
    return ratings_df, movies_df


ratings_df, movies_df = load_movielens_data()

users = ratings_df['user_id'].unique()
movies = ratings_df['movie_id'].unique()
num_users = len(users)
num_movies = len(movies)

user_mapping = {user: idx for idx, user in enumerate(ratings_df['user_id'].unique())}
movie_mapping = {movie: idx for idx, movie in enumerate(ratings_df['movie_id'].unique())}

ratings_df['user_id'] = ratings_df['user_id'].map(user_mapping)
ratings_df['movie_id'] = ratings_df['movie_id'].map(movie_mapping)

user_movie_matrix = np.zeros((len(user_mapping), len(movie_mapping)))
for row in ratings_df.itertuples():
    user_movie_matrix[row.user_id, row.movie_id] = row.rating

embedding_dim = 10

user_input = Input(shape=(1,))
movie_input = Input(shape=(1,))

user_embedding = Embedding(input_dim=num_users, output_dim=embedding_dim)(user_input)
movie_embedding = Embedding(input_dim=num_movies, output_dim=embedding_dim)(movie_input)

user_vec = Flatten()(user_embedding)
movie_vec = Flatten()(movie_embedding)

concat = Concatenate()([user_vec, movie_vec])

hidden_layer1 = Dense(128)(concat)
hidden_layer1 = ReLU()(hidden_layer1)
hidden_layer1 = Dropout(0.5)(hidden_layer1)

hidden_layer2 = Dense(64)(hidden_layer1)
hidden_layer2 = ReLU()(hidden_layer2)
hidden_layer2 = Dropout(0.5)(hidden_layer2)

output_layer = Dense(1)(hidden_layer2)

model = Model(inputs=[user_input, movie_input], outputs=output_layer)

learning_rate = 0.0005
model.compile(optimizer=Adam(lr=learning_rate), loss='mean_squared_error')

user_movie_matrix[user_movie_matrix > 0] = 1
user_movie_matrix[user_movie_matrix <= 0] = -1

sub_matrix = user_movie_matrix[:100, :100]

worker = kw.classical.SimulatedAnnealingOptimizer(initial_temperature=100,
                                                  alpha=0.99,
                                                  cutoff_temperature=0.001,
                                                  iterations_per_t=100,
                                                  size_limit=10)

solution = worker.solve(sub_matrix)

print("Optimized solution: ", solution)

def simulated_annealing_optimizer(model, ratings_df, epochs=300, batch_size=64, initial_temperature=100):
    temperature = initial_temperature
    best_loss = float('inf')
    best_weights = model.get_weights()
    temperatures = []
    losses = []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        # Train the model
        history = model.fit([ratings_df['user_id'], ratings_df['movie_id']], ratings_df['rating'],
                            batch_size=batch_size, epochs=1, verbose=1)

        loss = history.history['loss'][0]
        if loss < best_loss or np.random.rand() < np.exp((best_loss - loss) / temperature):
            best_loss = loss
            best_weights = model.get_weights()

        temperature *= 0.99
        temperatures.append(temperature)
        losses.append(loss)

        model.set_weights(best_weights)

    return model, temperatures, losses


model, temperatures, losses = simulated_annealing_optimizer(model, ratings_df, epochs=500)

movie_ids = [i + 1 for i in range(num_movies)]
for user_idx in range(solution.shape[0]):
    print(f"Recommended movies for user {user_idx + 1}:")
    recommended_movies = [movie_ids[i] for i in range(solution.shape[1]) if solution[user_idx][i] == 1]

    if recommended_movies:
        print(f"  Recommended movies: {recommended_movies}")
    else:
        print(f"  No recommended movies")
    print("-" * 40)

def plot_ising_matrix_comparison(before_matrix, after_matrix):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    sns.heatmap(before_matrix, cmap='coolwarm', cbar_kws={'label': 'Rating'}, annot=False)
    plt.title('Ising Matrix Before Optimization')

    plt.subplot(1, 2, 2)
    sns.heatmap(after_matrix, cmap='coolwarm', cbar_kws={'label': 'Rating'}, annot=False)
    plt.title('Ising Matrix After Optimization')

    plt.tight_layout()
    plt.show()

def plot_simulated_annealing_progress(temperatures, losses):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(temperatures, color='blue')
    plt.title('Temperature Decay Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Temperature')

    plt.subplot(1, 2, 2)
    plt.plot(losses, color='red')
    plt.title('Loss Function Change Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.show()

plot_ising_matrix_comparison(user_movie_matrix[:100, :100], solution)
plot_simulated_annealing_progress(temperatures, losses)
