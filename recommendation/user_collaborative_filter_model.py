import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import cosine_similarity


class UserCollaborativeFilterModel(BaseEstimator):
    def __init__(self, n_neighbors=5, similarity="cosine"):
        self.n_neighbors = n_neighbors
        self.similarity = similarity

    def fit(self, X, y=None):
        """
        Creates a similarity matrix for the user-item matrix

        X: Matrix of user-item ratings. Rows are users, columns are items
            ratings.
        """
        self.user_item_matrix_ = np.array(X)
        if self.similarity == "cosine":
            self.similarity_matrix_ = cosine_similarity(self.user_item_matrix_)
            np.fill_diagonal(self.similarity_matrix_, 0)
        else:
            raise ValueError(
                f"Unsupported similarity metric: {self.similarity}")

        return self

    def predict(self, user_index, item_index):
        """
        Predicts the rating for a user-item pair

        user_index: Index of the user
        item_index: Index of the item
        """

        # Get the similarity scores for the user-item pair
        user_similarity_scores = self.similarity_matrix_[user_index, :]
        item_ratings = self.user_item_matrix_[:, item_index]

        # Exclude users who haven't rated the item
        mask = item_ratings > 0
        user_similarity_scores = user_similarity_scores[mask]
        item_ratings = item_ratings[mask]

        if len(item_ratings) == 0:
            return np.nan  # Cannot predict

        # Get the top n_neighbors
        top_n_neighbors_idx = np.argsort(
            user_similarity_scores)[-self.n_neighbors:]
        top_k_sims = user_similarity_scores[top_n_neighbors_idx]
        top_k_ratings = item_ratings[top_n_neighbors_idx]

        # Weighted average of the top n neighbors
        numerator = np.sum(top_k_sims * top_k_ratings)
        denominator = np.sum(top_k_sims)

        if denominator == 0:
            return np.mean(item_ratings)

        return numerator / denominator

    def recommend(self, user_index, n_recommendations=5):
        """
        Recommends items for a user

        user_index: Index of the user
        n_recommendations: Number of recommendations to return
        """
        user_ratings = self.user_item_matrix_[user_index, :]
        unrated_items = np.where(user_ratings == 0)[0]

        recommendations = []
        for item_index in unrated_items:
            rating = self.predict(user_index, item_index)
            recommendations.append((item_index, rating))

        # Sort by rating and return the top n_recommendations
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]
