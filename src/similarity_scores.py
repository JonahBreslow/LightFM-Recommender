import numpy as np
from joblib import delayed, Parallel
import os
import pandas as pd

NUM_CPUS = os.cpu_count()


def euclidean_score(df, user1, user2):
    assert user1 in df.userId
    assert user2 in df.userId

    ratings_1 = df[df.userId == user1]
    ratings_2 = df[df.userId == user2]

    ratings_both = ratings_1.merge(ratings_2, 'inner', 'movieId')
    if ratings_both.shape[0] == 0:
        return 0

    squared_diff = np.square(ratings_both.rating_x - ratings_both.rating_y).to_list()
    return [user1, user2, 1 / (1 + np.sqrt(np.sum(squared_diff)))]


def cosine_similarity(df, user1, user2):
    assert user1 in df.userId
    assert user2 in df.userId

    df_wide = df.pivot(index='userId', columns='movieId', values='rating')
    u1_vector = np.array(df_wide[df_wide.index == user1].fillna(0)).flatten()
    u2_vector = np.array(df_wide[df_wide.index == user2].fillna(0)).flatten()

    cosine = (np.dot(u1_vector, u2_vector) /
              (np.linalg.norm(u1_vector) * np.linalg.norm(u2_vector))
              )
    return [user1, user2, cosine]


def find_similar_users(df, user, num_users=5, num_cpus=NUM_CPUS):

    """
    :param df: the ratings dataframe with columns `userId`, `movieId`, and `rating`
    :param user: the use for whom you want to find similar users
    :param num_users: the number of most similar users. -1 means all users.
    :param num_cpus: the number of CPU's on your machine for parallelism
    :return: an array of users and similarity scores
    """
    assert user in df.userId

    # Compute cosine similarity between input user
    # and all the users in the dataset
    scores = np.array(
        Parallel(n_jobs=num_cpus)(delayed(cosine_similarity)(df, user, x) for x in df.userId.unique() if x != user)
    )

    # Sort the scores in decreasing order
    scores_sorted = np.argsort(scores[:, 2])[::-1]

    if num_users == -1:
        return_array = scores[scores_sorted]
    else:
        return_array = scores[scores_sorted][0:num_users]

    similarity_df = pd.DataFrame(return_array, columns=['input_user', 'userId', 'similarity'])
    similarity_df['scaled_similarity'] = similarity_df.similarity / np.sum(similarity_df.similarity)
    return similarity_df


def get_recommendations(df, input_user, num_recs=5, num_users=-1):
    assert input_user in df.userId

    similarity_scores = find_similar_users(df, input_user, num_users)
    movie_recs = df.merge(similarity_scores, 'inner', 'userId')
    movie_recs['wt_rating'] = movie_recs.scaled_similarity * movie_recs.rating

    movie_recs = (movie_recs
                  .groupby('movieId')[['wt_rating']]
                  .sum()
                  .sort_values('wt_rating', ascending=False)
                  .reset_index()
                  .head(num_recs)
                  )

    return movie_recs

