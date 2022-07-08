import math
from black import main
import numpy as np
import pandas as pd

import ray
from ray.data.preprocessors import MinMaxScaler, MultiHotEncoder, OrdinalEncoder
from ray.rllib.policy.sample_batch import SampleBatch

ray.init(ignore_reinit_error=True)


def get_user_ds():
    # read the dataset into a pandas df
    users = ray.data.read_text("ml-1m/users.dat")

    def convert_df(batch):
        cols = ["user_id", "gender", "age", "occupation", "zipcode"]
        df = pd.DataFrame(batch, columns=cols)

        return df

    users = users \
        .map(lambda r: r.split("::")) \
        .map_batches(convert_df)
    return users

def preprocess_users(users):
    # apply ordinal encoding to the following columns
    cols = ["gender", "age", "occupation", "zipcode"]
    # or assuming that the type of the columns is correctly specified in 
    # the schema you can filter based on their dtype. This does not work here 
    # becuase the dtypes are all objects but if you directly import your 
    # relational db the schema will be inherited from there. 
    # cols = [name for name, dtype in users.schema() if dtype == 'int64']
    oe = OrdinalEncoder(
        columns=cols, encode_lists=False
    )
    oe.fit(users)
    users = oe.transform(users)

    return users.to_random_access_dataset(key="user_id")


def get_perprocessed_movie_ds():
    # read the dataset into a pandas df
    movies = ray.data.read_text("ml-1m/movies.dat", encoding="latin-1")
    movies = movies \
        .map(lambda r: r.split("::")) \
        .map(lambda r: (r[0], r[1], r[2].split("|"))) \
        .map_batches(lambda b: pd.DataFrame(
            b, columns=["movie_id", "title", "genre"]))

    # apply ordinal encoding to the following columns
    oe = OrdinalEncoder(columns=["genre"], encode_lists=True)
    oe.fit(movies)
    movies = oe.transform(movies)

    # apply onehote encoding to the following columns
    mhe = MultiHotEncoder(columns=["genre"])
    mhe.fit(movies)
    movies = mhe.transform(movies)

    return movies.to_random_access_dataset(key="movie_id")


def get_perprocessed_ratings_ds():
    # read the dataset into a pandas df and repartition it into 5 blocks
    ratings = ray.data.read_text("ml-1m/ratings.dat") \
        .repartition(20)
    ratings = ratings \
        .map(lambda r: r.split("::")) \
        .map(lambda r: (r[0], r[1], int(r[2]), int(r[3]))) \
        .map_batches(lambda b: pd.DataFrame(
            b, columns=["user_id", "movie_id", "rating", "ts"]))

    # apply min-max scaling normalization to rating column
    mms = MinMaxScaler(columns=["rating"])
    mms.fit(ratings)
    ratings = mms.transform(ratings)

    return ratings


OBS_SHAPE = 22

def sample_batches_ds(users, movies, ratings):
    def sample_batch(rating):
        # user = users.filter({"user_id": rating["user_id"]).to_list()
        user = users.multiget(rating["user_id"].to_list())
        movie = movies.multiget(rating["movie_id"].to_list())

        df = {
            "user_id": [],
            "obs": [],
            "rewards": [],
            "t": [],
        }
        for u, m in zip(user, movie):
            obs = [u["gender"], u["age"], u["occupation"], u["zipcode"]] + list(m["genre"])
            assert len(obs) == OBS_SHAPE, f"{len(obs)} != {OBS_SHAPE}"
            df["obs"].append(np.array(obs))
        for _, r in rating.iterrows():
            df["user_id"].append(r["user_id"])
            df["rewards"].append(r["rating"])
            df["t"].append(r["ts"])
        
        return pd.DataFrame(df)

    sample_batches = ratings.map_batches(sample_batch)

    return sample_batches


def episode_aggregator(b):
    sorted_batch = b.sort_values(by=SampleBatch.T)

    obs = np.stack(sorted_batch[SampleBatch.OBS], 0)
    obs_shape = obs.shape[1:]
    new_obs = np.concatenate([obs[1:], np.zeros((1,) + obs_shape)], 0)
    rewards = np.stack(sorted_batch[SampleBatch.REWARDS], 0)
    t = np.stack(sorted_batch[SampleBatch.T], 0)

    # have not really figured out the action and done yet. 
    actions = [1] * len(t)
    dones = [False] * (len(t) - 1) + [True]

    return pd.DataFrame({
        "type": ["SampleBatch"],
        SampleBatch.OBS: [obs],
        SampleBatch.NEXT_OBS: [new_obs],
        SampleBatch.ACTIONS: [actions],
        SampleBatch.REWARDS: [rewards],
        SampleBatch.DONES: [dones],
        SampleBatch.T: [t],
    })


if __name__ == '__main__':
    users_ds = get_user_ds()
    users_ds = preprocess_users(users_ds)
    movies_ds = get_perprocessed_movie_ds()
    ratings_ds = get_perprocessed_ratings_ds()


    sample_batches = sample_batches_ds(users_ds, movies_ds, ratings_ds)
    episode_ds = sample_batches.groupby("user_id").map_groups(episode_aggregator)
    breakpoint()




