"""Prepare MovieLens 1M dataset: download, parse, map ids, and split.
Outputs:
- data/processed/train.csv
- data/processed/test.csv
- data/processed/user_map.json
- data/processed/item_map.json
"""
import os
import argparse
import json
import urllib.request
import zipfile
import pandas as pd

ML_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def download_and_extract(dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, "ml-1m.zip")
    try:
        if not os.path.exists(zip_path):
            print("Downloading MovieLens 1M...")
            urllib.request.urlretrieve(ML_1M_URL, zip_path)
        extract_dir = os.path.join(dest_dir, "ml-1m")
        if not os.path.exists(extract_dir):
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(dest_dir)
        return os.path.join(dest_dir, "ml-1m")
    except Exception as e:
        print(f"Warning: failed to download or extract MovieLens dataset: {e}")
        print("Falling back to generating a small synthetic dataset for demonstration.")
        # generate simple synthetic directory layout
        synth_dir = os.path.join(dest_dir, "ml-1m-synth")
        os.makedirs(synth_dir, exist_ok=True)
        return synth_dir


def load_ratings(ml_dir):
    ratings_path = os.path.join(ml_dir, "ml-1m", "ratings.dat")
    if not os.path.exists(ratings_path):
        # older zip structure
        ratings_path = os.path.join(ml_dir, "ratings.dat")
    if not os.path.exists(ratings_path):
        # generate synthetic dataset
        print("ratings.dat not found; generating synthetic dataset")
        import numpy as np
        num_users = 200
        num_items = 1000
        num_interactions = 10000
        rng = np.random.RandomState(42)
        user_ids = rng.randint(1, num_users+1, size=num_interactions)
        item_ids = rng.randint(1, num_items+1, size=num_interactions)
        ratings = rng.randint(1,6, size=num_interactions)
        timestamps = rng.randint(1000000000, 1300000000, size=num_interactions)
        df = pd.DataFrame({"user_id":user_ids, "item_id":item_ids, "rating":ratings, "timestamp":timestamps})
        return df
    # user::movie::rating::timestamp
    df = pd.read_csv(ratings_path, sep="::", engine="python", names=["user_id","item_id","rating","timestamp"] )
    return df


def load_item_titles(ml_dir):
    movies_path = os.path.join(ml_dir, "ml-1m", "movies.dat")
    if not os.path.exists(movies_path):
        movies_path = os.path.join(ml_dir, "movies.dat")
    titles = {}
    if os.path.exists(movies_path):
        # movieid::title::genres
        with open(movies_path, encoding='latin-1') as f:
            for line in f:
                parts = line.strip().split('::')
                if len(parts) >= 2:
                    mid = int(parts[0])
                    titles[mid] = parts[1]
    return titles


def build_mappings(df):
    unique_users = sorted(df['user_id'].unique())
    unique_items = sorted(df['item_id'].unique())
    user_map = {int(u): int(i) for i,u in enumerate(unique_users)}
    item_map = {int(it): int(i) for i,it in enumerate(unique_items)}
    return user_map, item_map


def remap_df(df, user_map, item_map):
    df = df.copy()
    df['user_idx'] = df['user_id'].map(user_map)
    df['item_idx'] = df['item_id'].map(item_map)
    return df


def leave_one_out_split(df):
    # for each user, keep the latest interaction as test
    df_sorted = df.sort_values(['user_idx','timestamp'])
    test_idx = df_sorted.groupby('user_idx').tail(1).index
    test = df_sorted.loc[test_idx]
    train = df_sorted.drop(test_idx)
    return train, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-dir', default='./data/raw', help='raw data directory')
    parser.add_argument('--out-dir', default='./data/processed', help='processed output directory')
    args = parser.parse_args()

    raw_dir = args.raw_dir
    out_dir = args.out_dir
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    extracted = download_and_extract(raw_dir)
    # ratings
    df = load_ratings(extracted)
    df['user_id'] = df['user_id'].astype(int)
    df['item_id'] = df['item_id'].astype(int)

    user_map, item_map = build_mappings(df)

    df_m = remap_df(df, user_map, item_map)

    train, test = leave_one_out_split(df_m)

    # save
    train[['user_idx','item_idx','rating','timestamp']].to_csv(os.path.join(out_dir,'train.csv'), index=False)
    test[['user_idx','item_idx','rating','timestamp']].to_csv(os.path.join(out_dir,'test.csv'), index=False)

    with open(os.path.join(out_dir,'user_map.json'),'w') as f:
        json.dump(user_map, f)
    with open(os.path.join(out_dir,'item_map.json'),'w') as f:
        json.dump(item_map, f)
    # save item titles by original id
    titles = load_item_titles(raw_dir)
    with open(os.path.join(out_dir,'item_titles.json'),'w', encoding='utf-8') as f:
        json.dump(titles, f)

    print('Saved processed files to', out_dir)


if __name__ == '__main__':
    main()
