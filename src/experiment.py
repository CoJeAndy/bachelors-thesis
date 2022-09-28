from itertools import combinations
from typing import List
import pickle
import csv
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd


def dbscan_experiment(features: np.array) -> tuple[List[float], List[int]]:
    """Run dbscan with different epsilon parameter, for our data values 0.1, 0.15, 0.2 yielded best results

    Args:
        features (np.array): chosen features for experiment run

    Returns:
        tuple[List[float], List[int]]: silhouette scores, number of clusters for each epsilon parameter
    """
    silhouette_avg = []
    cluster_count = []
    for epsilon in [0.1, 0.15, 0.2]:
        dbscan = DBSCAN(eps=epsilon, min_samples=20).fit(features)
        silhouette_avg.append(silhouette_score(features, dbscan.labels_))
        cluster_count.append(len(set(dbscan.labels_)))
    return silhouette_avg, cluster_count


def elbow_method(features: np.array) -> List[float]:
    """Compute sum of squared distances for 2-10 values of K for K-means 

    Args:
        features (np.array): chosen features for experiment run

    Returns:
        List[float]: sum of squared distances for each K parameter
    """
    sum_of_squared_distances = []
    cluster_range = range(2,11)
    for num_clusters in cluster_range :
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(features)
        sum_of_squared_distances.append(kmeans.inertia_)

    return sum_of_squared_distances


def silhouette_analysis(data: np.array) -> List[float]:
    """Compute silhouette score for 2-10 values of K for K-means 

    Args:
        data (np.array): chosen features for experiment run

    Returns:
        List[float]: silhouette scores for each K parameter
    """
    silhouette_avg = []
    cluster_range = range(2,11)

    for num_clusters in cluster_range:
        # initialise kmeans
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(data)
        cluster_labels = kmeans.labels_

        # silhouette score
        silhouette_avg.append(silhouette_score(data, cluster_labels))

    return silhouette_avg


def run_multiple_experiments(user_data: pd.DataFrame, base_features: List[str], other_features: List[str]) -> List:
    """Run experiment for every combination from chosen features 

    Args:
        user_data (pd.DataFrame): logs of exercises
        base_features (List[str]): primary features in used in every experiment
        other_features (List[str]): secondary features that change in every experiment

    Returns:
        List: results from K-means and DBSCAN for each combination
    """
    results = []
    i = 1
    for combination in combinations(other_features, 3):
        print(i, list(combination))
        results.append((list(combination), run_experiment(user_data, base_features + list(combination))))
        i += 1
    ## final combination from all behaviour pattern features
    behaviour_combination = ['average_exercise_order', 'average_exercise_distance', 'repeated_exercise_count', 'Základy: FJDK']
    print(i+1, behaviour_combination)
    results.append(behaviour_combination, run_experiment(user_data, behaviour_combination))
    return results


def run_experiment(user_data: pd.DataFrame, features: List[str]) -> tuple:
    """Run DBSCAN and K-means for chosen combination of features

    Args:
        user_data (pd.DataFrame): logs of exercises
        features (List[str]): chosen combination of features

    Returns:
        tuple: Results of DBSCAN and K-means for chosen combination of features
    """
    print(user_data.shape)
    chosen_features = user_data[features]
    elbow_result = elbow_method(chosen_features.to_numpy())
    silhouette_result = silhouette_analysis(chosen_features.to_numpy())
    db_silhouette, db_clusters = dbscan_experiment(chosen_features)
    print(elbow_result)
    print(silhouette_result)
    print(db_silhouette, db_clusters)
    return elbow_result, silhouette_result, db_silhouette, db_clusters



## chosen base features that do not change
base_features = ['number_logs', 'success_rate']
## other feature from which all combinations of 3 are tried
other_features = ['average_exercise_order', 'number_dif_exercise',
       'average_WPM', 'average_exercise_distance',
       'average_moves', 'number_dif_days', 'repeated_exercise_count', 'Základy: FJDK']

with open("raw_user_10.pickle", "rb") as fh:
    data = pickle.load(fh)

results = run_multiple_experiments(data, base_features, other_features)
## run_experiment(data, ['number_logs', 'success_rate', 'average_time', 'average_moves', 'repeated_exercise_count'])
run_experiment(data, ['average_exercise_order', 'average_exercise_distance', 'repeated_exercise_count', 'Základy: FJDK'])

results.sort(key=lambda x: max(x[1][1]), reverse=True)

## save results to CSV file
with open('experiment_K_means_10_new.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow([i for i in range(0, 18)])
    for i in range(56):
        best_K_anot = [" "]*9
        best_DB_anot = [" "]*3
        best_K_anot[results[i][1][1].index(max(results[i][1][1]))] = 'X'
        best_DB_anot[results[i][1][2].index(max(results[i][1][2]))] = 'X'
        writer.writerow(['|'.join(results[i][0]), 'elbow method:'] + results[i][1][0] + ['', '|'.join(results[i][0]),
                        'number of clusters:'] + results[i][1][3])
        writer.writerow(['','silhouette score:']+ results[i][1][1] + ['', '', 'silhouette score:'] + results[i][1][2])
        writer.writerow(['','best number of K:'] + best_K_anot + ['', '', 'best epsilon:'] + best_DB_anot)