from typing import Dict, List
import pandas as pd


def create_groups_of_exercises(ps_problem: pd.DataFrame, ps_ids: List[int]) -> Dict[int, List[int]]:
    """Assign exercises to problem sets

    Args:
        ps_problem (pd.DataFrame): ids of exercises, ids of problem sets, other unnecessary information
        ps_ids (List[int]): ids of problem sets

    Returns:
        Dict[int, List[int]]: grouped exercies for problem sets
    """    
    valid_ps_problem = ps_problem[ps_problem['ps'].isin(ps_ids)]
    exercise_groups = dict()
    for ps_id, item_id in zip(valid_ps_problem['ps'], valid_ps_problem['problem']):
        exercise_groups[ps_id] = exercise_groups.get(ps_id, []) + [item_id]
    return exercise_groups


def add_ordering_to_exercise_group_dict(exercise_groups_dict: Dict[int, List[int]], psani_deseti_ps: pd.DataFrame) -> Dict:
    """Add ordering and name to problem sets

    Args:
        exercise_groups_dict (Dict[int, List[int]]): grouped problem sets
        psani_deseti_ps (pd.DataFrame): information about problem sets

    Returns:
        Dict: exercise ids, ordering, name for problem sets
    """
    for ps_id, ordering, topic in zip(psani_deseti_ps['id'], psani_deseti_ps['ordering'], psani_deseti_ps['topic']):
        exercise_groups_dict[ps_id] = {'group': exercise_groups_dict[ps_id], 'ordering': ordering, 'name': topic}
    return exercise_groups_dict


def get_absolute_ordering(data_item: pd.DataFrame, exercise_groups_dict: Dict) -> Dict:
    """Order all exercises based on webpage ordering

    Args:
        data_item (pd.DataFrame): exercise information
        exercise_groups_dict (Dict): exercise ids, ordering, name for problem sets

    Returns:
        Dict: webpage order of all exercises
    """
    absolute_order_dict = dict()
    for item_id, item_ordering in zip(data_item['id'], data_item['ordering']):
        for key in exercise_groups_dict.keys():
            if item_id in exercise_groups_dict[key]['group']:
                absolute_order_dict[item_id] = (exercise_groups_dict[key]['ordering'], item_ordering)
                break
    absolute_order_dict = dict(sorted(absolute_order_dict.items(), key= lambda x: absolute_order_dict[x[0]]))
    for index, key in enumerate(absolute_order_dict.keys()):
        absolute_order_dict[key] = index + 1
    return absolute_order_dict


def get_exercise_moves_thresholds(threshold150: List[int], threshold200: List[int], threshold300: List[int], 
                                  exercise_id: int, exercise_moves: int, size: str) -> int:
    """Get different marker size based on number of written characters for specific exercise

    Args:
        threshold150 (List[int]): exercise ids with min limit 150 chars
        threshold200 (List[int]): exercise ids with min limit 200 chars
        threshold300 (List[int]): exercise ids with min limit 300 chars
        exercise_id (int): specific exericse id
        exercise_moves (int): number of written characters 
        size (str): chosen marker size

    Returns:
        int: marker size
    """
    marker_sizes = [2, 4, 8, 16] 
    if size == "medium":
        marker_sizes = [3, 6, 12, 24]
    if size == "large":
        marker_sizes = [6, 12, 24, 48]
    min_moves =  100
    if exercise_id in threshold150:
        min_moves = 150
    elif exercise_id in threshold200:
        min_moves = 200
    elif exercise_id in threshold300:
        min_moves = 300


    if exercise_moves < min_moves:
        return marker_sizes[0]
    if min_moves <= exercise_moves < min_moves + 125:
        return marker_sizes[1]
    if min_moves + 125 <= exercise_moves < 1000:
        return marker_sizes[2]
    return marker_sizes[3]


def get_average_solving_times(data: pd.DataFrame) -> Dict:
    """Get average response time for each exercise

    Args:
        data (pd.Dataframe): logs of exercises

    Returns:
        Dict: average response times for eac exercise
    """
    exercise_average_times = data.groupby(['item']).agg(average_time=('responseTime', 'mean'))
    exercise_average_times = exercise_average_times.reset_index()
    return dict(zip(exercise_average_times['item'], exercise_average_times['average_time']))
