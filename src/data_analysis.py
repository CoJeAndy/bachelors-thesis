import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List


def basic_analysis_graphs(logs: pd.DataFrame, title: str):
    """Create 3 different scatterplot graphs for data analisis

    Args:
        logs (pd.DataFrame): logs of exercises
        title (str): title for graphs
    """    
    # transform time from milisecond to hours
    logs['responseTime_hours'] = logs['responseTime'] / (1000 * 60 * 60) 

    sns.scatterplot(data=logs, x='moves', y='log_count')
    plt.title(title)
    plt.xlabel('počet prepísaných znakov')
    plt.ylabel('počet riadkov')
    plt.show()

    sns.scatterplot(data=logs, x='moves', y='responseTime_hours', alpha=0.3)
    plt.title(title)
    plt.xlabel('počet prepísaných znakov')
    plt.ylabel('čas riešenia v hodinách')
    plt.show()

    g = sns.scatterplot(data=logs[logs['moves'] < 500], x='moves', y='error_count', hue='correct', alpha=0.2)
    plt.title(title)
    plt.xlabel('počet prepísaných znakov')
    plt.ylabel('počet chýb')
    new_title = 'Úspešnosť'
    g.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    g.legend_.set_title(new_title)


    new_labels = ['neúspešný', 'úspešný']
    for t, l in zip(g.legend_.texts, new_labels):
        t.set_text(l)
    plt.show()


def get_typing_speed_wpm(responseTime: int, moves: int) -> int:
    """Compute words per minute (WPM) for exercise

    Args:
        responseTime (int): response time of exercise
        moves (int): count of written characters of exercise

    Returns:
        int: WPM for exercise
    """
    return moves / ((responseTime / 60000) * 5)


def get_number_errors(answer: str) -> int:
    """Get number of errors of exercise, errros are divided by "|"

    Args:
        answer (str): logged errors of exercise

    Returns:
        int: number of errors of excerice
    """    
    if type(answer) == str:
        return answer.count('|')
    return 0


def get_max_error_sequence(answer: str) -> int:
    """Get maximal number of errors for one character in text

    Args:
        answer (str): logged errors for exercise

    Returns:
        int: maximal number of errors for one character
    """    
    if type(answer) != str:
        return 0
    last_index = ''
    last_sequence = 0
    max_sequence = 0
    
    for error in answer.split('|'):
        if error[:-1] != last_index:
            last_index = error[:-1]
            max_sequence = max(max_sequence, last_sequence)
            last_sequence = 1
        else:
            last_sequence += 1
    
    return max(last_sequence, max_sequence)


## EXERCISE ANALISIS
def get_data_for_exercises(data: pd.DataFrame) -> pd.DataFrame:
    """Get average response time for each exercise and number of correct/all logs
       for each exercise

    Args:
        data (pd.DataFrame): logs of exercises

    Returns:
        pd.DataFrame: average response time, number of correct/all logs for each exercise
    """    
    df = data.groupby(['item']).agg({
        'responseTime': ['mean'],
        'correct': ['count', 'sum'],
    }).reset_index(level='item', col_level=1)
    df.columns = df.columns.droplevel(level=0)
    df = df.rename(columns={"count" : "all_count", "sum" : "correct_count"})
    df['incorrect_count'] = df["all_count"] - df["correct_count"]
    return df


## ERROR ANALISIS
def add_new_error_indexes(index_dict: Dict[int, int], answer: str) -> Dict[int, int]:
    """Get number of errors for specific index in specific exercise

    Args:
        index_dict (Dict[int, int]): dictionary of errors for specific index
        answer (str): logged errors for exercise

    Returns:
        Dict[int, int]: updated dictionary of errors for specific index
    """    
    start = 0
    end =  answer.find('|')
    while end != -1:
        try:
            index = int(answer[start:end-1])
            index_dict[index] = index_dict.get(index, 0) + 1
        except:
            pass
        start = end + 1
        end = answer.find('|', start)
    return index_dict


# get number of error for each excercise and number of errors for specific indexes in exercises
def get_error_indexes_sums_for_exercises(exercise_id_all_count: Dict[int, int], 
                                         data: pd.DataFrame) -> tuple[Dict, Dict]:
    """Get average number of erros for each exercise and number of erros 
       for specific index for each exercise

    Args:
        exercise_id_all_count (Dict[int, int]): number of logs for each exercise
        data (pd.DataFrame): logs of exercises

    Returns:
        Tuple(Dict, Dict): average number of erros for each exercise, 
                           number of erros for specific index for each exercise
    """    
    exercise_id_number_error = dict()
    exercise_id_index_error = dict()
    
    for item, answer in zip(data['item'], data['answer']):
        if type(answer) == str:
            exercise_id_number_error[item] = exercise_id_number_error.get(item, 0) + answer.count('|')
            exercise_id_index_error[item] = add_new_error_indexes(exercise_id_index_error.get(item, dict()), answer)
    
    for key, value in exercise_id_number_error.items():
        exercise_id_number_error[key] = value / exercise_id_all_count[key]
    
    return exercise_id_number_error, exercise_id_index_error


def add_new_error_for_character(char_errors: Dict[str, Dict[str, int]], exercise_texts: Dict[int, str], item_id: int, answer: str, only_first_errors: bool =False) -> Dict[str, Dict[str, int]]:
    """Update error dictionary for specific character

    Args:
        char_errors (Dict[str, Dict[str, int]]): current error dictionary {char: {total: all_errors, mistake_char: number_errors ...}}
        exercise_texts (Dict[int, str]): dictionary {exercise_id: exercise_text}
        item_id (int): exercise id
        answer (str): errors for exercise

    Returns:
        Dict[str, Dict[str, int]]: updated error dictionary
    """    
    start = 0
    end =  answer.find('|')
    error_index = -1
    while end != -1:
        try:
            if end + 2 < len(answer) and answer[end+1] == '|' and answer[end+2].isdigit():
                end += 1

            index = int(answer[start:end-1])
            if (only_first_errors and index != error_index) or (not only_first_errors):
                error_index = index
                error = answer[end-1]
                expected_char = exercise_texts[item_id][index]

                if (expected_char.isupper() and error.islower()) or expected_char.islower() and error.isupper():
                    char_errors["shift"]['total'] = char_errors[expected_char]['total'] + 1

                error = error.lower()
                expected_char = expected_char.lower()
                
                ## only error in shift but the correct key was pressed
                if error != expected_char:
                    if expected_char in char_errors:
                        char_errors[expected_char]['total'] = char_errors[expected_char]['total'] + 1
                        char_errors[expected_char][error] = char_errors[expected_char].get(error, 0) + 1
                    else:
                        char_errors[expected_char] = {'total': 1, error: 1}

        except:
            ## ignore badly logged errors 
            pass

        start = end + 1
        end = answer.find('|', start)
    return char_errors


def get_number_of_errors_for_specific_char(whole_text_logs: pd.DataFrame, user_ids: List, exercise_texts: Dict[int, str], only_first_errors: bool = False) -> Dict[str, Dict[str, int]]:
    """Get total number of errors for characters and specific mistakes for each key

    Args:
        whole_text_logs (pd.DataFrame): logs of whole text exercises
        user_ids (List): user ids
        exercise_texts (Dict[int, str]): dictionary {exercise_id: exercise_text}
        only_first_errors (bool, optional): get only first error for character. Defaults to False.


    Returns:
        Dict[str, Dict[str, int]]: error dictionary {char: {total: all_errors, mistake_char: number_errors ...}}
    """
    user_whole_logs = whole_text_logs[whole_text_logs["user"].isin(user_ids)]
    char_errors = dict()
    char_errors['shift'] = {'total': 0}
    for item_id, answer in zip(user_whole_logs['item'], user_whole_logs['answer']):
        if type(answer) == str:
            char_errors = add_new_error_for_character(char_errors, exercise_texts, item_id, answer, only_first_errors)
    return char_errors


basic_letters = 'abcdefghijklmnopqrstuvwxyz'
substitution_dict = {'q': 'wa', 'w': 'qes', 'e': 'wrd', 'r': 'etf', 't':'rzg', 'z': 'tuh', 'u': 'zij', 'i': 'uok', 'o': 'ipl',
                     'p': 'o', 'a': 'sqy', 's': 'adwx', 'd':'sfec', 'f':'dgrv', 'g': 'fhtb', 'h': 'gjzn',
                     'j': 'hkum', 'k': 'jli', 'l': 'ko', 'y':'xa', 'x': 'ycs', 'c':'xvd', 'v':'cbf', 'b':'vng',
                     'n': 'bmh', 'm':'nj'}


def get_substitution_errors(char_errors: Dict[str, Dict[str, int]]) -> tuple[int, int]:
    """compute substitution errors

    Args:
        char_errors (Dict[str, Dict[str, int]]): error dictionary {char: {total: all_errors, mistake_char: number_errors ...}}

    Returns:
        tuple[int, int]: total errors, substitution errors
    """
    total_substitution_errors = 0
    total_basic_letters_errors = 0
    for char in basic_letters:
        if not char_errors.get(char, {}):
            continue
        total_basic_letters_errors += sum([char_errors[char].get(error, 0) for error in basic_letters])
        for error in substitution_dict[char]:
            total_substitution_errors += char_errors[char].get(error, 0)
    return total_basic_letters_errors, total_substitution_errors


def create_error_heatmap(char_errors: Dict[str, Dict[str, int]]):
    """Create heatmap that represents expected character and mistakenly typed characters

    Args:
        char_errors (Dict[str, Dict[str, int]]): error dictionary {char: {total: all_errors, mistake_char: number_errors ...}}
    """    
    error_array = []
    substitution_errors = []
    for char in basic_letters:
        error_array.append([0]*len(basic_letters))
        substitution_errors.append([' ']*len(basic_letters))
        
        if not char_errors.get(char, {}):
            continue

        for i, error in enumerate(basic_letters):
            if error in substitution_dict[char]:
                substitution_errors[-1][i] = 'X'
            error_array[-1][i] = char_errors[char].get(error, 0)
    
    sns.set(font_scale=2.5)
    plt.figure(figsize=(15,15))
    plt.title('X - označené chyby typu zámeny', fontsize=18)
    sns.heatmap(error_array, xticklabels=basic_letters.upper(), yticklabels=basic_letters.upper(), annot=substitution_errors, fmt = '', annot_kws={"size": 18})
    plt.yticks(rotation=0, fontsize=18)
    plt.xticks(fontsize=18)
    plt.show()
    sns.set(font_scale=1)
