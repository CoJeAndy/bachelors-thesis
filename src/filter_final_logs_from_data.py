from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import unquote
from typing import List
import pandas as pd


ANSWER, WRITTEN_LINES_COUNT, TIME, CORRECT, MOVES = 0, 1, 3, 4, 5


def create_new_csv_with_logs(data: pd.DataFrame, final_logs_ids: List[int], csv_name: str, log_counts=None):
    """Create new csv file from only final logs and optionaly add new colum for number of
       rewritten lines

    Args:
        data (pd.DataFrame): all logs
        final_logs_ids (List[int]): ids of final logs
        csv_name (str): name of new csv file
        log_counts (pd.Dataframe, optional): dataframe consisting of ids of final longs and count of
                                             rewritten lines + 1 before final log. Defaults to None.
    """
    filtered_data = data[data['id'].isin(final_logs_ids)]
    filtered_data = filtered_data.drop_duplicates()

    if log_counts:
        log_counts_dict = dict(zip([item[0] for item in log_counts], [item[1] for item in log_counts]))
        filtered_data['log_count'] = filtered_data['id'].map(log_counts_dict)

    filepath = Path(csv_name)  
    filtered_data.to_csv(filepath, sep=';', index=False)  


def update_log(old_logs: List, answer: str, log_id: int, time: str, correct: int, moves: int) -> List:
    """Update logs based on matching errors and moves

    Args:
        old_logs (List): previous logs
        answer (str): errors of thee last log
        log_id (int): id of the last log
        time (str): time of recording of the last log
        correct (int): information 1 - completed, 0 - unfinished exercise of the last log
        moves (int): number of written character of the last log

    Returns:
        List: return updated list of processed logs
    """
    working_logs = [item for item in old_logs if isinstance(item[0]) == str and item[0] != '#0,0,0']
    start_logs = [item for item in old_logs if (isinstance(item[0]) == str and item[0] == '#0,0,0') or isinstance(item[0]) != str]

    ## update working_log - at least one line was rewritten
    for i, working_log in enumerate(working_logs):
        if working_log[4]:
            continue
        old_answer = working_log[ANSWER]
        ## for logs that
        if old_answer == answer:
            working_logs[i] = (answer, working_log[WRITTEN_LINES_COUNT] + 1, log_id, time, correct, moves)
            return working_logs + start_logs
        ## new logs end with @@ and old logs end with #
        last_error_index = old_answer.rfind('@@')
        if last_error_index == -1:
            last_error_index = old_answer.rfind('#')
        
        ## check if substring of errors is the same
        substring = old_answer[:last_error_index]
        if substring and substring == answer[:len(substring)]:
            working_logs[i] = (answer, working_log[WRITTEN_LINES_COUNT] + 1, log_id, time, correct, moves)
            return working_logs + start_logs
        ## check if moves is at least 20 chacters bigger then previous if no mistakes were made
        if not substring and working_log[MOVES] + 20 < moves:
            working_logs[i] = (answer, working_log[WRITTEN_LINES_COUNT] + 1, log_id, time, correct, moves)
            return working_logs + start_logs

    if start_logs == []:
        return working_logs + [(answer, 0, log_id, time, correct, moves)]

    ## update last starting log - starting log is created when exercise is opened
    index = 0
    shortest_time_dif = timedelta(seconds=0)
    for i, start_log in enumerate(start_logs):
        if start_log[CORRECT]:
            continue
        temp_time = datetime.strptime(time, '%Y-%m-%d %H:%M:%S') - datetime.strptime(start_log[TIME], '%Y-%m-%d %H:%M:%S')
        if temp_time > shortest_time_dif:
            shortest_time_dif = temp_time
            index = i
    start_logs[index] = (answer, start_logs[index][WRITTEN_LINES_COUNT] + 1, log_id, time, correct, moves)

    return working_logs + start_logs


def filter_data(data: pd.DataFrame) -> pd.DataFrame:
    """Filter logs based on matching errors and moves

    Args:
        data (pd.DataFrame): all logs

    Returns:
        pd.DataFrame: final logs - after completing or exiting exercise
    """
    data = data.drop_duplicates()
    last_logs = dict()
    
    for log_id, user, item, answer, correct, moves, response_time, time in zip(data['id'], data['user'], data['item'], 
                                                                               data['answer'], data['correct'], data['moves'], 
                                                                               data['responseTime'], data['time']):
        if user in last_logs:
            if item in last_logs[user]:
                ## add new starting log for same user and same exercise
                if response_time == 0 and moves == 0:
                    last_logs[user][item].append((answer, 0, log_id, time, correct, moves))

                else:
                    last_logs[user][item] = update_log(last_logs[user][item], answer, log_id, time, correct, moves)
                
            else:
                last_logs[user][item] = [(answer, 0, log_id, time, correct, moves)]

        else:
            last_logs[user] = {item:[(answer, 0, log_id, time, correct, moves)]}
    
    final_log_ids_counts = []
    for items in last_logs.values():
        for logs in items.values():
            final_log_ids_counts += [(item[2], item[1]) for item in logs]
    return final_log_ids_counts


def decode_special_characters(string: str) -> str:
    """Decode special characters from %C3%A1 to á

    Args:
        string (str): string of badly encoded errors

    Returns:
        str: string of errors in utf-8
    """
    if isinstance(string) == str:
        return unquote(string)
    return string


## load data
data_item = pd.read_csv('csv/umimeprogramovatcz-psani_deseti_item.csv', sep=';')
data_log = pd.read_csv('csv/final_logs_new_1.csv', sep=';')

## decode charasters from %C3%5 to utf-8
data_log['answer'] = data_log['answer'].apply(decode_special_characters)

ordering = data_item['id'].unique()
ordering.sort()

# valid logs only to current exercises
valid_data_log = data_log[data_log['item'].isin(ordering)]

test_data = pd.read_csv('csv/test2.csv', sep=';')
final_logs = filter_data(valid_data_log)

create_new_csv_with_logs(data_log, [item[0] for item in final_logs], 'csv/final_logs_new_4.csv', final_logs)
