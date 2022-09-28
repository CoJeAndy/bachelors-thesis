from datetime import datetime, timedelta
from typing import Dict, List, Optional
from plotly.subplots import make_subplots
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
import matplotlib.cm as cm
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px


def exercise_success_rate(data: pd.DataFrame, absolute_order_dict: Dict):
    """Create barplot graph to see activity and success rate for specific exercises

    Args:
        data (pd.DataFrame): logs of exercises
        absolute_order_dict (Dict): webpage order of exercises
    """
    grouped = data.groupby(['item'])['correct'].value_counts().reset_index(name='counts')
    grouped['order'] = grouped['item'].map(absolute_order_dict)
    grouped = grouped.sort_values(by='order')
    grouped = grouped.pivot_table(index='order', columns='correct', values='counts').reset_index(level='order')
    grouped['all'] = grouped[0] + grouped[1]

    plt.figure(figsize=(30,10))
    plt.xticks(fontsize=7, rotation=90)

    sns.barplot(x='order', y='all', data=grouped, color='red')
    sns.barplot(x='order', y=1, data=grouped, color='green')
    plt.xlabel('ordered exercises')
    plt.ylabel('number of users')
    plt.show()



def create_histograms(hist_parameters: tuple, features: pd.DataFrame, transf_features: pd.DataFrame, rows_num: int):
    """Create histograms for each feature/ transformed feature 

    Args:
        hist_parameters (tuple): title, scale, ... for histogram
        features (pd.DataFrame): user features
        transf_features (pd.DataFrame): normalized user features
        rows_num (int): number of features
    """
    _, axis = plt.subplots(nrows=rows_num, ncols=2 ,figsize=(20,10*rows_num))
    for i, (index, name, xlabel_name, ylabel_name, scale) in enumerate(hist_parameters):
        values = features[index]
        axis[i, 0].set_title('all users ' + name)
        axis[i, 0].set_xlabel(f'{xlabel_name}\nmax: {max(features[index])}, \
                                 min: {min(features[index])}, \
                                 avg: {sum(features[index])/len(features)}')
        axis[i, 0].set_ylabel(ylabel_name)
        axis[i, 0].set_yscale(scale)
        axis[i, 0].hist(values, bins=20)

        transf_values = transf_features[index]
        axis[i, 1].set_title('users with > 10 exercises ' + name)
        axis[i, 1].set_xlabel(f'{xlabel_name}\nmax: {max(transf_features[index])}, \
                                 min: {min(transf_features[index])}, \
                                 avg: {sum(transf_features[index])/len(transf_features)}')
        axis[i, 1].set_ylabel(ylabel_name)
        axis[i, 1].set_yscale(scale)
        axis[i, 1].hist(transf_values, bins=20)
    plt.show()
    plt.close()


def get_day_lines_positions(data: pd.DataFrame) -> List:
    """Get positions of new days, different lines for more than 1 day pause between
       activity

    Args:
        data (pd.DataFrame): logs of exercises belonging to specific user

    Returns:
        List: line positions and types
    """    
    line_positions_types = []
    for i in range(1, len(data['time'])):
        if data.loc[i, 'time'] != data.loc[i-1, 'time']:
            date1 = datetime.strptime(data.loc[i-1, 'time'], '%Y-%m-%d')
            date2 = datetime.strptime(data.loc[i, 'time'], '%Y-%m-%d')
            line_positions_types.append((i-0.5, 'big' if date2 - date1 > timedelta(days=1) else 'small'))

    return line_positions_types


def create_day_lines(data: pd.DataFrame):
    """create vertical lines to devide new days

    Args:
        data (pd.DataFrame): logs of exercises belonging to specific user
    """    
    lines = get_day_lines_positions(data)
    for line in lines:
        plt.axvline(x=line[0], color='black', linestyle='--' if line[1] == 'small' else (0, (5, 10)))


## Noninteractive prototype with sns
def create_user_activity_graph(data: pd.DataFrame, user_id: int, absolute_order_dict: Dict):
    """Create visualization of exercises that user tried to solve

    Args:
        data (pd.DataFrame): logs of exercises
        user_id (int): specific user id
        absolute_order_dict (Dict): webpage order of exercises - dictionary {exercise_id: order_on_webpage}
    """

    user_data = data[data['user'] == user_id]
    user_data.reset_index(inplace=True, drop=True)
    user_data.reset_index(inplace=True)
    user_data['item_order'] = user_data['item'].map(absolute_order_dict)
    user_data['time'].replace(to_replace=r'([0-9]{4}-[0-9]{2}-[0-9]{2}).*', value=r'\1', regex=True, inplace=True)

    ## group by days
    x_axis_days = user_data.groupby('time').agg({'item': 'count', 'index': 'mean'}).reset_index()

    plt.figure(figsize = (40,15))
    ax = sns.lineplot(data=user_data, x='index', y='item_order', color='lightgrey')

    plt.scatter(x=user_data[(user_data['correct'] == 1)]['index'], y=user_data[(user_data['correct'] == 1)]['item_order'], color='green')
    plt.scatter(x=user_data[(user_data['correct'] == 0)]['index'], y=user_data[(user_data['correct'] == 0)]['item_order'], color='red')

    create_day_lines(user_data)

    ax.set_title('User Activity')
    ax.set_ylabel('exercise')
    ax.set_xlabel('activity logs')
    ax.set_xticks(x_axis_days['index'])
    ax.set_xticklabels(x_axis_days['time'])
    plt.xticks(fontsize=12, rotation=90)
    plt.show()
    plt.close()


def get_grouped_exercise_df(data: pd.DataFrame, user_id: int, exercise_to_group: Dict) -> pd.DataFrame:
    """Group exercises to problem sets and compute aggregate features

    Args:
        data (pd.DataFrame): logs of exercises
        user_id (int): specific user id
        exercise_to_group (Dict): dictionary {exercise_id: (group_name, group_id)}

    Returns:
        pd.DataFrame: grouped exercises to problem sets with aggregated features
    """
    user_data = data[data['user'] == user_id]
    user_data['time'].replace(to_replace=r'([0-9]{4}-[0-9]{2}-[0-9]{2}).*', value=r'\1', regex=True, inplace=True)
    user_data.reset_index(inplace=True, drop=True)

    exercise_groups_df = pd.DataFrame(columns=['order', 'group_order', 'all_count', 'correct_count', 'time'])

    ## inicialize first group parameters
    group = exercise_to_group[user_data.loc[0, 'item']][1]
    dif_exercise_ids = set()
    correct_count = 0
    all_count = 0
    index = 0
    last_day = user_data.loc[0, 'time']

    for row in user_data.itertuples():
        if group != exercise_to_group[row.item][1] or last_day != row.time:
            exercise_groups_df.loc[len(exercise_groups_df.index)] = \
                [index, group, all_count, correct_count, last_day]
            
            ## clear parameters
            group = exercise_to_group[row.item][1]
            dif_exercise_ids = set()
            dif_exercise_ids.add(row.item)
            correct_count = row.correct
            all_count = 1
            index += 1
            last_day = row.time
        else:
            dif_exercise_ids.add(row.item)
            correct_count += row.correct
            all_count += 1

    exercise_groups_df.loc[len(exercise_groups_df.index)] = [index, group, all_count, correct_count, last_day]

    return exercise_groups_df


def create_chosen_users_graph(data: pd.DataFrame, users: pd.Series, absolute_order_dict: Dict, chosen_users: Dict):
    """Create visualization for exercises for specific chosen users to show different behaviour patterns

    Args:
        data (pd.DataFrame): logs of exercises
        users (pd.Series): ids of users
        absolute_order_dict (Dict): webpage order of exercises - dictionary {exercise_id: order_on_webpage}
        chosen_users (Dict): ids of chosen users with behavious labels {user_id: behavoius label}
    """
    data['item_order'] = data['item'].map(absolute_order_dict)
    data['typ'] = data['user'].map(chosen_users)
    users_data = pd.DataFrame()

    for user in users:
        user_data = data[data['user'] == user]
        user_data.reset_index(inplace=True, drop=True)
        user_data.reset_index(inplace=True)
        user_data.columns = ['order'] + list(user_data.columns)[1:]

        if user == users.iloc[0]:
            users_data = user_data
        else:
            users_data = users_data.append(user_data, ignore_index=True)


    plt.figure(figsize=(20,10))

    sns.lineplot(data=users_data, x='order', y='item_order', hue='typ', palette='bright', units="user", estimator=None)
    sns.scatterplot(data=users_data, x='order', y='item_order', color='gray')

    plt.xlabel('Počet riešených úloh', fontsize=20)
    plt.ylabel('Poradie úlohy v systéme', fontsize=20)
    plt.legend(fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()
    plt.close()


def create_multiple_user_graph(data: pd.DataFrame, all_users_ids: List[pd.Series], absolute_order_dict: Dict, graph_dimensions: tuple[int, int] = (1,1)):
    """Create visualization for exercises for multiple users from the same cluster

    Args:
        data (pd.DataFrame): logs of exercises
        all_users_ids (List[pd.Series]): ids of users from 1 or more clusters
        absolute_order_dict (Dict): webpage order of exercises - dictionary {exercise_id: order_on_webpage}
        graph_dimensions (tuple[int, int]): graph dimensions number of rows, col of subplots
    """

    fig, subplots = plt.subplots(graph_dimensions[0], graph_dimensions[1], figsize=(20,10))

    ## if graph have more subplots on more rows flatten subplot array so we can iterate through it
    if graph_dimensions[0] > 1:
        subplots = [s for sub in subplots for s in sub]
    ## if there is only one subplot put it is not returned as list and needs to be put in one to be consistent
    if graph_dimensions == (1,1):
        subplots = [subplots]

    data['item_order'] = data['item'].map(absolute_order_dict)
    users_data = pd.DataFrame()

    for subplot_index, users in enumerate(all_users_ids):
        for user in users:
            user_data = data[data['user'] == user]
            user_data.reset_index(inplace=True, drop=True)
            user_data.reset_index(inplace=True)
            user_data.columns = ['order'] + list(user_data.columns)[1:]

            if user == users.iloc[0]:
                users_data = user_data
            else:
                users_data = users_data.append(user_data, ignore_index=True)


        subplots[subplot_index] = sns.lineplot(ax=subplots[subplot_index], data=users_data, 
                                               x='order', y='item_order', color='gray', 
                                               alpha=0.5, units='user', estimator=None)
        subplots[subplot_index] = sns.scatterplot(ax=subplots[subplot_index], data=users_data, 
                                                  x='order', y='item_order', hue='correct', 
                                                  palette=sns.diverging_palette(10, 150, n=2))

        ## subplot configurations
        subplots[subplot_index].set_ylim([0,250])
        subplots[subplot_index].xaxis.set_tick_params(labelsize=20)
        subplots[subplot_index].yaxis.set_tick_params(labelsize=20)
        subplots[subplot_index].set_xlabel("")
        subplots[subplot_index].set_ylabel("")
        subplots[subplot_index].set_title(f"Skupina {subplot_index+1}", fontsize=20)

        if graph_dimensions != (1, 1):
            subplots[subplot_index].legend().remove()

    ## plot configuratoins
    if graph_dimensions == (1,1):
        subplots[0].legend(fontsize=20, title_fontsize=23)
        subplots[0].legend_.set_title('Úspešnosť')
        for text, legend in zip(subplots[0].legend_.texts, ['neúspešný', 'úspešný']):
            text.set_text(legend)
    else:
        symbol, label = subplots[0].get_legend_handles_labels()
        leg = fig.legend(symbol, label, bbox_to_anchor=(0.55, 1.1), title_fontsize=20, fontsize=20)
        leg.set_title('Úspešnosť')
        for text, legend in zip(leg.texts, ['neúspešný', 'úspešný']):
            text.set_text(legend)
    
    fig.text(0.5, 0.04, 'Počet riešených úloh', ha='center', fontsize=25)
    fig.text(0.04, 0.5, 'Poradie úloh', va='center', rotation='vertical', fontsize=25)
    
    plt.show()
    plt.close()


def create_day_lines_plotly(fig, data, ymax, ymin=0, row=1):
    lines = get_day_lines_positions(data)
    for line in lines:
        fig.add_shape(yref='y', xref='x', x0=line[0], y0=ymin, x1=line[0], y1=ymax, type='line',
                      line=dict(color='black', dash='dot' if line[1] == 'small' else 'longdash'), row = row, col = 1)


def create_grouped_exercise_user_activity_plotly(data: pd.DataFrame, user_id: int, exercise_to_group: Dict, exercise_groups: Dict):
    """Create interactive visualization of exercises grouped to problem sets that user tried to solve

    Args:
        data (pd.DataFrame): logs of exercises
        user_id (int): specific user id
        exercise_to_group (Dict): dictionary {exercise_id: (group_name, group_id)}
        exercise_groups (Dict): dictionary {group_id: ([exercise_ids], group_name)}
    """
    grouped_df = get_grouped_exercise_df(data, user_id, exercise_to_group)
    grouped_df['success_rate'] = grouped_df['correct_count'] / grouped_df['all_count']
    grouped_df['success_bin'] = grouped_df['success_rate'].apply(round)
    grouped_df['success_bin'] = grouped_df['success_bin'].astype(str)
    grouped_df['success_rate'] = grouped_df['success_rate'].astype(float)
    grouped_df['all_count'] = grouped_df['all_count'].astype(int)
    grouped_df['group_name'] = grouped_df['group_order'].apply(lambda x: exercise_groups[x][1])

    fig = px.line(grouped_df, x='order', y='group_name')
    fig.add_trace(px.scatter(grouped_df, x='order', y='group_name', color_continuous_scale='rdylgn',
                             color='success_rate', size='all_count').data[0])
    fig.update_traces(line_color='lightgray')
    fig.update_traces(hovertemplate='Sada úloh: %{y} <br>Počet riešených úloh: %{marker.size} <br>Úspešnosť riešených úloh: %{marker.color:.2f}<extra></extra>')
    fig.update_coloraxes(colorscale='rdylgn')
    print("plotly express hovertemplate:", fig.data[1].hovertemplate)


    ## create day lines
    create_day_lines_plotly(fig, grouped_df, 29)

    ## configure graph
    x_axis_days = grouped_df.groupby('time').agg({'all_count': 'count', 'order': 'mean'}).reset_index()
    fig.update_xaxes(tickangle=90, tickmode='array', tickvals=x_axis_days['order'], ticktext=x_axis_days['time'])
    sorted_names = sorted(list(exercise_groups.items()))
    sorted_names = [item[1][1] for item in sorted_names]
    fig.update_yaxes(categoryarray=sorted_names, categoryorder='array')
    
    fig.update_layout(yaxis_title=None, xaxis_title='Aktívne dni',
    annotations=[
        dict(text="Úspešnosť", x=1, xanchor='left', xref="paper", y=1.1, yref="paper",
                             align="left", showarrow=False),])
    fig.show()
    return fig


## ERROR VISUALIZATIONS

## assigns color to keyboard keys based on number of errors for that key
def assign_color(max_error: int, error_count: Optional[int]) -> str:
    """assign one from 10 color options based on number of errors

    Args:
        max_error (int): maximum number of errors for one key
        error_count (Optional[int]): number of errors for current key

    Returns:
        str: rgb color
    """
    if not error_count:
        return "rgb(211,211,211)"
    color_spectrum_rgb = ["rgb(255, 255, 205)", "rgb(255, 239, 154)", "rgb(255, 229, 115)",
                          "rgb(255, 239, 83)", "rgb(255, 244, 67)", "rgb(255, 229, 57)",
                          "rgb(255, 211, 47)", "rgb(255, 198, 40)", "rgb(255, 183, 28)"]
    interval = max_error//10
    color_index = error_count//interval if error_count//interval < len(color_spectrum_rgb) else -1
    return color_spectrum_rgb[color_index]


def create_button_placement(specia_multiple_char_keys: Dict, char_errors: Dict) -> Dict:
    """Create button placement, error count (color) for keyboard 
       except for enter key

    Args:
        specia_multiple_char_keys (Dict): dictionary for upper row with 2 character options: +/1
        char_errors (Dict): dictionary of errors {character: {total: number_errors, character: number_errors, ...}}

    Returns:
        Dict: information for keyboard keys: {char: {name: key chars, x: list, y: list, error_total: number_erros, color: error_color}}
    """
    keys_info = dict()
    ## get maximum error count for all keys
    max_error_count = max([errors["total"] for errors in char_errors.values()])
    ## 3 most substituted mistakes for character
    most_frequent_errors_3 = dict()
    for key, item in char_errors.items():
        most_frequent_errors_3[key] = dict(sorted(item.items(), key=lambda x: x[1], reverse=True)[1:4])

    ## upper number keys
    for i, char in enumerate(";+ěščřžýáíé=´"):
        char_total_error = char_errors.get(char, {"total":0})["total"]
        keys_info[char] = {"name": specia_multiple_char_keys.get(char, char.upper()),
                           "x":[i, i+0.9, i+0.9, i, i], "y":[4, 4, 4.9, 4.9, 4],
                           "error_total": char_total_error,
                           "color": assign_color(max_error_count, char_total_error),
                           "frequent_errors": most_frequent_errors_3.get(char, {})}
    
    ## upper char keys
    for i, char in enumerate("qwertyuiopú)"):
        char_total_error = char_errors.get(char, {"total":0})["total"]
        keys_info[char] = {"name": specia_multiple_char_keys.get(char, char.upper()),
                           "x":[i+1.5, i+2.4, i+2.4, i+1.5, i+1.5], "y":[3, 3, 3.9, 3.9, 3],
                           "error_total": char_total_error,
                           "color": assign_color(max_error_count, char_total_error),
                           "frequent_errors": most_frequent_errors_3.get(char, {})}
    
    ## middle char keys
    for i, char in enumerate("asdfghjklů§¨"):
        char_total_error = char_errors.get(char, {"total":0})["total"]
        keys_info[char] = {"name": specia_multiple_char_keys.get(char, char.upper()),
                           "x":[i+1.8, i+2.7, i+2.7, i+1.8, i+1.8], "y":[2, 2, 2.9, 2.9, 2],
                           "error_total": char_total_error, 
                           "color": assign_color(max_error_count, char_total_error),
                           "frequent_errors": most_frequent_errors_3.get(char, {})}
    
    ## bottom char keys
    for i, char in enumerate("zxcvbnm,.-"):
        char_total_error = char_errors.get(char, {"total":0})["total"]
        keys_info[char] = {"name": specia_multiple_char_keys.get(char, char.upper()),
                           "x":[i+2.3, i+3.2, i+3.2, i+2.3, i+2.3], "y":[1, 1, 1.9, 1.9, 1],
                           "error_total": char_total_error,
                           "color": assign_color(max_error_count, char_total_error),
                           "frequent_errors": most_frequent_errors_3.get(char, {})}

    ## special keys
    for i, char in enumerate(["ctrl", "empty", "alt"]):
        keys_info[char] = {"name": char.upper() if char != "empty" else "", 
                           "x":[i*1.5, i*1.5+1.4, i*1.5+1.4, i*1.5, i*1.5], "y":[0, 0, 0.9, 0.9, 0],
                           "error_total": 0, "color": "rgb(211,211,211)"}
    for i, char in enumerate(["altgr", "empty2", "ctrl "]):
        keys_info[char] = {"name": char.upper() if char != "empty2" else "",
                           "x":[i*1.5+10.6, i*1.5+12, i*1.5+12, i*1.5+10.6, i*1.5+10.6],
                           "y":[0, 0, 0.9, 0.9, 0], "error_total": 0, "color": "rgb(211,211,211)"}

    keys_info["backspace"] = {"name": "BACKSPACE", "x":[13, 15, 15, 13, 13], "y":[4, 4, 4.9, 4.9, 4],
                              "error_total": 0, "color": "rgb(211,211,211)"}
    keys_info["tab"] =       {"name": "TAB", "x":[0, 1.4, 1.4, 0, 0], "y":[3, 3, 3.9, 3.9, 3],
                              "error_total": 0, "color": "rgb(211,211,211)"}
    keys_info["caps"] =      {"name": "CAPS", "x":[0, 1.7, 1.7, 0, 0], "y":[2, 2, 2.9, 2.9, 2],
                              "error_total": 0, "color": "rgb(211,211,211)"}
    shift_errors = char_errors.get("shift", {"total":0})["total"]
    space_errors = char_errors.get(" ", {"total":0})["total"]
    keys_info["shift"] = {"name": "SHIFT", "x":[0, 2.2, 2.2, 0, 0], "y":[1, 1, 1.9, 1.9, 1], 
                          "x0": 0, "y0": 1, "x1": 2.2, "y1": 1.9, "error_total": shift_errors, 
                          "color": assign_color(max_error_count, shift_errors), 
                          "frequent_errors": most_frequent_errors_3.get("shift", {})}
    keys_info["shift2"] = {"name": "SHIFT", "x":[12.3, 15, 15, 12.3, 12.3], "y":[1, 1, 1.9, 1.9, 1],
                           "x0": 12.3, "y0": 1, "x1": 15, "y1": 1.9, "error_total": 0, 
                           "color": "rgb(211,211,211)"}
    keys_info["space"] = {"name": "SPACE", "x":[4.5, 10.5, 10.5, 4.5, 4.5], "y":[0, 0, 0.9, 0.9, 0],
                          "x0": 4.5, "y0": 0, "x1": 10.5, "y1": 0.9, "error_total": space_errors,
                          "color": assign_color(max_error_count, space_errors),
                          "frequent_errors": most_frequent_errors_3.get(" ", {})}

    return keys_info


def custom_hover_template(info: Dict) -> str:
    """Create custom hover template for keyboard visualization

    Args:
        info (Dict): information for key {name: key char, error_total: number of errors for key, frequent_errors: {3 most frequent errors for key}}

    Returns:
        str: custom hower template
    """
    ## <extra></extra> removes unwanted trace tag at the end of hover text info
    if info.get("frequent_errors", {}):
        return f'{info["name"].replace("</br></br>", " / ")}<br><i>Total errors</i>: {info["error_total"]}' + "".join([f'<br><i>Typed {mistake if mistake != " " else "SPACE"}</i>: {error_count}' for mistake, error_count in info['frequent_errors'].items()]) + '<extra></extra>'
    return f'{info["name"].replace("</br></br>", " / ")}<br><i>Total errors</i>: {info["error_total"]}<extra></extra>'


def create_keyboard(fig, specia_multiple_char_keys: Dict, char_errors: Dict):
    """Create visualization of errors for each key for user in system

    Args:
        fig (go.Figure): information for keyboard keys: {char: {name: key chars, x: list, y: list, error_total: number_erros, color: error_color}}
        specia_multiple_char_keys (Dict): dictionary for upper row with 2 character options: +/1
        char_errors (Dict): dictionary of errors {character: {total: number_errors, character: number_errors, ...}}

    """
    ## enter key
    keys_placement = create_button_placement(specia_multiple_char_keys, char_errors)
    fig.add_trace(go.Scatter(x=[13.5,13.5,15,15,13.8,13.8, 13.5], y=[3,3.9,3.9,2,2,3,3], 
                             fill="toself", marker=dict(size=1), fillcolor="rgb(211,211,211)", 
                             showlegend = False, hoverinfo="none", line=dict(color="RoyalBlue")))
    ## upper row number keys
    for char, info in keys_placement.items():
        fig.add_trace(go.Scatter(x=info["x"], y=info["y"], fill="toself", marker=dict(size=1), 
                                 fillcolor=info['color'], showlegend = False, hoverinfo="none",
                                 line=dict(color="RoyalBlue")))
        if char == "backspace":
            fig.add_trace(go.Scatter(x=[info["x"][0]+1], y=[info["y"][0]+0.4], text=[info["name"]],
                                     mode="text", showlegend = False,
                                     hoverinfo = "none"))
        elif char == "altgr":
            fig.add_trace(go.Scatter(x=[info["x"][0]+0.7], y=[info["y"][0]+0.4], text=[info["name"]],
                                     mode="text", showlegend = False,
                                     hoverinfo = "none"))
        elif char == "space":
            fig.add_trace(go.Scatter(x=[info["x"][0]+3], y=[info["y"][0]+0.4], text=[info["name"]],
                                     mode="text", showlegend = False,
                                     hovertemplate = custom_hover_template(info)))

        else:
            fig.add_trace(go.Scatter(x=[info["x"][0]+0.5], y=[info["y"][0]+0.4], text=[info["name"]],
                                     mode="text", showlegend = False,
                                     hovertemplate = custom_hover_template(info)))
    fig.add_trace(go.Scatter(x=[14.3], y=[3.4], text=["ENTER"],
                             mode="text", showlegend = False,
                             hoverinfo = "none"))


    fig.add_shape(type="rect",
        x0=4.5, y0=0, x1=10.5, y1=0.9,
        line=dict(color="RoyalBlue"),)

    fig.show()
    return fig


def silhouette_visualization(chosen_features: pd.DataFrame, n_clusters: int):
    """Create clustering based on chosen features and show visualization of silhouette
       score for this clustering

    Args:
        chosen_features (pd.DataFrame): chosen features for clustering
        n_clusters (int): number of clusters
    """
    _, ax1 = plt.subplots(1, 1)

    # The silhouette coefficient can range from -1, 1 but in this example all
    ax1.set_xlim([-1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(chosen_features) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters)
    cluster_labels = clusterer.fit_predict(chosen_features)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(chosen_features, cluster_labels)
    print(f"For n_clusters = {n_clusters} The average silhouette_score is : {silhouette_avg}"
    )

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(chosen_features, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        cmap = cm.get_cmap('nipy_spectral')
        color = cmap(float(i) / n_clusters)

        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("Vizualizácia silhouette skóre pre jednotlivé zhluky")
    ax1.set_xlabel("Hodnoty silhouette koeficientov")
    ax1.set_ylabel("Označené zhluky")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.show()


def create_histograms_for_clusters(hist_parameters: List[str], scales: List[str], features: pd.DataFrame, cluster_number: int, cluster_algo: str, feature_number: int, label_start: int=0):
    """Create histogram for each cluster to better interpret in which features are users in cluster similar

    Args:
        hist_parameters (List[str]): parameters for histograms
        scales (List[str]): scales for featues in histograms
        features (pd.DataFrame): chosen features for which were the clusters created
        cluster_number (int): number of final clusters
        cluster_algo (str): column name of cluster labels
        feature_number (int): number of chosen features
        label_start (int, optional): starting cluster label for DBSCAN it can be -1. Defaults to 0.
    """
    for cluster_index in range(label_start, cluster_number):
        cluster = features[features[cluster_algo] == cluster_index]
        print(len(cluster))
        rows_num = feature_number // 2 if feature_number % 2 == 0 else feature_number // 2 + 1
        fig, axis = plt.subplots(nrows=rows_num, ncols=2 ,figsize=(15,7.5*rows_num))
        fig.suptitle(f'cluster {cluster_index}: {len(cluster)} users', fontsize=20)

        for i, (index, name, xlabel_name, ylabel_name) in enumerate(hist_parameters):
            values = features[index]
            cluster_values = cluster[index]
            axis[i // 2, i % 2].set_title(name)
            axis[i // 2, i % 2].set_xlabel(f'{xlabel_name}\nmax: {cluster_values.max()} \
                                              min: {cluster_values.min()} \
                                              avg: {cluster_values.sum()/len(cluster)}')
            axis[i // 2, i % 2].set_ylabel(ylabel_name)
            axis[i // 2, i % 2].set_yscale(scales[i])
            custom_bins = np.linspace(values.min(), values.max(), 20)
            axis[i // 2, i % 2].hist(values, bins=custom_bins, alpha=0.5)
            axis[i // 2, i % 2].hist(cluster_values, bins=custom_bins, alpha=0.5)

        plt.show()
        plt.close()


def create_user_activity_plotly(user_data: pd.DataFrame, exercise_data: pd.DataFrame):
    """Create interactive visualization of exercises that user tried to solve
       Buttons: after zooming in/out you can change marker size with buttons to
                - mala - for large number of exercise logs
                - velka - for smaller number of exercise logs
                - absolutna - not recommended (disadvantages explained in thesis) - absolute size that can range 0-30 000

    Args:
        user_data (pd.DataFrame): logs or exercises for specific user
        exercise_data (pd.DataFrame): information about exercises: id, name, ...
    """
    exercise_name_dict = dict(zip(exercise_data['item_order'], exercise_data['name']))
    correct_col = {"1": "green", "0": "red"}
    user_data['item_name'] = user_data['item_order'].map(exercise_name_dict)
    user_data['color'] = user_data['correct'].map(correct_col)
    user_data['resTimeSek'] = user_data['responseTime'] / 1000

    x_axis_days = user_data.groupby('time').agg({'item': 'count', 'index': 'mean'}).reset_index()

    fig = make_subplots(rows=2, cols=1, row_heights=[0.2, 0.8], vertical_spacing = 0.02, shared_yaxes=False, shared_xaxes=True)

    ## max size for resizing traces, 20 is default in plotly
    max_size = 20

    print(user_data.columns)
    ## individualy add traces to show trace legend
    for limit, name, color in zip([2, 4, 8, 16],
                                  ['pod limit', 'ok', 'mierne nad limit', 'extra nad limit'],
                                  ['#800080', '#2b83ba', '#fdae61', '#d7191c']):
        user_data_limit = user_data[user_data['moves_size_small'] == limit]
        fig.add_trace(go.Bar(customdata=user_data_limit, x=user_data_limit['index'],
                             y=user_data_limit['moves'], name = name, legendgroup='1',
                             marker_color=color, legendgrouptitle_text="Limit znakov"), row = 1, col = 1)

    user_data_green = user_data[user_data['correct'] == '1']
    fig.add_trace(go.Scatter(customdata=user_data_green ,x=user_data_green['index'], y=user_data_green['item_name'],
                             mode='markers', name='uspesny', legendgroup='2', legendgrouptitle_text="Úspešnosť",
                             marker_color='green', marker_size=user_data_green['moves'], marker_sizemode="area",
                             marker_sizeref = user_data["moves"].max() / max_size ** 2), row = 2, col = 1)
    user_data_red = user_data[user_data['correct'] == '0']
    fig.add_trace(go.Scatter(customdata=user_data_red ,x=user_data_red['index'], y=user_data_red['item_name'],
                             mode='markers', name='neuspesny', legendgroup='2',
                             marker_color='red', marker_size=user_data_red['moves'], marker_sizemode="area",
                             marker_sizeref = user_data["moves"].max() / max_size ** 2), row = 2, col = 1)

    fig.update_traces(hovertemplate='Úloha: %{y} <br> Počet znakov: %{customdata[6]} <br>Čas riešenia v sek: %{customdata[20]} <br>Počet chýb: %{customdata[11]}<extra></extra>')
    fig.add_trace(go.Scatter(x=user_data['index'], y=user_data['item_name'], mode='lines', line=dict(color="lightgray"), showlegend=False), row = 2, col = 1)


    ## create day lines
    create_day_lines_plotly(fig, user_data, 240, row=2)

    ## configure graph
    fig.update_layout(autosize=True, height=700, yaxis_title=None, xaxis2_title='Aktívne dni', legend_tracegroupgap = 80, legend_itemsizing='constant',
                      annotations=[
                            dict(text="Ovládanie veľkosti:", x=1.02, xanchor='left', xref="paper", y=0.5, yref="paper",
                                 align="left", showarrow=False),])
    fig.update_xaxes(tickfont_size=10, tickangle=90, tickmode='array', tickvals=x_axis_days['index'], ticktext=x_axis_days['time'])

    exercise_name_order = exercise_data[['name', 'item_order']]
    sorted_df = exercise_name_order.sort_values(by='item_order')
    fig.update_yaxes(categoryarray=sorted_df['name'], categoryorder='array')
    
    add_resize_buttons(fig, user_data, user_data_green, user_data_red)
    fig.show()
    return fig

def add_resize_buttons(fig, user_data, user_data_green, user_data_red):
    max_size = 20
    fig.update_layout(updatemenus=[
                    dict(
                        active=2,
                        buttons=[
                        dict(
                            args=[{"marker.size": [user_data_green['moves_size_small'], user_data_red['moves_size_small']],
                            "marker.sizeref": user_data["moves_size_small"].max() / (max_size / 3) ** 2}],
                            label='malá',
                            method='update'
                        ),
                            dict(
                            args=[{"marker.size": [user_data_green['moves_size_small'], user_data_red['moves_size_small']],
                            "marker.sizeref": user_data["moves_size_small"].max() / (max_size * 1.5) ** 2}],
                            label='veľká',
                            method='update'
                        ),
                        dict(
                            args=[{"marker.size": [user_data_green['moves'], user_data_red['moves']],
                            "marker.sizeref": user_data["moves"].max() / max_size ** 2}],
                            label='absolútná',
                            method='update'
                        )
                        ],
                    xanchor='left',
                    x=1.02,
                    y=0.45,
                    type='buttons',
                    name='marker size',),])
