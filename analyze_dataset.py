import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

import json
import glob
import os
import argparse
from typing import Tuple, Union, List
from collections import Counter
from tqdm import tqdm
from multiprocessing import Pool


pd.options.mode.chained_assignment = None  # default='warn'


# ====================================================================


def get_data(json_pth: Union[str, os.PathLike]) -> dict:
    """Get a single data from the given file.json path"""
    with open(json_pth, 'r') as f:
        data = json.load(f)
    # Remove unwanted/extra data
    for extra in ['ego_position', 'acceleration']:
        try:
            del data[extra]
        except KeyError:
            pass
    data['parent_path'] = os.path.dirname(json_pth)
    data['file_name'] = os.path.basename(json_pth)
    return data


def get_original_df(
        path: Union[str, os.PathLike],
        filename: str,
        processes_per_cpu: int = 2,
        ignore_npy: bool = False) -> pd.DataFrame:
    """Get a DataFrame from all the can_bus*.json files in the dataset"""
    save_path = os.path.join(os.getcwd(), 'data_analysis', filename)

    if os.path.isfile(save_path) and not ignore_npy:
        print('.npy file exists, loading it...')
        data = list(np.load(save_path, allow_pickle=True))
    else:
        # Construct the dataset
        msg = '.npy file ignored, constructing it again...' if ignore_npy else '.npy file not found, constructing it...'
        print(msg)
        all_data_paths = sorted(glob.glob(os.path.join(path, '**/can_bus*.json'), recursive=True))

        with Pool(os.cpu_count() * processes_per_cpu) as p:
            data = list(tqdm(p.imap(get_data, all_data_paths), total=len(all_data_paths)))

        np.save(save_path, data)

    # Create dataframe with the data
    df = pd.DataFrame(data)
    print('Numerical values in dataframe...')
    print(df.describe(include=[np.number]))
    print('String values in dataframe...')
    print(df.describe(include='O'))

    return df


# ====================================================================


def violin_plot(df: pd.DataFrame, save_name: str) -> None:
    """Save violin plot for the interesting parameters using df"""
    direction_dict = {'No Action': 4.0, 'Turn Left': 1.0, 'Turn Right': 2.0, 'Continue Straight': 3.0}

    # Auxiliary function for setting the quartile lines
    def set_lines(ax):
        for l in ax.lines:
            l.set_linestyle('--')
            l.set_linewidth(0.6)
            l.set_color('white')
            l.set_alpha(0.7)
        for l in ax.lines[1::3]:
            l.set_linestyle('-')
            l.set_linewidth(1.3)
            l.set_color('black')
            l.set_alpha(0.8)

    for key in direction_dict:
        # Get respective subset of the dataframe
        data = df[df['direction'] == direction_dict[key]]
        fig = plt.figure(figsize=(8, 6))
        gs = fig.add_gridspec(1, 4)

        fig.add_subplot(gs[0, 0])
        ax = sns.violinplot(y='steer', data=data, color='r', inner='quartile')
        set_lines(ax)

        fig.add_subplot(gs[0, 1])
        ax = sns.violinplot(y='throttle', data=data, color='g', inner='quartile')
        set_lines(ax)

        fig.add_subplot(gs[0, 2])
        ax = sns.violinplot(y='brake', data=data, color='b', inner='quartile')
        set_lines(ax)

        fig.add_subplot(gs[0, 3])
        ax = sns.violinplot(y='speed', data=data, color='m', inner='quartile')
        set_lines(ax)

        # When using tight layout, we need the title to be spaced accordingly
        fig.tight_layout()
        fig.subplots_adjust(top=0.88)

        stitle = f'Direction: {key} - $N={len(data)}$ - ${100 * len(data)/len(df):6.3f}$% of total'
        fig.suptitle(stitle, fontsize=16)

        fname = f'{save_name}-{key.replace(" ", "")}'
        fig_name = os.path.join(os.getcwd(), 'data_analysis', save_name, 'violin_plots', f'{fname}.png')
        os.makedirs(os.path.join(os.getcwd(), 'data_analysis', save_name, 'violin_plots'), exist_ok=True)
        plt.savefig(fig_name)
        plt.close()


# ====================================================================


def plot_routes(path: Union[str, os.PathLike], df: pd.DataFrame, ignore_directions: bool = False) -> None:
    """Plot the steer, throttle, brake, and speed of the vehicle collecting data in a route"""
    # Get dataset name and make the necessary directories
    dataset_name = os.path.basename(path)
    s_path = os.path.join(os.getcwd(), 'data_analysis', dataset_name, 'routes')
    os.makedirs(s_path, exist_ok=True)

    # Get the number of routes/cars that collected the data
    routes = df['parent_path'].unique().tolist()
    num_routes = len(routes)

    # Get the name of each client (its container and client number)
    named_routes = [rt.split(os.sep) for rt in routes]
    simple_route_names = [rt[-1] for rt in named_routes]  # RouteScenario_0000
    named_routes = sorted([f'{rt[-2]}_{rt[-1]}' for rt in named_routes])  # e.g., Roach_NoCrashRoutes_T1_4W_RouteScenario_0000

    # Total number of frames
    total_frames = len(df)
    print(f'Dataset size: {total_frames / (10 * 60 * 60):.2f} hours')  # Hardcoded 10 fps data collection!

    # Aux function
    def get_change_locs(df: pd.DataFrame) -> Tuple[List[int], List[float]]:
        """Get the index and direction from the df of the actions taken by the client"""
        df['direction_str'] = df['direction'].astype(str)  # In order to compare, turn direction into a string
        # Shift direction column by 1 (filling the top with the head), and compare to the original
        df['change'] = df['direction_str'].shift(1, fill_value=df['direction_str'].head(1)) != df['direction_str']

        # Get the rows where there's a change
        index_change = list(df['change'][df['change']].index.values)
        # Add the first for correctly coloring the background
        index_change = [0] + index_change
        # For these indexes, get the value of the direction
        dirs = list(df['direction'][index_change].values)
        # Add the last frame for correctly coloring the background
        index_change = index_change + [len(df) - 1]

        return index_change, dirs

    # Dictionaries containing the name and color for plotting the direction given to the car
    my_labels = {4.0: 'No Action', 1.0: 'Turn Left', 2.0: 'Turn Right', 3.0: 'Continue Straight'}
    colors = {4.0: 'gold', 1.0: 'gray', 2.0: 'cyan', 3.0: 'magenta'}

    # Initialize the total counts per action
    total_action_counts = Counter({1.0: 0, 2.0: 0, 3.0: 0, 4.0: 0})
    max_speed_routes = {}

    idx_change_routes = {}
    dirs_routes = {}

    # Make a plot for each client
    for route in tqdm(simple_route_names, total=num_routes, unit='routes'):
        named_route_idx = [idx for idx, rt in enumerate([route in nr for nr in named_routes]) if rt][0]
        # Get the dataframe for this route; TODO
        df_route = df[df['parent_path'].str.contains(route)]
        df_route.sort_values('file_name', inplace=True, ascending=True)

        frames_per_route = len(df_route)

        # The actual max speed for this route
        actual_max_speed = df_route['speed'].max()
        # Normalize the speed
        df_route['speed'] = df_route['speed'].div(actual_max_speed)  # normalize to range [0, 1]

        # Save this max speed per client
        max_speed_routes[named_routes[named_route_idx]] = actual_max_speed

        # Build the plot
        fig, ax = plt.subplots(figsize=(48, 16))
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        # We reindex the client's dataframe in order to have the correct frame number when plotting it
        new_index = [pd.Index(range(frames_per_route))]
        df_route = df_route.set_index(new_index)
        df_route.plot(y=['steer', 'throttle', 'brake', 'speed'], ax=ax)

        # Set the area colors for when a direction is taken
        if not ignore_directions:
            idx_change, dirs = get_change_locs(df_route)
            for idx, dir in enumerate(dirs):
                ax.axvspan(idx_change[idx], idx_change[idx + 1], facecolor=colors[dir], alpha=0.5, label=my_labels[dir])

            # Save these indexes and direction for each client
            idx_change_routes[f'{named_routes[named_route_idx]}'] = [int(idx) for idx in idx_change]
            dirs_routes[f'{named_routes[named_route_idx]}'] = [float(d) for d in dirs]

            # Count the directions taken by the client
            dirs_count = Counter(dirs)
            # Add this to the total for the whole dataset
            total_action_counts += dirs_count
            # Add the counts to the title
            total_actions = ''
            for key in my_labels:
                total_actions += f' - {my_labels[key]}: {dirs_count[key]}'
        else:
            total_actions = ''
        # Set title and x and y axes labels
        suptitle = f'{named_routes[named_route_idx]} - Max speed: {actual_max_speed:.4f} m/s'
        suptitle = f'{suptitle}{total_actions}'
        plt.suptitle(suptitle, fontsize=30)
        plt.xlabel('Frame idx', fontsize=22)
        plt.ylabel('Normed value', fontsize=22)
        plt.xticks(list(range(0, frames_per_route + 1, len(df_route) // 20)))  # ticks in 5% increments

        # Fix the legend / remove duplicated areas and labels
        hand, labl = ax.get_legend_handles_labels()
        handout = []
        lablout = []
        for h, l in zip(hand, labl):
            if l not in lablout:
                lablout.append(l)
                handout.append(h)

        ax.legend(handout, lablout, fontsize='x-large')
        sname = os.path.join(s_path, named_routes[named_route_idx])
        plt.savefig(f'{sname}.png', dpi=300)
        plt.close()

    # Add summary and save it as a JSON file
    if not ignore_directions:
        actions_summary = {
            'avg_no_action': total_action_counts[4.0] / num_routes,
            'avg_turn_left': total_action_counts[1.0] / num_routes,
            'avg_turn_right': total_action_counts[2.0] / num_routes,
            'avg_continue_straight': total_action_counts[3.0] / num_routes
        }
    else:
        actions_summary = {}

    summary = {
        'num_routes': num_routes,
        'num_frames_per_route': frames_per_route,
        'hours_per_client': frames_per_route / (10 * 60 * 60),
        'total_action_counts': total_action_counts,
        'actions_summary': actions_summary,
        'max_speed_clients': max_speed_routes,
        'idx_change_clients': idx_change_routes,
        'dirs_clients': dirs_routes
    }

    with open(os.path.join(s_path, f'{dataset_name}-summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)


# ====================================================================


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str, help='Path to the head of the dataset', required=True)
    parser.add_argument('--processes-per-cpu', '-proc', type=int, help='Processes per cpu (default: %(default)s)', default=2)
    parser.add_argument('--plot-routes', action='store_true', help='Add flag to plot the actions and speed of a client')
    parser.add_argument('--ignore-directions', '-id', action='store_true', help='Add flag to to ignore the commands (violin plots and background in plot)')
    parser.add_argument('--redo-npy', '-rn', action='store_true', help='Add flag to recreate the (non-augmented) .npy file, even if one already exists')

    args = parser.parse_args()

    # Create dir if it doesn't exist
    if not os.path.exists(os.path.join(os.getcwd(), 'data_analysis')):
        os.mkdir(os.path.join(os.getcwd(), 'data_analysis'))

    print('Getting the dataframe...')
    save_name = os.path.basename(args.path)
    filename = f'{save_name}.npy'
    df = get_original_df(args.path, filename, args.processes_per_cpu, args.redo_npy)

    # Create and save the violin plots
    if not args.ignore_directions:
        print('Plotting data...')
        violin_plot(df, save_name)

    if args.plot_routes:
        message = f'Plotting actions taken by all routes in {args.path}'
        message = f'{message} (ignoring directions)...' if args.ignore_directions else f'{message}...'
        print(message)
        plot_routes(path=args.path, df=df, ignore_directions=args.ignore_directions)

    print('Done!')


# ====================================================================


if __name__ == '__main__':
    main()


# ====================================================================
