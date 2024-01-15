import argparse
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
import matplotlib.cm as cm
from helper import filter_similar_and_get_best_rows, get_best_index

def create_plots(df_params, file_suffix = "manual", file_name = None):

    df_test_data = None

    if file_name != "df_params.pkl":
        # replace "_train" with "_test"
        test_data_file = file_name.replace("_train", "_test")
        df_test_data = pd.read_pickle(test_data_file)

    plt.clf()

    # df_params is your DataFrame
    df_params, best_rows = filter_similar_and_get_best_rows(df_params)

    # Create a scatter plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(best_rows['open_positions'], best_rows['overall_profit_24'], c=best_rows['score'], cmap='viridis')

    # Add a colorbar as the legend
    cbar = plt.colorbar(scatter)
    cbar.set_label('Score')

    # Add labels to the axes
    plt.xlabel('Open Positions')
    plt.ylabel('Overall Profit 24')

    # Add the index next to each data point
    for i, row in best_rows.iterrows():
        plt.text(row['open_positions'], row['overall_profit_24'], str(int(row['original_index'])))

    # Show the plot
    plt.savefig("best_rows_" + file_suffix + ".png")

    '''score over profit'''

    # Create a new column for the sum of overall_profit_24 and overall_profit_72
    # best_rows['total_profit'] = best_rows['overall_profit_24'] + best_rows['overall_profit_72']

    # Filter the DataFrame
    # best_rows = best_rows[best_rows['min_percent_change_24'] <= 20]

    # Generate a colormap with as many colors as there are data points
    colors = cm.rainbow(np.linspace(0, 1, len(best_rows)))

    # Create a dictionary that maps each 'original_index' to its corresponding color
    color_dict = {int(row['original_index']): colors[i] for i, (_, row) in enumerate(best_rows.iterrows())}

    # Fit the scaler on the training data
    scaler = MinMaxScaler().fit(best_rows[['score', 'overall_profit_24_rounded']])

    # Use the scaler to transform the 'score' and 'overall_profit_24_rounded' columns of the training data
    best_rows[['score', 'overall_profit_24_rounded']] = scaler.transform(best_rows[['score', 'overall_profit_24_rounded']])

    # Use the same scaler to transform the 'score' and 'overall_profit_24_rounded' columns of the test data
    if df_test_data is not None:
        df_test_data.loc[:, 'overall_profit_24_rounded'] = df_test_data['overall_profit_24'].round()
        df_test_data[['score', 'overall_profit_24_rounded']] = scaler.transform(df_test_data[['score', 'overall_profit_24_rounded']])

    # above_right_points and middle_point are defined somewhere in your script
    best_index, middle_point = get_best_index(best_rows)

    # Plot the data
    plt.figure(figsize=(10, 8))
    # plt.scatter(best_rows['score'], best_rows['overall_profit_24_rounded'], color='gray', label='Training Data')

    # Highlight the middle point
    plt.scatter(middle_point[0], middle_point[1], color='black', s=100, marker='x')

    # Highlight the best point
    plt.scatter(best_rows.loc[best_index, 'score'], best_rows.loc[best_index, 'overall_profit_24_rounded'], color='red', s=100, edgecolor='black', label="Best")

    # Add labels to the axes
    plt.xlabel('Score')
    plt.ylabel('overall_profit_24_rounded')

    x_range = best_rows['score'].max() - best_rows['score'].min()
    y_range = best_rows['overall_profit_24_rounded'].max() - best_rows['overall_profit_24_rounded'].min()

    x_offset = x_range * 0.01  # 1% of the x range
    y_offset = y_range * 0.01  # 1% of the y range

    # Add the index next to each data point
    for i, row in best_rows.iterrows():
        plt.text(row['score'] + x_offset, row['overall_profit_24_rounded'] + y_offset, str(int(row['original_index'])))

    # Plot the data points with their corresponding color
    for i, (index, row) in enumerate(best_rows.iterrows()):
        plt.scatter(row['score'], row['overall_profit_24_rounded'], color=color_dict[int(row['original_index'])])

    # Plot the test data
    if df_test_data is not None and df_test_data['best_row_index'].iat[0] in best_rows['original_index'].values:
        # Get the left limit of the x-axis
        x_left_limit = plt.gca().get_xlim()[0]

        # Get the color for the test data point based on its 'best_row_index'
        test_data_color = color_dict[df_test_data['best_row_index'].iat[0]]
        # plt.scatter(x_left_limit, df_test_data['overall_profit_24_rounded'].iat[0], color=test_data_color, edgecolors='black', s=100)

        # Add a label next to the test data point
        # plt.text(x_left_limit, df_test_data['overall_profit_24_rounded'].iat[0], "Tested: " + str(df_test_data['best_row_index'].iat[0]))

        # Annotate the point on the y-axis
        plt.annotate(str(df_test_data['best_row_index'].iat[0]) + " tested" + 
             "\nProfit: " + str(int(df_test_data['overall_profit_24'].iat[0])) +
             "\n t.p.d.: " + str(int(df_test_data['trades_per_day'].iat[0])),
             xy=(x_left_limit, df_test_data['overall_profit_24_rounded'].iat[0]), 
             xytext=(-25, -10), 
             textcoords='offset points',
             arrowprops=dict(facecolor=test_data_color, edgecolor=test_data_color, arrowstyle="->"),
             horizontalalignment='right')

    # Create a custom legend for each data point
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=f"{df_params.loc[index, 'original_index']}: Score: {df_params.loc[index, 'score']}, Profit: {df_params.loc[index, 'overall_profit_24_rounded']}", markerfacecolor=colors[i], markersize=10) for i, (index, row) in enumerate(best_rows.iterrows())]
    
    # Add the custom legend
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust the space around the plot to leave space for the legend
    plt.subplots_adjust(right=0.7)

    # Show the plot
    plt.savefig("score_vs_overall_profit_24_" + file_suffix + ".png")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='hyperopting script')
    parser.add_argument('--file_name', type=str, default="df_params.pkl", required=True)

    # Parse the arguments
    args = parser.parse_args()

    # Get the settings and action
    file_name = args.file_name

    df_params = pd.read_pickle(file_name)
    create_plots(df_params, "manual", file_name)