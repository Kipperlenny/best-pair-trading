import argparse
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd

def create_plots(df_params, file_suffix = "_manual"):
    plt.clf()
    # Reset the index of df_params
    df_params = df_params.reset_index()

    # Round 'open_positions' and 'overall_profit_24' to 2 decimal places
    df_params['open_positions_rounded'] = df_params['open_positions'].round(2)
    df_params['overall_profit_24_rounded'] = df_params['overall_profit_24'].round(2)

    # Filter rows with positive 'overall_profit_24_rounded'
    df_params = df_params[df_params['overall_profit_24_rounded'] > 0]

    # Group by 'open_positions_rounded' and 'overall_profit_24_rounded', and keep only the row with the highest score in each group
    df_params = df_params.loc[df_params.groupby(['open_positions_rounded', 'overall_profit_24_rounded'])['score'].idxmax()]

    # Remove similar points
    best_rows = df_params.drop_duplicates(subset=['score', 'overall_profit_24'])

    # Get the 10 best rows
    best_rows = df_params.nlargest(20, 'score')

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
        plt.text(row['open_positions'], row['overall_profit_24'], str(int(row['index'])))

    # Show the plot
    plt.savefig("best_rows_" + file_suffix + ".png")

    '''score over profit'''

    # Create a new column for the sum of overall_profit_24 and overall_profit_72
    # best_rows['total_profit'] = best_rows['overall_profit_24'] + best_rows['overall_profit_72']

    # Filter the DataFrame
    # best_rows = best_rows[best_rows['min_percent_change_24'] <= 20]

    # Create a scatter plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(best_rows['score'], best_rows['overall_profit_24'], c=best_rows['score'], cmap='viridis')

    # Add a colorbar as the legend
    cbar = plt.colorbar(scatter)
    cbar.set_label('Score')

    # Add labels to the axes
    plt.xlabel('Score')
    plt.ylabel('overall_profit_24')

    # Add the index next to each data point
    for i, row in best_rows.iterrows():
        plt.text(row['score'], row['overall_profit_24'], str(int(row['index'])))

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
    create_plots(df_params)