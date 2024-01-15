from sklearn.preprocessing import StandardScaler
from sklearn.cluster import OPTICS
import numpy as np

def filter_similar_and_get_best_rows(df_params):
    # Store the original index in a new column
    df_params['original_index'] = df_params.index

    # Reset the index of df_params
    df_params.reset_index(drop=True, inplace=True)

    # Create a copy of the specified columns and normalize them
    df_params_copy = df_params[['profit_margin', 'min_percent_change_24', 'min_percent_change_1', 'max_percent_change_24', 'max_percent_change_1', 'trades']].copy()
    scaler = StandardScaler()
    df_params_copy = scaler.fit_transform(df_params_copy)

    # Define the OPTICS model
    optics = OPTICS(min_samples=2)

    # Fit the model to the copied and normalized columns
    optics.fit(df_params_copy)

    # Add the OPTICS labels to the original dataframe
    df_params['labels'] = optics.labels_

    # Remove the rows that are marked as noise (label -1) and create a copy of the DataFrame
    df_params = df_params[df_params['labels'] != -1].copy()

    # Drop the 'labels' column as it's no longer needed
    df_params.drop(['labels'], axis=1, inplace=True)

    # Round 'overall_profit_24' and store the result in 'overall_profit_24_rounded'
    df_params.loc[:, 'overall_profit_24_rounded'] = df_params['overall_profit_24'].round()

    # Filter rows with positive 'overall_profit_24_rounded'
    df_params = df_params[df_params['overall_profit_24_rounded'] > 0]

    # Remove similar points
    df_params = df_params.drop_duplicates(subset=['score', 'overall_profit_24_rounded'])

    # Normalize 'score' and 'overall_profit_24_rounded' to the same scale
    df_params['score_normalized'] = df_params['score'] / df_params['score'].max()
    df_params['overall_profit_24_rounded_normalized'] = df_params['overall_profit_24_rounded'] / df_params['overall_profit_24_rounded'].max()

    # Define the OPTICS model
    optics = OPTICS(min_samples=5, max_eps=0.1)

    # Fit the model to 'score_normalized' and 'overall_profit_24_rounded_normalized'
    optics.fit(df_params[['score_normalized', 'overall_profit_24_rounded_normalized']])

    # Add the OPTICS labels to the dataframe
    df_params['labels'] = optics.labels_

    # Remove the rows that are marked as noise (label -1)
    df_params = df_params[df_params['labels'] != -1]

    # Drop the 'labels', 'score_normalized' and 'overall_profit_24_rounded_normalized' columns as they're no longer needed
    df_params.drop(['labels', 'score_normalized', 'overall_profit_24_rounded_normalized'], axis=1, inplace=True)

    # Get the 20 best rows
    best_rows = df_params.nlargest(20, 'score')

    return df_params, best_rows

def get_best_index(best_rows):
    # Calculate the middle point
    middle_point = [(best_rows['score'].max() + best_rows['score'].min()) / 2, (best_rows['overall_profit_24_rounded'].max() + best_rows['overall_profit_24_rounded'].min()) / 2]
    
    # Get the points that are both to the right and above the middle
    above_right_points = best_rows[(best_rows['score'] > middle_point[0]) & (best_rows['overall_profit_24_rounded'] > middle_point[1])]

    if not above_right_points.empty:
        # If there are points that are both to the right and above the middle,
        # calculate the Euclidean distance to the top right corner for each of these points
        top_right_corner = [best_rows['score'].max(), best_rows['overall_profit_24_rounded'].max()]
        distances_to_top_right = np.sqrt((top_right_corner[0] - above_right_points['score'])**2 + (top_right_corner[1] - above_right_points['overall_profit_24_rounded'])**2)
        
        # Get the index of the point with the smallest distance to the top right corner
        best_index = distances_to_top_right.idxmin()
    else:
        # Calculate the Euclidean distance to the middle point for each point
        distances_to_middle = np.sqrt((middle_point[0] - best_rows['score'])**2 + (middle_point[1] - best_rows['overall_profit_24_rounded'])**2)
        
        # If there are no points that are both to the right and above the middle,
        # get the index of the point with the smallest distance to the middle point
        best_index = distances_to_middle.idxmin()
    
    return best_index, middle_point