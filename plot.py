
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd

def create_plots(df_params):
    plt.clf()

    # Create a scatter plot of percentile vs profit_margin
    scatter = plt.scatter(df_params['percentile'], df_params['profit_margin'], 
                        c=df_params['overall_profit_24'],  # Color by overall_profit_24
                        s=max(df_params['total_hours_24']) - df_params['total_hours_24'],     # Size inversely proportional to total_hours_24
                        cmap='viridis', alpha=0.6)

    # Add a colorbar
    plt.colorbar(scatter, label='Overall Profit 24')

    # Set the x and y labels
    plt.xlabel('Percentile')
    plt.ylabel('Profit Margin')

    plt.savefig("scatter_plot_24.png")

    ''' LINE PLOT '''

    plt.clf()

    # Initialize a MinMaxScaler
    scaler = MinMaxScaler()

    # Normalize total_hours_24 and overall_profit_24
    df_params[['total_hours_24', 'overall_profit_24']] = scaler.fit_transform(df_params[['total_hours_24', 'overall_profit_24']])

    # Create a line plot of percentile vs total_hours_24 and overall_profit_24
    plt.plot(df_params['percentile'], df_params['total_hours_24'], label='Total Hours 24')
    plt.plot(df_params['percentile'], df_params['overall_profit_24'], label='Overall Profit 24')

    # Add a legend
    plt.legend()

    # Set the x and y labels
    plt.xlabel('Percentile')
    plt.ylabel('Normalized Value')

    plt.savefig("line_plot_24.png")

    ''' SCATTER PLOT FOR 72 HOURS '''

    plt.clf()

    # Create a scatter plot of percentile vs profit_margin
    scatter = plt.scatter(df_params['percentile'], df_params['profit_margin'], 
                        c=df_params['overall_profit_72'],  # Color by overall_profit_72
                        s=max(df_params['total_hours_72']) - df_params['total_hours_72'],     # Size inversely proportional to total_hours_72
                        cmap='viridis', alpha=0.6)

    # Add a colorbar
    plt.colorbar(scatter, label='Overall Profit 72')

    # Set the x and y labels
    plt.xlabel('Percentile')
    plt.ylabel('Profit Margin')

    plt.savefig("scatter_plot_72.png")

    ''' LINE PLOT FOR 72 HOURS '''

    plt.clf()

    # Normalize total_hours_72 and overall_profit_72
    df_params[['total_hours_72', 'overall_profit_72']] = scaler.fit_transform(df_params[['total_hours_72', 'overall_profit_72']])

    # Create a line plot of percentile vs total_hours_72 and overall_profit_72
    plt.plot(df_params['percentile'], df_params['total_hours_72'], label='Total Hours 72')
    plt.plot(df_params['percentile'], df_params['overall_profit_72'], label='Overall Profit 72')

    # Add a legend
    plt.legend()

    # Set the x and y labels
    plt.xlabel('Percentile')
    plt.ylabel('Normalized Value')

    plt.savefig("line_plot_72.png")

    plt.clf()

    '''scatter_plot_open_positions_max_open_positions'''

    # Create a scatter plot of open_positions and max_open_positions vs overall_profit_24
    scatter = plt.scatter(df_params['open_positions'] + df_params['max_open_positions'], df_params['overall_profit_24'], 
                        c=df_params['percentile'],  # Color by percentile
                        s=max(df_params['profit_margin']) - df_params['profit_margin'],     # Size inversely proportional to profit_margin
                        cmap='viridis', alpha=0.6)

    # Add a colorbar
    plt.colorbar(scatter, label='Percentile')

    # Set the x and y labels
    plt.xlabel('Open Positions + Max Open Positions')
    plt.ylabel('Overall Profit 24')

    plt.savefig("scatter_plot_open_positions_max_open_positions.png")

    '''scatter_plot_open_positions'''
    plt.clf()

    # Create a scatter plot of open_positions vs overall_profit_72
    scatter = plt.scatter(df_params['open_positions'], df_params['overall_profit_72'], 
                        c=df_params['percentile'],  # Color by percentile
                        s=max(df_params['profit_margin']) - df_params['profit_margin'],     # Size inversely proportional to profit_margin
                        cmap='viridis', alpha=0.6)

    # Add a colorbar
    plt.colorbar(scatter, label='Percentile')

    # Set the x and y labels
    plt.xlabel('Open Positions')
    plt.ylabel('Overall Profit 72')

    plt.savefig("scatter_plot_open_positions.png")

    plt.clf()

    '''scatter_plot_max_open_positions'''

    # Create a scatter plot of max_open_positions vs overall_profit_72
    scatter = plt.scatter(df_params['max_open_positions'], df_params['overall_profit_72'], 
                        c=df_params['percentile'],  # Color by percentile
                        s=max(df_params['profit_margin']) - df_params['profit_margin'],     # Size inversely proportional to profit_margin
                        cmap='viridis', alpha=0.6)

    # Add a colorbar
    plt.colorbar(scatter, label='Percentile')

    # Set the x and y labels
    plt.xlabel('Max Open Positions')
    plt.ylabel('Overall Profit 72')

    plt.savefig("scatter_plot_max_open_positions.png")


if __name__ == "__main__":
    df_params = pd.read_pickle("df_params.pkl")
    create_plots(df_params)