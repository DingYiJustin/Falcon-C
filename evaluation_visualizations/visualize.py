import pandas as pd

# def print_column_means(df):    
#     # Exclude the 'episode_id' column
#     columns_to_consider = df.drop(columns=['episode_id'], errors='ignore')
    
#     # Calculate the mean for each column
#     means = columns_to_consider.mean()
    
#     # Print the means
#     print(means)

def print_column_means_and_percentage(df):
    # Exclude the 'episode_id' column
    columns_to_consider = df.drop(columns=['episode_id'], errors='ignore')
    
    # Calculate the mean for each column
    means = columns_to_consider.mean()
    
    # Print the means
    print("Means of each column (excluding 'episode_id'):")
    print(means)
    
    # Calculate the percentage of 'distance_to_goal' lower than 0.2
    if 'distance_to_goal' in df.columns:
        lower_than_0_2 = (df['distance_to_goal'] < 0.2).sum()
        # Print the percentage
        total_count = df['distance_to_goal'].count()
        percentage = (lower_than_0_2 / total_count) * 100 if total_count > 0 else 0
        print(f"\n'distance_to_goal' values lower than 0.2: {lower_than_0_2:.2f} {percentage:.6f}")
    else:
        print("Column 'distance_to_goal' not found in the DataFrame.")
        
    # Calculate the percentage of 'distance_to_goal' lower than 0.2
    if 'success' in df.columns:
        success = (df['success'] == 1).sum()
        # Print the percentage
        total_count = df['success'].count()
        percentage = (success / total_count) * 100 if total_count > 0 else 0
        print(f"\n'success' values == 1: {success:.2f} {percentage:.6f}")
    else:
        print("Column 'success' not found in the DataFrame.")


def compare_episode_data(df1, df2):
    """
    Compare data from two DataFrames based on common episode_ids.

    Parameters:
    df1 (pd.DataFrame): The first DataFrame containing episode_id.
    df2 (pd.DataFrame): The second DataFrame containing episode_id.

    Returns:
    None
    """
    # Ensure 'episode_id' is in both DataFrames
    if 'episode_id' not in df1.columns or 'episode_id' not in df2.columns:
        print("One of the DataFrames does not contain 'episode_id'.")
        return
    
    # Get the common episode_ids
    common_episode_ids = df1['episode_id'].isin(df2['episode_id'])
    
    # Extract rows with common episode_ids from both DataFrames
    df1_common = df1[common_episode_ids]
    df2_common = df2[df2['episode_id'].isin(df1['episode_id'])]
    
    # Print the extracted data
    print("Data from df1 with common episode_ids:")
    print(df1_common)
    
    print("\nData from df2 with common episode_ids:")
    print(df2_common)
    
    # Compare the results (example: checking if the number of rows is the same)
    if df1_common.shape[0] == df2_common.shape[0]:
        print("\nBoth DataFrames have the same number of common episode_ids.")
    else:
        print(f"\nNumber of common episode_ids in df1: {df1_common.shape[0]}")
        print(f"Number of common episode_ids in df2: {df2_common.shape[0]}")
    
    print('df1 common:')
    print_column_means_and_percentage(df1_common)
    print('df2 common:')
    print_column_means_and_percentage(df2_common)

def filter_unsuccessful_episodes(input_csv, output_csv, success_column='success'):
    """
    Filter episodes with distance_to_goal < 0.2 and not successful, and save to a CSV file.

    Parameters:
    input_csv (str): Path to the input CSV file.
    output_csv (str): Path to the output CSV file.
    success_column (str): The column name that indicates success (default is 'success').

    Returns:
    None
    """
    # Load the input CSV file into a DataFrame
    df = pd.read_csv(input_csv)
    
    # Ensure the necessary columns are present
    if 'distance_to_goal' not in df.columns or success_column not in df.columns:
        print("Input DataFrame does not contain the necessary columns.")
        return
    
    # Filter for episodes with distance_to_goal < 0.2 and not successful
    filtered_df = df[(df['distance_to_goal'] < 0.2) & (df[success_column] != 1)]
    
    # Save the filtered DataFrame to the output CSV file
    filtered_df.to_csv(output_csv, index=False)
    
    print(f"Filtered episodes saved to {output_csv}")


def output_failed_episodes_to_csv(dataframe, csv_filename):
    """
    This function filters episodes where true_success is not met and outputs them to a CSV file.

    Parameters:
    dataframe (pd.DataFrame): The DataFrame containing episode data with a 'true_success' column.
    csv_filename (str): The name of the output CSV file.

    Returns:
    None
    """
    # Filter episodes where true_success is not met
    failed_episodes = dataframe[dataframe['true_success'] != True]

    # Output to CSV
    failed_episodes.to_csv(csv_filename, index=False)

df = pd.read_csv('./evaluation dtgcf_hmap_self_stop_fast_a40 hm3d checkpoints ckpt.26.pth.csv')
# print_column_means(df)
print_column_means_and_percentage(df)

df2 = pd.read_csv('evaluation dtgcf_hmap_self_stop_fast_a40 hm3d checkpoints ckpt.41.pth.csv')#'./pretrained_model falcon_noaux_25.pth.csv')
compare_episode_data(df,df2)

# filter_unsuccessful_episodes('./evaluation falcon_hmap_1 hm3d checkpoints ckpt.10-1.pth.csv', 'to_get_video.csv')
output_failed_episodes_to_csv(df2, 'to_get_video.csv')