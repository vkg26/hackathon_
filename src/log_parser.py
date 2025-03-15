import pandas as pd
import re


def parse_log_file(log_file_path):
    with open(log_file_path, 'r') as file:
        logs = file.readlines()

    log_data = []
    total_cost = 0
    total_time_taken = 0
    agent_start_times = {}

    for log in logs:
        timestamp_match = re.search(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', log)
        if timestamp_match:
            timestamp = timestamp_match.group(0)
            log_level = log.split(' - ')[2]
            message = log.split(' - ')[-1].strip()

            # Capture agent start times
            start_match = re.search(r'^(.*Agent|.*Action) started', message)
            if start_match:
                agent = start_match.group(1)
                agent_start_times[agent] = timestamp

            agent_match = re.search(r'^(.*Agent|.*Action)', message)
            cost_match = re.search(r'Cost: ([\d\.e\-]+) dollars', message)
            time_taken_match = re.search(r'Time Taken: ([\d\.e\-]+) seconds', message)

            if agent_match and cost_match and time_taken_match:
                agent = agent_match.group(1)
                cost = float(cost_match.group(1))
                time_taken = float(time_taken_match.group(1))
                total_cost += cost
                total_time_taken += time_taken

                log_data.append({
                    'Timestamp': timestamp,
                    'Agent': agent,
                    'Cost ($)': cost,
                    'Time Taken (s)': time_taken,
                    'Start Time': agent_start_times.get(agent, 'N/A')
                })
            elif 'Total Time Taken' in message:
                total_time = float(re.search(r'Total Time Taken: ([\d\.]+) seconds', message).group(1))
                log_data.append({
                    'Timestamp': timestamp,
                    'Agent': 'Total',
                    'Cost ($)': total_cost,
                    'Time Taken (s)': total_time,
                    'Start Time': 'N/A'
                })

    return log_data


def create_log_table(log_data):
    df = pd.DataFrame(log_data)
    return df


def display_log_table(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)


def save_log_table(df, file_path):
    df.to_excel(file_path, index=False)


log_file_path = '../app_logs.log'  # Adjust the path as necessary
log_data = parse_log_file(log_file_path)
log_df = create_log_table(log_data)
display_log_table(log_df)

# Optionally save the table to a file
output_file_path = '../app_logs_summary.xlsx'  # Adjust the path as necessary
save_log_table(log_df, output_file_path)
