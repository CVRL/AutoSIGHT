import os
import re
import json
import copy
import traces
import pandas as pd
import numpy as np
from datetime import timedelta

#######################################################################################################################
"""
 ██╗   ██╗████████╗██╗██╗     ██╗████████╗██╗███████╗███████╗
 ██║   ██║╚══██╔══╝██║██║     ██║╚══██╔══╝██║██╔════╝██╔════╝
 ██║   ██║   ██║   ██║██║     ██║   ██║   ██║█████╗  ███████╗
 ██║   ██║   ██║   ██║██║     ██║   ██║   ██║██╔══╝  ╚════██║
 ╚██████╔╝   ██║   ██║███████╗██║   ██║   ██║███████╗███████║
  ╚═════╝    ╚═╝   ╚═╝╚══════╝╚═╝   ╚═╝   ╚═╝╚══════╝╚══════╝
                                                             
"""
def contains_text_fragment(text, fragment):
  match = re.search(fragment, text)
  if match:
    return True
  else:
    return False
  
def make_new_directories(participant_path, File_NCs, newPath):
    for index, File_NC in enumerate(File_NCs):
        new_dir = f'{participant_path}/{File_NC}/{newPath}'
        os.makedirs(new_dir, exist_ok=True)

#######################################################################################################################
"""
         ████████╗██╗███╗   ███╗███████╗███████╗████████╗ █████╗ ███╗   ███╗██████╗ 
         ╚══██╔══╝██║████╗ ████║██╔════╝██╔════╝╚══██╔══╝██╔══██╗████╗ ████║██╔══██╗
            ██║   ██║██╔████╔██║█████╗  ███████╗   ██║   ███████║██╔████╔██║██████╔╝
            ██║   ██║██║╚██╔╝██║██╔══╝  ╚════██║   ██║   ██╔══██║██║╚██╔╝██║██╔═══╝ 
            ██║   ██║██║ ╚═╝ ██║███████╗███████║   ██║   ██║  ██║██║ ╚═╝ ██║██║     
            ╚═╝   ╚═╝╚═╝     ╚═╝╚══════╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝     ╚═╝╚═╝     
                                                                                    
                 ███████╗██╗   ██╗███╗   ██╗ ██████╗██╗███╗   ██╗ ██████╗   
                 ██╔════╝╚██╗ ██╔╝████╗  ██║██╔════╝██║████╗  ██║██╔════╝   
                 ███████╗ ╚████╔╝ ██╔██╗ ██║██║     ██║██╔██╗ ██║██║  ███╗  
                 ╚════██║  ╚██╔╝  ██║╚██╗██║██║     ██║██║╚██╗██║██║   ██║  
                 ███████║   ██║   ██║ ╚████║╚██████╗██║██║ ╚████║╚██████╔╝  
                 ╚══════╝   ╚═╝   ╚═╝  ╚═══╝ ╚═════╝╚═╝╚═╝  ╚═══╝ ╚═════╝   
"""
def timestampSyncing(participant_path, File_NCs):
    for file_NC in File_NCs:
        gazePositionsFile = f'{participant_path}/{file_NC}/80_MS/gaze_positions.csv'
        print(f'Processing -> {file_NC}')
        try:
            info_player_file = f'{participant_path}/{file_NC}/80_MS/info.player.json'

            Fin = open(info_player_file)
            jsondata = json.load(Fin)
            Fin.close()

            start_time_system = jsondata["start_time_system_s"]
            start_time_synced = jsondata["start_time_synced_s"]
            start_timestamp_diff = start_time_system - start_time_synced

            gaze_df = pd.read_csv(gazePositionsFile)
            gaze_df["gaze_timestamp_unix"] = gaze_df["gaze_timestamp"] + start_timestamp_diff
            gaze_df["gaze_timestamp_datetime"] = pd.to_datetime(gaze_df["gaze_timestamp_unix"], unit="s")
            gaze_df.to_csv(f'{participant_path}/{file_NC}/80_MS/gaze_positions_unix_datetime.csv', index=False)
        except:
            print(f'Error Processing {file_NC}')

#######################################################################################################################
"""
    ███████╗██╗  ██╗████████╗██████╗  █████╗  ██████╗████████╗                                  
    ██╔════╝╚██╗██╔╝╚══██╔══╝██╔══██╗██╔══██╗██╔════╝╚══██╔══╝                                  
    █████╗   ╚███╔╝    ██║   ██████╔╝███████║██║        ██║                                     
    ██╔══╝   ██╔██╗    ██║   ██╔══██╗██╔══██║██║        ██║                                     
    ███████╗██╔╝ ██╗   ██║   ██║  ██║██║  ██║╚██████╗   ██║                                     
    ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝   ╚═╝                                     
                                                                                                
                 ███████╗███████╗ ██████╗ ██╗   ██╗███████╗███╗   ██╗ ██████╗███████╗███████╗
                 ██╔════╝██╔════╝██╔═══██╗██║   ██║██╔════╝████╗  ██║██╔════╝██╔════╝██╔════╝
                 ███████╗█████╗  ██║   ██║██║   ██║█████╗  ██╔██╗ ██║██║     █████╗  ███████╗
                 ╚════██║██╔══╝  ██║▄▄ ██║██║   ██║██╔══╝  ██║╚██╗██║██║     ██╔══╝  ╚════██║
                 ███████║███████╗╚██████╔╝╚██████╔╝███████╗██║ ╚████║╚██████╗███████╗███████║
                 ╚══════╝╚══════╝ ╚══▀▀═╝  ╚═════╝ ╚══════╝╚═╝  ╚═══╝ ╚═════╝╚══════╝╚══════╝
"""
def extractGazePositionSequences(jsonFolderPath, participant_path, File_NCs):
    for file_NC in File_NCs:
        print(f'Processing -> {file_NC} ...')
        jsonFilePath = f"{jsonFolderPath}/GRI/{file_NC}_comprehensive_2.json"
        with open(jsonFilePath, 'r') as Fin:
            imageJsonData = json.load(Fin)

        gazePositionsFile = f'{participant_path}/{file_NC}/80_MS/gaze_positions_unix_datetime.csv'
        gaze_df = pd.read_csv(gazePositionsFile)
        gaze_df["world_index"] = pd.to_numeric(gaze_df["world_index"], errors='coerce')

        sequences = []

        for jsonObject in imageJsonData:
            startIndex = int(jsonObject["imageStartIndex"])
            endIndex = int(jsonObject["finalDecisionIndex"])

            sequence_df = gaze_df[
                (gaze_df["world_index"] >= startIndex) &
                (gaze_df["world_index"] <= endIndex)
            ]

            sequences.append(sequence_df)

        # Concatenate sequences for this file_NC
        final_df = pd.concat(sequences, ignore_index=True)

        output_path = f"{participant_path}/{file_NC}/Sliding_Window/{file_NC}_concatenated_gaze_sequences.csv"
        final_df.to_csv(output_path, index=False)

#######################################################################################################################
"""
    ███████╗██╗     ██╗██████╗ ██╗███╗   ██╗ ██████╗                  
    ██╔════╝██║     ██║██╔══██╗██║████╗  ██║██╔════╝                  
    ███████╗██║     ██║██║  ██║██║██╔██╗ ██║██║  ███╗                 
    ╚════██║██║     ██║██║  ██║██║██║╚██╗██║██║   ██║                 
    ███████║███████╗██║██████╔╝██║██║ ╚████║╚██████╔╝                 
    ╚══════╝╚══════╝╚═╝╚═════╝ ╚═╝╚═╝  ╚═══╝ ╚═════╝                  
                                                                   
                 ██╗    ██╗██╗███╗   ██╗██████╗  ██████╗ ██╗    ██╗
                 ██║    ██║██║████╗  ██║██╔══██╗██╔═══██╗██║    ██║
                 ██║ █╗ ██║██║██╔██╗ ██║██║  ██║██║   ██║██║ █╗ ██║
                 ██║███╗██║██║██║╚██╗██║██║  ██║██║   ██║██║███╗██║
                 ╚███╔███╔╝██║██║ ╚████║██████╔╝╚██████╔╝╚███╔███╔╝
                  ╚══╝╚══╝ ╚═╝╚═╝  ╚═══╝╚═════╝  ╚═════╝  ╚══╝╚══╝ 
"""
def recalculate_elapsed_time(sequence, gap_threshold=100):
    fragments = []
    new_fragment = []
    RET = 0                     # RET = Recalculated Elapsed Time
    prev_world_index = None

    for index, item in enumerate(sequence):
        if len(new_fragment) == 0:
            new_fragment.append(item)
            prev_world_index = item["world_index"]
        else:
            difference = item["world_index"] - prev_world_index
            if difference < gap_threshold:
                new_fragment.append(item)
                prev_world_index = item["world_index"]
            else:
                fragments.append(copy.deepcopy(new_fragment))
                new_fragment = []
                new_fragment.append(item)
                prev_world_index = item["world_index"]

    fragments.append(new_fragment)

    for fragment in fragments:
        RET += (fragment[-1]["gaze_timestamp_unix"] - fragment[0]["gaze_timestamp_unix"])

    return RET

def recalculate_elapsed_time_speedup(sequence, gap_threshold=100):
    if not sequence:
        return 0

    RET = 0
    start_time = sequence[0]["gaze_timestamp_unix"]
    prev_world_index = sequence[0]["world_index"]

    for i in range(1, len(sequence)):
        curr_item = sequence[i]
        diff = curr_item["world_index"] - prev_world_index

        if diff >= gap_threshold:
            # End of a fragment, add its elapsed time
            RET += sequence[i - 1]["gaze_timestamp_unix"] - start_time
            start_time = curr_item["gaze_timestamp_unix"]

        prev_world_index = curr_item["world_index"]

    # Add the last fragment
    RET += sequence[-1]["gaze_timestamp_unix"] - start_time

    return RET

#######################################################################################################################
"""
    The final sequence is essentially never an even window size, the sliding
    window will advance (window_size // 2) seconds each time. On the final sequence, 
    we just reverse the dataframe and append a few extra rows until we have an evenly 
    spaced (window_size) second window of data.
"""
def pad_final_sequence(final_sequence, SST, fix_df, window_size=10):
    original_elapsed_time = recalculate_elapsed_time(final_sequence)
    start_index = fix_df.index[fix_df["gaze_timestamp_unix"] == SST].tolist()[0]
    reversed_df = fix_df.iloc[:start_index+1].iloc[::-1]

    new_sequence = []
    new_SST = None
    NS_Duration = None

    for i, row in reversed_df.iterrows():
        row_dict = row.to_dict()

        if new_SST is None:
            new_SST = row_dict["gaze_timestamp_unix"]
        else:
            new_sequence.append(row_dict)
            NS_Duration = recalculate_elapsed_time(new_sequence)

            ## Since Dataframe is now reversed, elapsed time will be negative so need to flip the sign
            NS_Duration = abs(NS_Duration)
            new_elapsed_time = original_elapsed_time + NS_Duration

            if new_elapsed_time > window_size:
                new_sequence.extend(final_sequence)
                sorted_new_sequence = sorted(new_sequence, key=lambda x: x["gaze_timestamp_unix"])
                return sorted_new_sequence

#######################################################################################################################
def segmentGazeSequences_by_SlidingWindowSize(participant_path, File_NCs, windowSize, output_folder):
    for file_NC in File_NCs:
        print(f'Processing -> {file_NC}')
        window_size = windowSize
        step_size = window_size // 2
        gap_threshold = 100
        sequence_count = 1

        sequences = []
        current_sequence = []
        sequence_start_time = None
        current_row_world_index = None
        previous_row_world_index = None
        gap_sequence = False

        # Load CSV file
        df = pd.read_csv(f"{participant_path}/{file_NC}/Sliding_Window/{file_NC}_concatenated_gaze_sequences.csv")
        df["gaze_timestamp_unix"] = df["gaze_timestamp_unix"].astype(float)

        # Ensure timestamps are sorted
        df = df.sort_values(by="gaze_timestamp_unix")

        # Iterate through the data
        for i, row in df.iterrows():
            timestamp = row["gaze_timestamp_unix"]
            current_row_world_index = row["world_index"]

            if previous_row_world_index is None:
                previous_row_world_index = row["world_index"]

            row_dict = row.to_dict()

            if sequence_start_time is None:
                sequence_start_time = timestamp

            elapsed_time = timestamp - sequence_start_time
            world_index_diff = current_row_world_index - previous_row_world_index


            if world_index_diff > gap_threshold or gap_sequence == True:
                elapsed_time = recalculate_elapsed_time(current_sequence)
                gap_sequence = True


            if elapsed_time <= window_size:
                # Append to current sequence
                current_sequence.append(row_dict)
                previous_row_world_index = current_row_world_index
                
            else:
                # Store completed sequence and start new one
                sequences.append(current_sequence)

                sequence_count += 1
                sequence_start_time += step_size

                # Remove old points outside new window
                current_sequence = [r for r in current_sequence if r["gaze_timestamp_unix"] >= sequence_start_time]
                current_sequence.append(row_dict)
                sequence_start_time = current_sequence[0]["gaze_timestamp_unix"]
                elapsed_time = recalculate_elapsed_time(current_sequence)
                gap_sequence = False

        # Add the last sequence
        if current_sequence:
            final_elapsed_time = recalculate_elapsed_time(current_sequence)

            if final_elapsed_time <= window_size:
                new_final_sequence = pad_final_sequence(current_sequence, sequence_start_time, df, window_size)
                current_sequence = new_final_sequence
            sequences.append(current_sequence)

        # Write sequences to CSV file with blank rows in between
        output_dir = f"PATH/{file_NC}/Sliding_Window/{output_folder}"

        for index, sequence in enumerate(sequences):
            df = pd.DataFrame(sequence, columns=["world_index", "norm_pos_x", "norm_pos_y", "gaze_timestamp_unix", "gaze_timestamp_datetime"])
            df.to_csv(f'{output_dir}/{file_NC}_gaze_sequence_{index+1}.csv', index=False)
         
        print(f'Pushing {len(sequences)} Sequences for Participant -> {file_NC}')


#####################################################################################################################################################
"""
        ██████╗ ███████╗███████╗ █████╗ ███╗   ███╗██████╗ ██╗     ███████╗
        ██╔══██╗██╔════╝██╔════╝██╔══██╗████╗ ████║██╔══██╗██║     ██╔════╝
        ██████╔╝█████╗  ███████╗███████║██╔████╔██║██████╔╝██║     █████╗  
        ██╔══██╗██╔══╝  ╚════██║██╔══██║██║╚██╔╝██║██╔═══╝ ██║     ██╔══╝  
        ██║  ██║███████╗███████║██║  ██║██║ ╚═╝ ██║██║     ███████╗███████╗
        ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝     ╚══════╝╚══════╝
                                                                            
                 ████████╗██████╗  █████╗  ██████╗███████╗███████╗  
                 ╚══██╔══╝██╔══██╗██╔══██╗██╔════╝██╔════╝██╔════╝  
                    ██║   ██████╔╝███████║██║     █████╗  ███████╗  
                    ██║   ██╔══██╗██╔══██║██║     ██╔══╝  ╚════██║  
                    ██║   ██║  ██║██║  ██║╚██████╗███████╗███████║  
                    ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚══════╝╚══════╝  
"""
def resample_sequence_traces(df_seq, target_samples, windowSize):
    """
        Resamples a sequence using the `traces` library, interpolating X and Y coordinates separately.
    """
    df_seq = df_seq.copy()

    # Convert timestamps to datetime (ensures correct arithmetic)
    df_seq['gaze_timestamp_datetime'] = pd.to_datetime(df_seq['gaze_timestamp_datetime'])

    # Ensure timestamps are sorted
    df_seq = df_seq.sort_values(by='gaze_timestamp_datetime')

    # Convert timestamps to seconds relative to the first timestamp
    df_seq['time_seconds'] = (df_seq['gaze_timestamp_datetime'] - df_seq['gaze_timestamp_datetime'].min()).dt.total_seconds()

    # Create traces TimeSeries for X and Y
    ts_x = traces.TimeSeries(dict(zip(df_seq['time_seconds'], df_seq['norm_pos_x'])))
    ts_y = traces.TimeSeries(dict(zip(df_seq['time_seconds'], df_seq['norm_pos_y'])))

    # Define new uniform time grid using numpy
    new_time_index = np.linspace(0, df_seq['time_seconds'].max(), target_samples)

    # Interpolate values at new time points using sample()
    x_resampled = ts_x.sample(sampling_period=timedelta(seconds=0.005), start=0, end=windowSize, interpolate='linear')
    y_resampled = ts_y.sample(sampling_period=timedelta(seconds=0.005), start=0, end=windowSize, interpolate='linear')


    # Convert interpolated time points back to datetime format
    df_resampled = pd.DataFrame({
        'x': [i[1] for i in x_resampled],
        'y': [j[1] for j in y_resampled]
    })

    return df_resampled


def extract_coordinate_sequence_data(participant_path, File_NCs, windowSize):
    for file_NC in File_NCs:
        dirPath = f'{participant_path}/{file_NC}/Sliding_Window/{windowSize}'
        frags = windowSize.split("_")

        windowTimeFrame = int(frags[0])
        sampleSize = 200 * windowTimeFrame

        print(f'Processing {file_NC} for sample size -> {sampleSize}')

        for filename in os.listdir(dirPath):
            if filename.endswith(".csv"):
                df = pd.read_csv(f'{dirPath}/{filename}')
                df_resampled = resample_sequence_traces(df, target_samples=sampleSize, windowSize=windowTimeFrame)

                filename = filename.replace(".csv", "")
                fragments = filename.split("_")
                sequence_number = int(fragments[-1])

                Fout = f'{dirPath}/Resampled/{file_NC}_resampled_sequence_{sequence_number}.csv'
                df_resampled.to_csv(Fout, index=False)

        print(f"Resampling complete for -> {file_NC} -> All sequences have exactly {sampleSize} samples.")
    print('\n\n')


#################################################################################################################################
"""
 ███████╗██╗  ██╗████████╗██████╗  █████╗  ██████╗████████╗                        
 ██╔════╝╚██╗██╔╝╚══██╔══╝██╔══██╗██╔══██╗██╔════╝╚══██╔══╝                        
 █████╗   ╚███╔╝    ██║   ██████╔╝███████║██║        ██║                           
 ██╔══╝   ██╔██╗    ██║   ██╔══██╗██╔══██║██║        ██║                           
 ███████╗██╔╝ ██╗   ██║   ██║  ██║██║  ██║╚██████╗   ██║                           
 ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝   ╚═╝                           
                                                                                   
                 ███████╗██╗██╗  ██╗ █████╗ ████████╗██╗ ██████╗ ███╗   ██╗███████╗
                 ██╔════╝██║╚██╗██╔╝██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║██╔════╝
                 █████╗  ██║ ╚███╔╝ ███████║   ██║   ██║██║   ██║██╔██╗ ██║███████╗
                 ██╔══╝  ██║ ██╔██╗ ██╔══██║   ██║   ██║██║   ██║██║╚██╗██║╚════██║
                 ██║     ██║██╔╝ ██╗██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║███████║
                 ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝
"""
def get_matching_rows_vectorized(fix_df, sequence_indexes):
    # Create a boolean mask initialized to all False
    mask = pd.Series(False, index=fix_df.index)

    # Loop through sequence indexes and update the mask (still faster than apply)
    for val in sequence_indexes:
        mask |= (fix_df['start_frame_index'] <= val) & (val <= fix_df['end_frame_index'])

    return fix_df[mask]


def extrapolateFixationStatsfromGazeSequences(participant_path, File_NCs, windowSize):
    print(f'\n\nWindow Size: {windowSize}\n')
    for file_NC in File_NCs:
        print(f'Processing -> {file_NC}')
        sequence_path = f'{participant_path}/{file_NC}/Sliding_Window/{windowSize}'
        fixationFile = f'{participant_path}/{file_NC}/80_MS/fixations.csv'

        fix_df = pd.read_csv(fixationFile)

        for filename in os.listdir(sequence_path):
            if filename.endswith('.csv'):
                fullPath = f'{sequence_path}/{filename}'
                sw_df = pd.read_csv(fullPath)
                sw_df["world_index"] = sw_df["world_index"].astype(int)
                sequenceIndexes = sw_df["world_index"].unique()
                sequenceIndexes = pd.to_numeric(sequenceIndexes, errors='coerce')

                filename = filename.replace(".csv", "")
                fragments = filename.split("_")
                sequence_number = int(fragments[-1])

                matching_rows = get_matching_rows_vectorized(fix_df, sequenceIndexes)
                outputDir = f'PATH/{file_NC}/Fixations/Sliding_Window/{windowSize}'
                outputFileName = f'{file_NC}_fixation_sequence_{sequence_number}.csv'
                outputPath = f'{outputDir}/{outputFileName}'

                matching_rows.to_csv(outputPath, index=False)


#################################################################################################################################
"""                                             
                                                          
 ███╗   ███╗ █████╗ ██╗███╗   ██╗                         
 ████╗ ████║██╔══██╗██║████╗  ██║                         
 ██╔████╔██║███████║██║██╔██╗ ██║                         
 ██║╚██╔╝██║██╔══██║██║██║╚██╗██║                         
 ██║ ╚═╝ ██║██║  ██║██║██║ ╚████║                         
 ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝                         
                                                          
                 ██████╗ ██╗      ██████╗  ██████╗██╗  ██╗
                 ██╔══██╗██║     ██╔═══██╗██╔════╝██║ ██╔╝
                 ██████╔╝██║     ██║   ██║██║     █████╔╝ 
                 ██╔══██╗██║     ██║   ██║██║     ██╔═██╗ 
                 ██████╔╝███████╗╚██████╔╝╚██████╗██║  ██╗
                 ╚═════╝ ╚══════╝ ╚═════╝  ╚═════╝╚═╝  ╚═╝
"""
if __name__ == '__main__':

    File_NCs = ['NE_002', 'NE_003', 'NE_004', 'NE_005', 'E_068', 'NE_006', 'NE_007', 'NE_008', 'NE_009', 'NE_010', 'NE_011', 'NE_012', 'NE_013',
                'NE_014', 'NE_015', 'NE_016', 'NE_017', 'NE_018', 'NE_019', 'NE_020', 'NE_022', 'NE_024', 'NE_025', 'E_038', 'NE_027', 'NE_029', 'NE_030', 
                'NE_031', 'NE_033', 'NE_034', 'NE_036', 'NE_037', 'NE_039', 'NE_040', 'NE_042', 'NE_043', 'NE_044', 'NE_045', 'NE_046', 
                'NE_047', 'NE_048', 'NE_051', 'NE_052', 'NE_053', 'NE_054', 'NE_056', 'NE_057', 'NE_058', 'NE_059', 'NE_060', 'NE_061', 'NE_062', 'NE_063', 
                'NE_064', 'NE_065', 'E_067', 'E_069', 'E_072']
        
    attackTypes = ["StyleGAN2", "StyleGAN3", "Diseased", "Textured_Contact", "Contacts_+_Print", "Real_Iris","Printout", "Artificial", "Post_Mortem", 
                    "Synthetic", "Glass_Prosthesis"]
    
    participant_path = '*** REPLACE WITH YOUR ACTUAL PATH! ***'
    jsonFolderPath = '*** REPLACE WITH YOUR ACTUAL PATH! ***'



    # make_new_directories(participant_path, File_NCs, 'Sliding_Window/5_Second/Resampled')
    # timestampSyncing(participant_path, File_NCs)
    # extractGazePositionSequences(jsonFolderPath, participant_path, File_NCs)    

    # segmentGazeSequences_by_SlidingWindowSize(participant_path, File_NCs, windowSize=30, output_folder='30_Second')
    # segmentGazeSequences_by_SlidingWindowSize(participant_path, File_NCs, windowSize=20, output_folder='20_Second')
    # segmentGazeSequences_by_SlidingWindowSize(participant_path, File_NCs, windowSize=15, output_folder='15_Second')
    # segmentGazeSequences_by_SlidingWindowSize(participant_path, File_NCs, windowSize=10, output_folder='10_Second')
    # segmentGazeSequences_by_SlidingWindowSize(participant_path, File_NCs, windowSize=5, output_folder='5_Second')

    # extract_coordinate_sequence_data(participant_path, File_NCs, '30_Second')
    # extract_coordinate_sequence_data(participant_path, File_NCs, '20_Second')
    # extract_coordinate_sequence_data(participant_path, File_NCs, '15_Second')
    # extract_coordinate_sequence_data(participant_path, File_NCs, '10_Second')
    # extract_coordinate_sequence_data(participant_path, File_NCs, '5_Second')


    # extrapolateFixationStatsfromGazeSequences(participant_path, File_NCs, '5_Second')
    # extrapolateFixationStatsfromGazeSequences(participant_path, File_NCs, '10_Second')
    # extrapolateFixationStatsfromGazeSequences(participant_path, File_NCs, '15_Second')
    # extrapolateFixationStatsfromGazeSequences(participant_path, File_NCs, '20_Second')
    # extrapolateFixationStatsfromGazeSequences(participant_path, File_NCs, '30_Second')