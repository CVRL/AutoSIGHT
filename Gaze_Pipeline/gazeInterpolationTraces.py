import os
import re
import traces
import pandas as pd
import numpy as np
from datetime import timedelta


def contains_text_fragment(text, fragment):
  match = re.search(fragment, text)
  if match:
    return True
  else:
    return False
  

def make_new_directories(File_NCs, attackTypes):
    participant_path = '*** REPLACE WITH YOUR ACTUAL PATH! ***'

    for index, File_NC in enumerate(File_NCs):
        for attackType in attackTypes:
            new_dir = f'{participant_path}/{File_NC}/Gaze_Positions/{attackType}/Resampled'
            os.makedirs(new_dir, exist_ok=True)


def resample_sequence_traces(df_seq, target_samples=2000):
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
    x_resampled = ts_x.sample(sampling_period=timedelta(seconds=0.005), start=0, end=10, interpolate='linear')
    y_resampled = ts_y.sample(sampling_period=timedelta(seconds=0.005), start=0, end=10, interpolate='linear')


    # Convert interpolated time points back to datetime format
    df_resampled = pd.DataFrame({
        'x': [i[1] for i in x_resampled],
        'y': [j[1] for j in y_resampled]
    })

    return df_resampled



if __name__ == '__main__':

    File_NCs = ['NE_002', 'NE_003', 'NE_004', 'NE_005', 'E_068', 'NE_006', 'NE_007', 'NE_008', 'NE_009', 'NE_010', 'NE_011', 'NE_012', 'NE_013',
                'NE_014', 'NE_015', 'NE_016', 'NE_017', 'NE_018', 'NE_019', 'NE_020', 'NE_022', 'NE_024', 'NE_025', 'E_038', 'NE_027', 'NE_029', 'NE_030', 
                'NE_031', 'NE_033', 'NE_034', 'NE_036', 'NE_037', 'NE_039', 'NE_040', 'NE_042', 'NE_043', 'NE_044', 'NE_045', 'NE_046', 
                'NE_047', 'NE_048', 'NE_051', 'NE_052', 'NE_053', 'NE_054', 'NE_056', 'NE_057', 'NE_058', 'NE_059', 'NE_060', 'NE_061', 'NE_062', 'NE_063', 
                'NE_064', 'NE_065', 'E_067', 'E_069']

        
    attackTypes = ["StyleGAN2", "StyleGAN3", "Diseased", "Textured_Contact", "Contacts_+_Print", "Real_Iris","Printout", "Artificial", "Post_Mortem", 
                    "Synthetic", "Glass_Prosthesis"]
    

    # Make a new directory for resampled results
    make_new_directories(File_NCs, attackTypes)


    for file_NC in File_NCs:
        for attackType in attackTypes:
            dirPath = f'PATH/{file_NC}/Gaze_Positions/{attackType}'

            for filename in os.listdir(dirPath):
                if filename.endswith(".csv"):
                    if contains_text_fragment(filename, '_unix_datetime') == True:
                        df = pd.read_csv(f'{dirPath}/{filename}')
                        df_resampled = resample_sequence_traces(df)

                        filename = filename.replace(".csv", "")
                        fragments = filename.split("_")
                        sequence_number = int(fragments[-3])

                        Fout = f'{dirPath}/Resampled/{file_NC}_{attackType}_resampled_sequence_{sequence_number}.csv'
                        df_resampled.to_csv(Fout, index=False)

                        print(f"Resampling complete! All sequences for {file_NC} -> {attackType}  have exactly 2000 samples.")
            print()
        print('\n\n')
