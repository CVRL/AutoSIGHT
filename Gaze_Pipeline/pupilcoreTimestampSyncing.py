import os
import re
import datetime
import json
import numpy as np
import pandas as pd


def contains_text_fragment(text, fragment):
  match = re.search(fragment, text)
  if match:
    return True
  else:
    return False
  

if __name__ == '__main__':

    File_NCs = ['NE_002', 'NE_003', 'NE_004', 'NE_005', 'E_068', 'NE_006', 'NE_007', 'NE_008', 'NE_009', 'NE_010', 'NE_011', 'NE_012', 'NE_013',
                'NE_014', 'NE_015', 'NE_016', 'NE_017', 'NE_018', 'NE_019', 'NE_020', 'NE_022', 'NE_024', 'NE_025', 'E_038', 'NE_027', 'NE_029', 'NE_030', 
                'NE_031', 'NE_033', 'NE_034', 'NE_036', 'NE_037', 'NE_039', 'NE_040', 'NE_042', 'NE_043', 'NE_044', 'NE_045', 'NE_046', 
                'NE_047', 'NE_048', 'NE_051', 'NE_052', 'NE_053', 'NE_054', 'NE_056', 'NE_057', 'NE_058', 'NE_059', 'NE_060', 'NE_061', 'NE_062', 'NE_063', 
                'NE_064', 'NE_065', 'E_067', 'E_069']

        
    attackTypes = ["StyleGAN2", "StyleGAN3", "Diseased", "Textured_Contact", "Contacts_+_Print", "Real_Iris","Printout", "Artificial", "Post_Mortem", 
                    "Synthetic", "Glass_Prosthesis"]

    participant_path = '*** REPLACE WITH YOUR ACTUAL PATH! ***'

    for file_NC in File_NCs:
        for attackType in attackTypes:
            currentDirectory = f'{participant_path}/{file_NC}/Gaze_Positions/{attackType}'
            print(f'Processing -> {file_NC} on {attackType}')
            for filename in os.listdir(currentDirectory):
                if contains_text_fragment(filename, '_merged_') == False:
                    print(f'Processing -> {filename}')
                    try:
                        info_player_file = f'PATH/{file_NC}/80_MS/info.player.json'
                        gaze_sequence_file = f'{currentDirectory}/{filename}'

                        stripped_filename = filename.replace(".csv", "")

                        Fin = open(info_player_file)
                        jsondata = json.load(Fin)
                        Fin.close()

                        start_time_system = jsondata["start_time_system_s"]
                        start_time_synced = jsondata["start_time_synced_s"]
                        start_timestamp_diff = start_time_system - start_time_synced

                        gaze_df = pd.read_csv(gaze_sequence_file)
                        gaze_df["gaze_timestamp_unix"] = gaze_df["gaze_timestamp"] + start_timestamp_diff
                        gaze_df["gaze_timestamp_datetime"] = pd.to_datetime(gaze_df["gaze_timestamp_unix"], unit="s")
                        gaze_df.to_csv(f'{currentDirectory}/{stripped_filename}_unix_datetime.csv', index=False)
                    except:
                       print(f'Error Processing {filename}')