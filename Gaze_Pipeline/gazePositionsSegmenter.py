import os
import re
import json
import copy
import pandas as pd


def contains_text_fragment(text, fragment):
  match = re.search(fragment, text)
  if match:
    return True
  else:
    return False
  

def concatenate_gaze_files(file_NC, attackType):
    dirPath = f'PATH/{file_NC}/Gaze_Positions/{attackType}'
    gazeFiles = []

    for gazeFile in os.listdir(dirPath):
        if gazeFile.endswith('.csv'):
            if contains_text_fragment(gazeFile, 'merged') == False and contains_text_fragment(gazeFile, 'unix_datetime') == False:
                gazeFiles.append(f'{dirPath}/{gazeFile}')
    
    
    df_list = [pd.read_csv(gazeFile) for gazeFile in gazeFiles]
    merged_df = pd.concat(df_list, ignore_index=True)
    merged_df['world_index'] = pd.to_numeric(merged_df['world_index'], errors='coerce')  # Convert to numeric if not already
    merged_df = merged_df.sort_values(by='world_index')

    merged_df.to_csv(f'{dirPath}/{attackType}_merged_output.csv', index=False)


def make_new_directories(File_NCs, attackTypes):
    participant_path = '*** REPLACE WITH YOUR ACTUAL PATH! ***'

    for index, File_NC in enumerate(File_NCs):
        for attackType in attackTypes:
            new_dir = f'{participant_path}/{File_NC}/Gaze_Positions/{attackType}'
            os.makedirs(new_dir, exist_ok=True)

def extract_sequence_number(filename):
    filename = filename.replace(".csv", "")
    fragments = filename.split("_")
    return int(fragments[-1])


def extract_gaze_sequences(file_NC, attackType):
    fixationSequenceDir = f'PATH/{file_NC}/Fixations/Fixation_Sequences/{attackType}'

    for sequenceFile in os.listdir(fixationSequenceDir):
        sequence_df = pd.read_csv(f'{fixationSequenceDir}/{sequenceFile}')
        sequenceNumber = extract_sequence_number(sequenceFile)
        gazeFile = f'PATH/{file_NC}/80_MS/gaze_positions.csv'
        gaze_df = pd.read_csv(gazeFile, dtype={'world_index': 'int32'})

        extracted_rows = []

        for index, row in sequence_df.iterrows():
            start_idx = int(row["start_frame_index"])
            end_idx = int(row["end_frame_index"])

            # Extract rows from the gaze file within this index range
            extracted = gaze_df[(gaze_df["world_index"] >= start_idx) & (gaze_df["world_index"] <= end_idx)]
            extracted_rows.append(extracted)

        result_df = pd.concat(extracted_rows, ignore_index=True)
        Fout = f'PATH/{file_NC}/Gaze_Positions/{attackType}/{file_NC}_{attackType}_gaze_sequence_{sequenceNumber}.csv'
        result_df.to_csv(Fout, index=False)
        sequenceNumber += 1
  

if __name__ == '__main__':

    File_NCs = ['NE_002', 'NE_003', 'NE_004', 'NE_005', 'E_068', 'NE_006', 'NE_007', 'NE_008', 'NE_009', 'NE_010', 'NE_011', 'NE_012', 'NE_013',
                'NE_014', 'NE_015', 'NE_016', 'NE_017', 'NE_018', 'NE_019', 'NE_020', 'NE_022', 'NE_024', 'NE_025', 'E_038', 'NE_027', 'NE_029', 'NE_030', 
                'NE_031', 'NE_033', 'NE_034', 'NE_036', 'NE_037', 'NE_039', 'NE_040', 'NE_042', 'NE_043', 'NE_044', 'NE_045', 'NE_046', 
                'NE_047', 'NE_048', 'NE_051', 'NE_052', 'NE_053', 'NE_054', 'NE_056', 'NE_057', 'NE_058', 'NE_059', 'NE_060', 'NE_061', 'NE_062', 'NE_063', 
                'NE_064', 'NE_065', 'E_067', 'E_069']
        
    attackTypes = ["StyleGAN2", "StyleGAN3", "Diseased", "Textured_Contact", "Contacts_+_Print", "Real_Iris","Printout", "Artificial", "Post_Mortem", 
                    "Synthetic", "Glass_Prosthesis"]

    # # Make new directories for Gaze Positions
    # make_new_directories(File_NCs, attackTypes)

    for file_NC in File_NCs:
        for attackType in attackTypes:
            print(f'Processing -> {file_NC} on attack type -> {attackType}')
            concatenate_gaze_files(file_NC, attackType)
            extract_gaze_sequences(file_NC, attackType)
        print('DONE...\n')