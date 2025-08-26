import pandas as pd
import copy
import re
import os

def make_new_directories(File_NCs, attackTypes):
    participant_path = '/home/bdowlin2/Documents/Participants'

    for index, File_NC in enumerate(File_NCs):
        for attackType in attackTypes:
            new_dir = f'{participant_path}/{File_NC}/Fixations/Fixation_Sequences/{attackType}'
            os.makedirs(new_dir, exist_ok=True)


if __name__ == '__main__':

    File_NCs = ['NE_002', 'NE_003', 'NE_004', 'NE_005', 'E_068', 'NE_006', 'NE_007', 'NE_008', 'NE_009', 'NE_010', 'NE_011', 'NE_012', 'NE_013',
                'NE_014', 'NE_015', 'NE_016', 'NE_017', 'NE_018', 'NE_019', 'NE_020', 'NE_022', 'NE_024', 'NE_025', 'E_038', 'NE_027', 'NE_029', 'NE_030', 
                'NE_031', 'NE_033', 'NE_034', 'NE_036', 'NE_037', 'NE_039', 'NE_040', 'NE_042', 'NE_043', 'NE_044', 'NE_045', 'NE_046', 
                'NE_047', 'NE_048', 'NE_051', 'NE_052', 'NE_053', 'NE_054', 'NE_056', 'NE_057', 'NE_058', 'NE_059', 'NE_060', 'NE_061', 'NE_062', 'NE_063', 
                'NE_064', 'NE_065', 'E_067', 'E_069']

        
    attackTypes = ["StyleGAN2", "StyleGAN3", "Diseased", "Textured_Contact", "Contacts_+_Print", "Real_Iris","Printout", "Artificial", "Post_Mortem", 
                    "Synthetic", "Glass_Prosthesis"]


    participantPath = '*** REPLACE WITH YOUR ACTUAL PATH! ***'

    # make_new_directories(File_NCs, attackTypes)

    for file_NC in File_NCs:
        for attackType in attackTypes:
            try:
                # Load the CSV file
                input_file = f'PATH/{file_NC}/Fixations/Fixation_Sequences/parsed_{attackType}_sequences.csv'
                df = pd.read_csv(input_file, skip_blank_lines=False)

                # Identify blank rows by checking if all columns are NaN or empty
                blank_rows = df.isnull().all(axis=1) | (df.astype(str).apply(lambda x: x.str.strip()).eq("").all(axis=1))


                # Split the data into sequences
                sequences = []
                start_idx = 0

                for idx, is_blank in blank_rows.items():
                    if is_blank:
                        if start_idx < idx:  # Ensure it's not just consecutive blanks
                            sequences.append(df.iloc[start_idx:idx])
                        start_idx = idx + 1  # Move past the blank row

                # Handle the last sequence (if the file doesn’t end with a blank row)
                if start_idx < len(df):
                    sequences.append(df.iloc[start_idx:])

                # Save each sequence as a new CSV file
                for i, seq in enumerate(sequences):
                    output_file = f"{participantPath}/{file_NC}/Fixations/Fixation_Sequences/{attackType}/{file_NC}_{attackType}_sequence_{i+1}.csv"
                    seq.to_csv(output_file, index=False)
                    print(f"Saved {output_file}")
            except:
                print(f'Failed to process data: {file_NC} on -> {attackType}')