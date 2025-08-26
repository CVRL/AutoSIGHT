import pandas as pd
import copy
import re

"""
    Sliding Window Parameters:
        
        - Window Size       -> sequence length in seconds
        - Step Size         -> length in seconds that sequence start time advances
        - Gap threshold     -> Known gap size that indicates a non sequential image transition
                                i.e one row may refer to fixation data from image #2 and the next
                                row in sequence could be referring to fixation data from image #17
                                and occurs several minutes into the future
"""

#######################################################################################################################
def recalculate_elapsed_time(sequence, gap_threshold=6):
    fragments = []
    new_fragment = []
    RET = 0                     # RET = Recalculated Elapsed Time
    fragment_start_time = None

    for index, item in enumerate(sequence):
        if len(new_fragment) == 0:
            new_fragment.append(item["start_timestamp_unix"])
            fragment_start_time = item["start_timestamp_unix"]
        else:
            difference = item["start_timestamp_unix"] - fragment_start_time
            if difference < gap_threshold:
                new_fragment.append(item["start_timestamp_unix"])
            else:
                fragments.append(copy.deepcopy(new_fragment))
                new_fragment = []
                new_fragment.append(item["start_timestamp_unix"])
                fragment_start_time = item["start_timestamp_unix"]

    fragments.append(new_fragment)

    for fragment in fragments:
        RET += (fragment[-1] - fragment[0])

    # print(f'"Returning a Recalcuated Time of: {RET}')
    return RET

#######################################################################################################################
def contains_pattern(filename, pattern):
  match = re.search(pattern, filename)
  if match:
    return True
  else:
    return False
  
"""
    The final sequence is essentially never an even 10 seconds, the sliding
    window will advance 5 seconds each time. On the final sequence, we just reverse
    the dataframe and append a few extra rows until we have an evenly spaced 10
    second window of data.
"""
def pad_final_sequence(final_sequence, SST, fix_df, window_size=10):
    original_elapsed_time = recalculate_elapsed_time(final_sequence)
    start_index = fix_df.index[fix_df["start_timestamp_unix"] == SST].tolist()[0]
    reversed_df = fix_df.iloc[:start_index+1].iloc[::-1]

    new_sequence = []
    new_SST = None
    NS_Duration = None

    for i, row in reversed_df.iterrows():
        row_dict = row.to_dict()

        if new_SST is None:
            new_SST = row_dict["start_timestamp_unix"]
        else:
            new_sequence.append(row_dict)
            NS_Duration = recalculate_elapsed_time(new_sequence)

            ## Since Dataframe is now reversed, elapsed time will be negative so need to flip the sign
            NS_Duration = abs(NS_Duration)
            new_elapsed_time = original_elapsed_time + NS_Duration

            if new_elapsed_time > window_size:
                new_sequence.extend(final_sequence)
                print(f'Returning final sequence with duration of: {new_elapsed_time}')
                sorted_new_sequence = sorted(new_sequence, key=lambda x: x["start_timestamp_unix"])
                return sorted_new_sequence
            
#######################################################################################################################

File_NCs = ['NE_002', 'NE_003', 'NE_004', 'NE_005', 'E_068', 'NE_006', 'NE_007', 'NE_008', 'NE_009', 'NE_010', 'NE_011', 'NE_012', 'NE_013',
            'NE_014', 'NE_015', 'NE_016', 'NE_017', 'NE_018', 'NE_019', 'NE_020', 'NE_022', 'NE_024', 'NE_025', 'E_038', 'NE_027', 'NE_029', 'NE_030', 
            'NE_031', 'NE_033', 'NE_034', 'NE_036', 'NE_037', 'NE_039', 'NE_040', 'NE_042', 'NE_043', 'NE_044', 'NE_045', 'NE_046', 
            'NE_047', 'NE_048', 'NE_051', 'NE_052', 'NE_053', 'NE_054', 'NE_056', 'NE_057', 'NE_058', 'NE_059', 'NE_060', 'NE_061', 'NE_062', 'NE_063', 
            'NE_064', 'NE_065', 'E_067', 'E_069']
    
attackTypes = ["StyleGAN2", "StyleGAN3", "Diseased", "Textured_Contact", "Contacts_+_Print", "Real_Iris","Printout", "Artificial", "Post_Mortem", 
                "Synthetic", "Glass_Prosthesis"]


for file_NC in File_NCs:
    for attackType in attackTypes:
        # try:
            window_size = 10
            step_size = 5
            gap_threshold = 6

            sequences = []
            current_sequence = []
            sequence_start_time = None
            last_timestamp = None 
            sequence_number = 1

            # Load CSV file
            df = pd.read_csv(f"PATH/{file_NC}/Fixations/{attackType}_merged_output.csv")
            df["start_timestamp_unix"] = df["start_timestamp_unix"].astype(float)

            # Ensure timestamps are sorted
            df = df.sort_values(by="start_timestamp_unix")


            #######################################################################################################################
            # Iterate through the data
            for i, row in df.iterrows():
                timestamp = row["start_timestamp_unix"]
                row_dict = row.to_dict()
                row_dict["sequence"] = sequence_number

                if sequence_start_time is None:
                    # Start a new sequence
                    sequence_start_time = timestamp

                elapsed_time = timestamp - sequence_start_time

                if elapsed_time > gap_threshold:
                    elapsed_time = recalculate_elapsed_time(current_sequence)

                print(f'Elapsed Time: {elapsed_time}')

                if elapsed_time <= window_size:
                    # Append to current sequence
                    current_sequence.append(row_dict)
                else:
                    # Store completed sequence and start new one
                    sequences.append(current_sequence)
                    print(f'Pushing Sequence with elapsed time: {elapsed_time}')
                    # Move window forward by step size
                    sequence_start_time += step_size

                    # Remove old points outside new window
                    current_sequence = [r for r in current_sequence if r["start_timestamp_unix"] >= sequence_start_time]
                    current_sequence.append(row_dict)
                    sequence_start_time = current_sequence[0]["start_timestamp_unix"]
                    elapsed_time = recalculate_elapsed_time(current_sequence)
                    sequence_number += 1

                # Update last seen timestamp
                last_timestamp = timestamp

            #######################################################################################################################

            # Add the last sequence
            if current_sequence:
                final_elapsed_time = recalculate_elapsed_time(current_sequence)

                if final_elapsed_time <= window_size:
                    print("Final Sequence is too short, let's fix it!")
                    new_final_sequence = pad_final_sequence(current_sequence, sequence_start_time, df)
                    current_sequence = new_final_sequence
                sequences.append(current_sequence)

            # Write sequences to CSV file with blank rows in between
            output_file = f"PATH/{file_NC}/Fixations/Fixation_Sequences/parsed_{attackType}_sequences.csv"
            with open(output_file, "w") as f:
                for sequence in sequences:
                    pd.DataFrame(sequence).to_csv(f, columns=["sequence","id","start_timestamp","duration","start_frame_index","end_frame_index","norm_pos_x","norm_pos_y","dispersion","start_timestamp_unix","start_timestamp_datetime"], index=False, header=f.tell()==0)
                    f.write("\n")  # Add blank row between sequences

            print(f"Extracted {len(sequences)} sequences and saved to {output_file}.")
        # except:
        #     print(f'Encountered Problem when Processing Participant: {file_NC} on Attack Type: {attackType}')