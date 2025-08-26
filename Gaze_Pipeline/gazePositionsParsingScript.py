import os
import re
import json
import copy
import pandas as pd


def make_new_directories(File_NCs, attackTypes):
    participant_path = '*** REPLACE WITH YOUR ACTUAL PATH! ***'

    for index, File_NC in enumerate(File_NCs):
        for attackType in attackTypes:
            new_dir = f'{participant_path}/{File_NC}/Gaze_Positions/{attackType}'
            os.makedirs(new_dir, exist_ok=True)
  

def make_pandas_dataframe(gazeFile, startIndex, endIndex):
    chunks = pd.read_csv(gazeFile, chunksize=25000, dtype={'world_index': 'int32'})
    filtered_rows = []

    for chunk in chunks:
        filtered_chunk = chunk[(chunk['world_index'] >= int(startIndex)) & (chunk['world_index'] <= int(endIndex))]
        if not filtered_chunk.empty:
            filtered_rows.append(filtered_chunk)

        # Stop early if we know we've passed the end_index (since file is sorted)
        if chunk['world_index'].max() > int(endIndex):
            break

    # Concatenate all filtered chunks
    result_df = pd.concat(filtered_rows, ignore_index=True) if filtered_rows else pd.DataFrame()

    return result_df

  

if __name__ == '__main__':

    parsingObject = {
        "participant": "",
        "attackType": "",
        "gazeSequenceCSVs": []
    }

    parsingObjects = []

    File_NCs = ['NE_002', 'NE_003', 'NE_004', 'NE_005', 'E_068', 'NE_006', 'NE_007', 'NE_008', 'NE_009', 'NE_010', 'NE_011', 'NE_012', 'NE_013',
                'NE_014', 'NE_015', 'NE_016', 'NE_017', 'NE_018', 'NE_019', 'NE_020', 'NE_022', 'NE_024', 'NE_025', 'E_038', 'NE_027', 'NE_029', 'NE_030', 
                'NE_031', 'NE_033', 'NE_034', 'NE_036', 'NE_037', 'NE_039', 'NE_040', 'NE_042', 'NE_043', 'NE_044', 'NE_045', 'NE_046', 
                'NE_047', 'NE_048', 'NE_051', 'NE_052', 'NE_053', 'NE_054', 'NE_056', 'NE_057', 'NE_058', 'NE_059', 'NE_060', 'NE_061', 'NE_062', 'NE_063', 
                'NE_064', 'NE_065', 'E_067', 'E_069']

        
    attackTypes = ["StyleGAN2", "StyleGAN3", "Diseased", "Textured_Contact", "Contacts_+_Print", "Real_Iris","Printout", "Artificial", "Post_Mortem", 
                    "Synthetic", "Glass_Prosthesis"]
    
    normAttackTypes = ["StyleGAN2", "StyleGAN3", "Diseased", "Textured Contact", "Contacts + Print", "Real Iris","Printout", "Artificial", 
                       "Post-Mortem", "Synthetic", "Glass Prosthesis / Artificial"]
    
    jsonAttackTypes = []

    jsonPath = '*** REPLACE WITH YOUR ACTUAL PATH! ***'

    # Make new directories for Gaze Positions
    make_new_directories(File_NCs, attackTypes)

    for file_NC in File_NCs:
        jsonFile = f'{jsonPath}/{file_NC}_comprehensive_2.json'
        Fin = open(jsonFile)
        jsondata = json.load(Fin)
        Fin.close()

        parsingObject["participant"] = file_NC

        for AT_Index, attackType in enumerate(attackTypes):
            print(f'Processing {file_NC} on {attackType}...')
            filePath = f'PATH/{file_NC}/80_MS'
            gaze_positions_file = f'{filePath}/gaze_positions.csv'

            parsingObject["attackType"] = attackType

            for index, jsonObject in enumerate(jsondata):
                parsed_attacktype = None

                try:
                    parsed_attacktype = attackType.replace('_', ' ')
                except:
                    continue

                if jsonObject["attackType"] == normAttackTypes[AT_Index]:
                    startIndex = jsonObject["imageStartIndex"]
                    endIndex = jsonObject["initialDecisionIndex"]
                    image_df = make_pandas_dataframe(gaze_positions_file,startIndex, endIndex)
                    parsingObject["gazeSequenceCSVs"].append(image_df)

            parsingObjects.append(copy.deepcopy(parsingObject))
            parsingObject["gazeSequenceCSVs"].clear()
            print('DONE...\n')

    
    for item in parsingObjects:
        participantPath = f'PATH/{item["participant"]}/Gaze_Positions'
        print(f'\nJSON Object: {item["participant"]} -> {item["attackType"]} -> Sequences: {len(item["gazeSequenceCSVs"])}')

        for index, gaze_df in enumerate(item["gazeSequenceCSVs"]):
            try:
                gazeFilePath = f'{participantPath}/{item["attackType"]}/{item["participant"]}_{item["attackType"]}_gaze_sequence_{index+1}.csv'
                # Save to CSV
                gaze_df.to_csv(gazeFilePath, index=False)
                print(f'Successfully created csv file: {gazeFilePath}')
            except:
                print(f'Failed to procecss gaze df')
