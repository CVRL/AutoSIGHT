### Gaze Coordinates Pipeline:
This section outlines the files needed to process coordinate gaze data to be preprocessed for use in the Sliding Window-style CNN and the correct order in which each file needs to be run as the process can be a bit confusing. These scripts will need to be heavily modified to suit your needs. Early on in the research process we segmented data by Iris PAD attack types which is why many of the scripts were concerned with aggregating data in this manner. Later on we re-segmented data in a continuous manner so we could do real-time analysis. After spending some time to evaluate which scripts or sections are relevant to your needs, this pipeline shows how you can segment raw gaze data into sliding window timed segments of your choosing and there is a pipeline to resample missing sequences due to fluctuating sampling rates in eye tracking recordings. This process provides a way to ensure a fixed input size for a window size which is necessary to run the AutoSIGHT model.

 
1. <strong><u>fixationParseConcat.ipynb</u></strong>
	- This segments fixation data by attack type and concatenates them in order by appearance
	- i.e. Fixations from the first instance of an attack type followed by the next instance of same attack type etc.
	- Final output is **{path}/Fixations/{attackType}_merged_output.csv**


2. <strong><u>sequenceParsingScript.py</u></strong>
	- This takes the merged_output.csv files by participant and attackType and extracts fixations into 10 second sequences that are concatenated into one parsed csv file with a blank row separating sequences.
  - Replace the 10 second variable to desired length to segement data into a different sliding window size
	- File that is created under the Fixation_Sequences folder with **parsed_{attackType}_sequences.csv** naming convention
	
	
3. <strong><u>fixationSequenceSegmenter.py</u></strong>
	- This script loads the parsed_{attackType}_sequences.csv files and creates individual fixation sequences.
	- These fixation sequences are saved as individual files such as:
		- **Fixations/Fixation_Sequences/{attackType}/{file_NC}_{attackType}_sequence{num}.csv**
	- These sequence files have the world index ranges that can then be used to objectively sample the same exact sequence but replace the fixation only data with gaze positions data that should have a higher sampling rate.
	  
4. <strong><u>gazePositionsParsingScript.py</u></strong>
	- This script will loop over the JSON lists of each participant, grab their start and end decision indexes for each image and concatenate the gaze positions rows for each image's initial phase and store them in a dictionary object.
	- Essentially if a participant saw 13 Live Irises in their experiment, this script will create 13 gaze sequences
	- These will later be concatenated as merged gaze sequences that can then be segmented out in time sliding window segments
	- **NOTE**, this script is quite slow to run for all participants but only needs to be run once
  - Replacing Pandas functions with Polars functions should speed up the process 
	
5. <strong><u>gazePositionsSegmenter.py</u></strong>
	- This script takes the concatenated gaze sequences and splits them back into timed sliding window segments of gaze data specific to attack type.
	- This is different from the ones that did this for the fixation data. But this script will loop over the frames to get the index ranges of gaze data that is used in the sliding window application.
	- This script is also very slow but only needs to be run once.

6. <strong><u>pupilcoreTimestampSyncing.py</u></strong>
	- This script pushes through Unix Timestamps to the newly created gaze sequences.
	- This is necessary to use the Traces library to interpolate missing data due to fluxating sampling rates in the recordings

7. <strong><u>gazeInterpolationTraces.py</u></strong>
	- This file starts by running a function to make new directories for each attack type in the **Gaze_Positions** folder
		- The function will also add a folder under each of these new attack type folders called **Resampled**
	- The script should then use the Traces library to linearly interpolate missing X, Y coordinate data if the gaze sequence is shorter than 2000 rows of data i.e. the Sampling Rate dipped below 200 Hz or reduce an oversampling rate down to exactly 2000 entries so that we can have an equal amount of data for a fixed input tensor size.
