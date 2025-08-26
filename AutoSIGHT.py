import os
import re
import math
import copy
import time
import json
import torch
import pprint
import numpy as np
import pandas as pd
import torch.nn as nn
from random import shuffle
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchmetrics import AUROC, ROC, Precision, Recall, F1Score, Accuracy
from torch.utils.data import Dataset, DataLoader
from scipy.stats import zscore


"""
 ██████╗  █████╗ ████████╗ █████╗ ██╗      ██████╗  █████╗ ██████╗ ███████╗██████╗ 
 ██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗██║     ██╔═══██╗██╔══██╗██╔══██╗██╔════╝██╔══██╗
 ██║  ██║███████║   ██║   ███████║██║     ██║   ██║███████║██║  ██║█████╗  ██████╔╝
 ██║  ██║██╔══██║   ██║   ██╔══██║██║     ██║   ██║██╔══██║██║  ██║██╔══╝  ██╔══██╗
 ██████╔╝██║  ██║   ██║   ██║  ██║███████╗╚██████╔╝██║  ██║██████╔╝███████╗██║  ██║
 ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═════╝ ╚══════╝╚═╝  ╚═╝
"""
class EyeTrackingDataLoader(Dataset):
    def __init__(self, data_dir, window_size, subs, numRows, labeled=True):
        self.subjects = subs
        self.data_directory = data_dir
        self.labeled = labeled
        self.numRows = numRows
        
        self.data = []
        self.AFDs = []          ## AFD = Average Fixation Duration
        self.FCs = []           ## FC = Fixation Counts
        self.XYEDs = []         ## Avg Euclidean Distance between Fixations
        self.labels = []
        self.sequence_ids = []  ## Needed so we can display sequences in order
                                ## Race conditions means the sequences are grabbed in a random order

        for file_NC in self.subjects:
            print(f'Loading Sequences for -> {file_NC} ...')
            sequence_file_path = f'{self.data_directory}/{file_NC}/Sliding_Window/{window_size}/Resampled'

            for sequenceFile in os.listdir(sequence_file_path):
                if sequenceFile.endswith(".csv"):
                    full_path = f'{sequence_file_path}/{sequenceFile}'
                    df = pd.read_csv(full_path, usecols=[0,1], nrows=numRows)
                    xy_data = df.to_numpy(dtype=np.float32)
                    xy_data = zscore(xy_data, axis=0)

                    ## Strip off the .csv and split string on '_' so we have the sequence number
                    ## i.e. 'E_068_resampled_sequence_17.csv' ---> [E, 068, resampled, sequence, 17]
                    sequenceFile = sequenceFile.replace(".csv", "")
                    fragments = sequenceFile.split("_")
                    sequence_number = int(fragments[-1])

                    ## This loads fixation data that corresponds to the same sliding window time frame as the XY Data
                    fixation_folder_path = f'{self.data_directory}/{file_NC}/Fixations/Sliding_Window/{window_size}'
                    fixation_file_path = f'{fixation_folder_path}/{file_NC}_fixation_sequence_{sequence_number}.csv'
                    fix_df = pd.read_csv(fixation_file_path)
                    afd = fix_df['duration'].astype(float).mean()
                    fc = fix_df['duration'].astype(float).count()

                    meanXYdist = self.average_euclidean_distance(fixation_file_path)

                    if math.isnan(meanXYdist):
                        print(f'NaN Found on Sequence #{sequence_number}, excluding sequence')
                    else:
                        self.AFDs.append(afd)
                        self.FCs.append(fc)
                        self.XYEDs.append(meanXYdist)
                        self.data.append(xy_data)
                        self.sequence_ids.append((file_NC, sequence_number))

                        """
                            Pseudo Identifiers:
                                E_000  = Expert
                                NE_000 = Non-Expert
                        """
                        if labeled:
                            label = 1 if file_NC[0] == 'E' else 0
                            self.labels.append(label)

        print('Finished Loading Sequences for Phase\n')

        ## Z-Normalize stats and convert everything to Tensors
        self.data = torch.tensor(np.array(self.data), dtype=torch.float32).permute(0, 2, 1)
        self.AFDs = torch.tensor(zscore(self.AFDs), dtype=torch.float32).unsqueeze(1)
        self.FCs = torch.tensor(zscore(self.FCs), dtype=torch.float32).unsqueeze(1)
        self.XYEDs = torch.tensor(zscore(self.XYEDs), dtype=torch.float32).unsqueeze(1)
        self.stats = torch.cat([self.AFDs, self.FCs, self.XYEDs], dim=1)

        if labeled:
            self.labels = torch.tensor(self.labels, dtype=torch.float32).unsqueeze(1)

    ## Regex utility function that returns True/False if a pattern is found in String
    def contains_pattern(self, filename, pattern):
        match = re.search(pattern, filename)
        return match is not None
    
    def average_euclidean_distance(self, file_path, x_col='norm_pos_x', y_col='norm_pos_y'):
        coordinate_df = pd.read_csv(file_path)
        coords = coordinate_df[[x_col, y_col]].values

        diffs = np.diff(coords, axis=0)
        distances = np.linalg.norm(diffs, axis=1)

        return distances.mean()

    def __len__(self):
        return len(self.data)

    ## Overloaded PyTorch Get item function for when looping on a DataLoader
    def __getitem__(self, idx):
        if self.labeled:
            return self.data[idx], self.stats[idx], self.labels[idx], self.sequence_ids[idx]
        else:
            return self.data[idx], self.stats[idx], self.sequence_ids[idx]


"""
 ███╗   ███╗ ██████╗ ██████╗ ███████╗██╗                                                       
 ████╗ ████║██╔═══██╗██╔══██╗██╔════╝██║                                                       
 ██╔████╔██║██║   ██║██║  ██║█████╗  ██║                                                       
 ██║╚██╔╝██║██║   ██║██║  ██║██╔══╝  ██║                                                       
 ██║ ╚═╝ ██║╚██████╔╝██████╔╝███████╗███████╗                                                  
 ╚═╝     ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝                                                  
                                                                                               
  █████╗ ██████╗  ██████╗██╗  ██╗██╗████████╗███████╗ ██████╗████████╗██╗   ██╗██████╗ ███████╗
 ██╔══██╗██╔══██╗██╔════╝██║  ██║██║╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝██║   ██║██╔══██╗██╔════╝
 ███████║██████╔╝██║     ███████║██║   ██║   █████╗  ██║        ██║   ██║   ██║██████╔╝█████╗  
 ██╔══██║██╔══██╗██║     ██╔══██║██║   ██║   ██╔══╝  ██║        ██║   ██║   ██║██╔══██╗██╔══╝  
 ██║  ██║██║  ██║╚██████╗██║  ██║██║   ██║   ███████╗╚██████╗   ██║   ╚██████╔╝██║  ██║███████╗
 ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚═╝   ╚═╝   ╚══════╝ ╚═════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚══════╝
                                                                                               
"""
class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=5):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


## MLP Architecture for a single stat MLP
class StatsMLP(nn.Module):
    def __init__(self, input_dim=1, hidden_dim1=128, hidden_dim2=64, output_dim=64):
        super(StatsMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


"""
    Basic Architecture:
    -------------------

          Input Data per Sliding Window                           Final MLP for E/NE Prediction
          _______________________________                          ___________________________
        - XY  Gaze Coordinates per window    --->   1D CNN   --->  | 64 output_dim | /\/\/\/ |  -   ____________
        - AFD Single Stat per window         --->     MLP    --->  | 64 output_dim | /\/\/\/ |  -  |   Single   |
        - FC  Single Stat per window         --->     MLP    --->  | 64 output_dim | /\/\/\/ |  -  | Prediction |
        - Avg Fix XY Eucl Dist per window    --->     MLP    --->  | 64 output_dim | /\/\/\/ |  -  |____________|

"""
class ResNet1D_MultiStream(nn.Module):
    def __init__(self, num_classes=1, kernel_size=5):
        super(ResNet1D_MultiStream, self).__init__()
        self.conv1 = nn.Conv1d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 64, 2, kernel_size)
        self.layer2 = self._make_layer(64, 128, 2, kernel_size, stride=2)
        self.layer3 = self._make_layer(128, 64, 2, kernel_size, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        self.duration_mlp = StatsMLP(input_dim=1, hidden_dim1=128, hidden_dim2=64, output_dim=64)
        self.count_mlp = StatsMLP(input_dim=1, hidden_dim1=128, hidden_dim2=64, output_dim=64)
        self.AED_mlp = StatsMLP(input_dim=1, hidden_dim1=128, hidden_dim2=64, output_dim=64)
        
        self.final_mlp = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def _make_layer(self, in_channels, out_channels, blocks, kernel_size, stride=1):
        layers = []
        layers.append(ResidualBlock1D(in_channels, out_channels, stride, kernel_size))
        for _ in range(1, blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels, kernel_size=kernel_size))
        return nn.Sequential(*layers)
    
    def forward(self, x, stats):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)                                     
        
        duration = stats[:, 0:1]                                            
        count = stats[:, 1:2]                                               
        AED = stats[:, 2:3]
        duration_out = self.duration_mlp(duration)                          
        count_out = self.count_mlp(count)                                   
        AED_out = self.AED_mlp(AED)
        
        out = torch.cat([out, duration_out, count_out, AED_out], dim=1)     
        
        out = self.final_mlp(out)                                           
        return out


"""
 ████████╗██████╗  █████╗ ██╗███╗   ██╗██╗███╗   ██╗ ██████╗     
 ╚══██╔══╝██╔══██╗██╔══██╗██║████╗  ██║██║████╗  ██║██╔════╝     
    ██║   ██████╔╝███████║██║██╔██╗ ██║██║██╔██╗ ██║██║  ███╗    
    ██║   ██╔══██╗██╔══██║██║██║╚██╗██║██║██║╚██╗██║██║   ██║    
    ██║   ██║  ██║██║  ██║██║██║ ╚████║██║██║ ╚████║╚██████╔╝    
    ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═══╝ ╚═════╝     
                                                                 
                 ██╗      ██████╗  ██████╗ ██████╗               
                 ██║     ██╔═══██╗██╔═══██╗██╔══██╗              
                 ██║     ██║   ██║██║   ██║██████╔╝              
                 ██║     ██║   ██║██║   ██║██╔═══╝               
                 ███████╗╚██████╔╝╚██████╔╝██║                   
                 ╚══════╝ ╚═════╝  ╚═════╝ ╚═╝     
"""
def train_and_evaluate(train_dataset, val_dataset, test_dataset, trial_num, window_size, kernel_size=5, batch_size=8, num_epochs=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}\n")
    print('Training Loop:')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    model = ResNet1D_MultiStream(kernel_size=kernel_size).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    accuracy = Accuracy(task="binary", threshold=0.5)
    best_val_accuracy = 0.0

    ## Saving Best Performing Model
    ## Timestamps are used for unique-ness
    timestamp = time.time()
    timestamp = int(timestamp)
    timestamp = str(timestamp)
    model_path_dir = '*** REPLACE WITH YOUR ACTUAL PATH! ***'
    best_model_path = f"{model_path_dir}/best_model_{timestamp}.pth"
    
    ## Training loop with validation
    model.train()
    for epoch in range(num_epochs):
        running_train_loss = 0.0
        for xy_data, stats, labels, _ in train_loader:
            xy_data = xy_data.to(device)
            stats = stats.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(xy_data, stats)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
        
        train_loss = running_train_loss / len(train_loader)
        
        ## Validation
        model.eval()
        running_val_loss = 0.0
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for xy_data, stats, labels, _ in val_loader:
                xy_data = xy_data.to(device)
                stats = stats.to(device)
                labels = labels.to(device)
                
                outputs = model(xy_data, stats)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                preds = torch.sigmoid(outputs)
                val_preds.append(preds.cpu())
                val_labels.append(labels.cpu())
        
        val_loss = running_val_loss / len(val_loader)
        val_preds = torch.cat(val_preds)
        val_labels = torch.cat(val_labels)
        val_accuracy = accuracy(val_preds, val_labels.long())
        
        ## Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model with validation accuracy: {best_val_accuracy:.4f}")
        
        model.train()
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
    
    ## Test evaluation
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    all_preds = []
    all_labels = []
    all_sequence_ids = []
    with torch.no_grad():
        for xy_data, stats, labels, seq_ids in test_loader:
            xy_data = xy_data.to(device)
            stats = stats.to(device)
            labels = labels.to(device)
            
            outputs = model(xy_data, stats)
            preds = torch.sigmoid(outputs)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_sequence_ids.append(seq_ids)
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    ## AUROC and ROC curve
    auroc_metric = AUROC(task="binary")
    roc = ROC(task="binary")
    auroc_score = auroc_metric(all_preds, all_labels.long())
    fpr, tpr, thresholds = roc(all_preds, all_labels.long())
    
    ## Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUROC = {auroc_score:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('roc_curve.png')
    plt.close()
    
    ## Additional metrics
    precision = Precision(task="binary", threshold=0.5)
    recall = Recall(task="binary", threshold=0.5)
    f1 = F1Score(task="binary", threshold=0.5)
    
    precision_score = precision(all_preds, all_labels.long())
    recall_score = recall(all_preds, all_labels.long())
    f1_score = f1(all_preds, all_labels.long())
    
    ## Write test predictions to CSV
    test_results = []
    for i in range(len(all_preds)):
        subject_id = all_sequence_ids[i][0][0]
        seq_num = all_sequence_ids[i][1].item()

        true_label = all_labels[i].item()
        pred_prob = all_preds[i].item()
        pred_class = 1 if pred_prob > 0.5 else 0
        test_results.append({
            'subject_id': subject_id,
            'sequence_number': seq_num,
            'true_label': true_label,
            'predicted_probability': pred_prob,
            'predicted_class': pred_class
        })
    
    results_df = pd.DataFrame(test_results)
    output_dir = '*** REPLACE WITH YOUR ACTUAL PATH! ***'
    df_name = f'Trial_{trial_num+1}_{window_size}_050925_KS{kernel_size}_{num_epochs}epochs.csv'
    results_df.to_csv(f'{output_dir}/{df_name}', index=False)
    
    # Print results
    print(f"\nTest AUROC: {auroc_score:.4f}")
    print(f"Test Precision: {precision_score:.4f}")
    print(f"Test Recall: {recall_score:.4f}")
    print(f"Test F1-Score: {f1_score:.4f}\n")

    return float(auroc_score), float(precision_score), float(recall_score), float(f1_score), str(df_name), str(best_model_path)


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
if __name__ == "__main__":

    non_experts = ['NE_002','NE_003','NE_004','NE_005','NE_006','NE_007','NE_009','NE_010','NE_011',
                    'NE_012','NE_013','NE_014','NE_015','NE_016','NE_017','NE_018','NE_019','NE_020',
                    'NE_022','NE_024','NE_025','NE_027','NE_029','NE_030','NE_031','NE_033','NE_034',
                    'NE_036','NE_037','NE_039','NE_040','NE_042','NE_043','NE_044','NE_045','NE_046', 
                    'NE_047','NE_048','NE_051','NE_052','NE_053','NE_054','NE_056','NE_057','NE_058',
                    'NE_059','NE_060','NE_061','NE_062','NE_063','NE_064', 'NE_065']
    
    experts = ['E_038', 'E_067', 'E_068', 'E_069', 'E_072']

    """
        Experiment Object:
            - Stores relevant data of each trial run
            - Allows for analyzing trends such as which experts or non-experts
              are contributing to better or worse performing splits.
            - Also shows which model path corresponds to which performance
        
        Example Object:
                {
                    "Trial": 1,
                    "Window_Size": "5_Second",
                    "Training_Subs": [
                        "E_072",
                        "E_069",
                        "E_067",
                        "NE_017",
                        "NE_027",
                        "NE_020"
                    ],
                    "Validation_Subs": [
                        "E_068",
                        "NE_063"
                    ],
                    "Testing_Subs": [
                        "E_038",
                        "NE_006"
                    ],
                    "Test AUROC": 0.7018,
                    "Test Precision": 0.7002,
                    "Test Recall": 0.8138,
                    "Test F1-Score": 0.7528,
                    "Results_csv": "Trial_1_5_Second_050925_KS15_80epochs.csv",
                    "best_model_path": "PATH/Saved_Models/best_model_1746835044.pth"
                }
    """
    experimentObject = {
        "Trial": 0,
        "Window_Size": "",
        "Training_Subs": [],
        "Validation_Subs": [],
        "Testing_Subs": [],
        "Test AUROC": 0.0,
        "Test Precision": 0.0,
        "Test Recall": 0.0,
        "Test F1-Score": 0.0,
        "Results_csv": "",
        "best_model_path": ""
    }

    
    """
        Test Run Process:
            - Experiment was set up to test each window size on a variety of kernel sizes
            - rows refers to the fixed input size of gaze data
            - sampling rate is 200 times a second so row size = 200x where x is window size
    """
    window_sizes = ['5_Second','10_Second','15_Second','20_Second', '30_Second']
    rows = [1000,2000,3000,4000,6000]
    kernel_sizes = [5,7,9,11]

    epochs = 80

    for index, window_size in enumerate(window_sizes):
        for kernel_size in kernel_sizes:

            experimentObjects = []
            trials = 12

            for i in range(trials):
                print(f'Trial #{i+1}')
                shuffle(non_experts)
                shuffle(experts)

                testingSubjects = []
                testingSubjects.append(experts[0])
                testingSubjects.append(non_experts[0])

                validationSubjects = []
                validationSubjects.append(experts[1])
                validationSubjects.append(non_experts[1])

                xTrainSubs = experts[2:]
                nxTrainSubs = non_experts[-len(xTrainSubs):]
                trainingSubjects = xTrainSubs + nxTrainSubs

                participantPath = '*** REPLACE WITH YOUR ACTUAL PATH! ***'

                experimentObject["Trial"] = i+1
                experimentObject["Window_Size"] = window_sizes[index]
                experimentObject["Training_Subs"] = trainingSubjects
                experimentObject["Validation_Subs"] = validationSubjects
                experimentObject["Testing_Subs"] = testingSubjects


                train_dataset = EyeTrackingDataLoader(participantPath, window_sizes[index], trainingSubjects, numRows=rows[index], labeled=True)
                val_dataset = EyeTrackingDataLoader(participantPath, window_sizes[index], validationSubjects, numRows=rows[index], labeled=True)
                test_dataset = EyeTrackingDataLoader(participantPath, window_sizes[index], testingSubjects, numRows=rows[index], labeled=True)

                results = []
                results = train_and_evaluate(train_dataset, val_dataset, test_dataset, i, window_sizes[index], kernel_size=kernel_size, num_epochs=epochs)

                experimentObject["Test AUROC"] = round(results[0], 4)
                experimentObject["Test Precision"] = round(results[1], 4)
                experimentObject["Test Recall"] = round(results[2], 4)
                experimentObject["Test F1-Score"] = round(results[3], 4)
                experimentObject["Results_csv"] = results[4]
                experimentObject["best_model_path"] = results[5]

                experimentObjects.append(copy.deepcopy(experimentObject))


            # Dump the results to new JSON file
            with open (f'Results_{window_sizes[index]}_KS{kernel_size}_{epochs}epochs.json', 'w') as Fout:
                json.dump(experimentObjects, Fout, indent=4)
