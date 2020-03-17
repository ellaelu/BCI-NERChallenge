import numpy as np
import pandas as pd
import glob
import time
from scipy import signal

'''
extract_d(files)
Ingest Data by looping through files

Epoch 1.3 seconds after feedbackevent == 1 using epoch_d function

Append values to list of arrays called temp


Input: 
    files: array of string of file names (Data_S*_Sess*.csv)
Output: 
    temp: final array of appended values
'''

train_files = glob.glob('examples\Data\Train\Data*.csv')
test_files = glob.glob('examples\Data\Test\Data*.csv')
print(train_files[0:6])

def butter_filter(order, low_pass, high_pass, fs,sig):
    nyq = 0.5 * fs
    lp = low_pass / nyq
    hp = high_pass / nyq
    sos = signal.butter(order, [lp, hp], btype='band', output = 'sos')
    return signal.sosfilt(sos, sig)

'''
extract_d(files)
Ingest Data by looping through files

Epoch 1.3 seconds after feedbackevent == 1 using epoch_d function

Append values to list of arrays called temp


Input: 
    files: array of string of file names (Data_S*_Sess*.csv)
Output: 
    temp: final array of appended values
'''
def extract_d(files, e_s = None, baseline = True, bandpass = True):
    start = time.time()
    
    training_subjects = 16 #num of training subjects
    num_of_fb = 340 #num of feedbacks / subject
    freq = 200 #sampling rate
    epoch_time = 1.3 #proposed epoching time in seconds
    epoch = int(freq * epoch_time) #epoch in indices 
    #epoch_s = int(freq * e_s)
    num_of_cols = int(59) 
    eeg_cols = int(56)
    b_s = int(-0.4*freq) #index where baseline starts relative to feedback (-400ms)
    b_e = int(-0.3*freq) #index where baseline ends relative to feedback (-300ms)
    order = 5 #butterworth order
    low_pass = 1 #low frequency pass for butterworth filter
    high_pass = 40 #high frequency pass for butterworth filter
    
    channels = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1',
       'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz',
       'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2',
       'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
       'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
       'PO7', 'POz', 'P08', 'O1', 'O2']
    
    temp = np.empty((1,epoch,num_of_cols), float)
    for i, f in enumerate(files):
        print(i,f, temp.shape)
        df = pd.read_csv(f) #read each file
        index_fb = df[df['FeedBackEvent'] == 1].index.values
        df_array = np.array(df) 
        
        #uncomment below for butterworth filter
        if bandpass == True:
            eeg = df_array[:,1:57] #only eeg values to apply butterworth filter
            for i, channel in enumerate(channels):
                raw_eeg = df[channel].values
                eeg_filtered = butter_filter(order, low_pass, high_pass, freq, raw_eeg) #butterworth filter applied
                eeg[:,i] = eeg_filtered
            df = np.array(df)
            df[:,1:57] = eeg #replacing old eeg values with new ones
        else:
            df = np.array(df)
        
        for j, indx in enumerate(index_fb): #epoching 260 indexes (1.3 seconds) after each stimulus
            if e_s != None:
                epoch_array = df[indx:(indx+int(epoch)),:]
                epoch_array = epoch_array.reshape((1,int(epoch),int(epoch_array.shape[1])))
            else:
                epoch_array = df[indx:(indx+int(epoch)),:]
                epoch_array = epoch_array.reshape((1,int(epoch),int(epoch_array.shape[1])))

            #uncomment below for baseline correction
            if baseline == True:
                baseline_array = df[indx+b_s:indx+b_e, 1:57] #baseline correction of 100ms (20 indexes), 400ms to 300ms before fb
                baseline_array = baseline_array.reshape((1,20,int(baseline_array.shape[1])))
                baseline_mean = np.mean(baseline_array, axis = 1)
                epoch_array[:,:,1:57] = epoch_array[:,:,1:57] - baseline_mean #noise subtracted from epoched data
            
            if i == 0:
                temp = np.vstack((temp,epoch_array)) #stacking the first epoch
            else:
                temp = np.vstack((temp,epoch_array))
                
    now = time.time()
    print('Elapsed Time: ' + str(int(now-start)) + ' seconds')
    return temp

train = extract_d(train_files)
test = extract_d(test_files)

np.save('tr1.npy',train[1:,:,:])
np.save('te1.npy',test[1:,:,:])
train = np.load('tr1.npy')
test = np.load('te1.npy')

print(train.shape)
print(test.shape)

training_subjects = int(16)
num_of_fb = int(340)
freq = int(200)
epoch_time = 1.3
epoch = int(freq * epoch_time)
num_of_cols = int(59)
eeg_cols = int(56)

train = np.reshape(train, (5440, num_of_cols, epoch))
test = np.reshape(test, (3400, num_of_cols, epoch))

EEG_train = train[:,1:57,:].reshape(5440*epoch, eeg_cols)
EEG_test = test[:,1:57,:].reshape(3400*epoch, eeg_cols)

train_filtered = EEG_train.reshape(5440, int(eeg_cols), int(epoch))
test_filtered = EEG_test.reshape(3400, int(eeg_cols), int(epoch))

print(train_filtered.shape)
print(test_filtered.shape)

np.save('examples\Data\X_train_bwbs.npy',train_filtered)
np.save('examples\Data\X_test_bwbs.npy',test_filtered)