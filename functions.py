import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import models
from processdata import TimeSeriesDataset




def sensor_loc_fun(sensor_path, sensor_place, df_2_data):
  # This function is specific to the dataset used in this project. Update it according the the types of signals (joint angles, EMG, etc) in your dataset.
  
  """
  Get column indices for sensor data based on sensor type and location

  Args: 
    sensor_path (str): Type of sensor data (e.g., '3acc_Training', '3gyro_Training').
    sensor_place (str): Sensor placement (e.g., 'RightAnkle', 'Chest').

  Returns:
    sensor_locations: array of column indices
    num_sensors: number of signal measurements from sensors (e.g, triaxial = 3)

  """

  if sensor_path == '3acc_Training':  # triaxial accelerometer
    num_sensors = 3 # number of signal measurements

    signal1 = sensor_place + "_Acceleration_x"  # append strings
    signal2 = sensor_place + "_Acceleration_y"
    signal3 = sensor_place + "_Acceleration_z"
    sensor_locations = np.array([df_2_data.columns.get_loc(signal1), df_2_data.columns.get_loc(signal2), df_2_data.columns.get_loc(signal3)])

    return sensor_locations, num_sensors

  elif sensor_path == '3gyro_Training':  # triaxial gyroscope
    num_sensors = 3 # number of signal measurements

    signal1 = sensor_place + "_AngularVelocity_x"  # append strings
    signal2 = sensor_place + "_AngularVelocity_y"
    signal3 = sensor_place + "_AngularVelocity_z"
    sensor_locations = np.array([df_2_data.columns.get_loc(signal1), df_2_data.columns.get_loc(signal2), df_2_data.columns.get_loc(signal3)])

    return sensor_locations, num_sensors

  elif sensor_path == '3acc3gyro_Training': # triaxial accelerometer and triaxial gyroscope
    num_sensors = 6 # number of signal measurements

    signal1 = sensor_place + "_Acceleration_x"  # append strings
    signal2 = sensor_place + "_Acceleration_y"
    signal3 = sensor_place + "_Acceleration_z"
    signal4 = sensor_place + "_AngularVelocity_x"
    signal5 = sensor_place + "_AngularVelocity_y"
    signal6 = sensor_place + "_AngularVelocity_z"
    sensor_locations = np.array([df_2_data.columns.get_loc(signal1), df_2_data.columns.get_loc(signal2), df_2_data.columns.get_loc(signal3), df_2_data.columns.get_loc(signal4), df_2_data.columns.get_loc(signal5), df_2_data.columns.get_loc(signal6)])

    return sensor_locations, num_sensors

  elif sensor_path == 'Xacc_Training': # uniaxial accelerometer
    num_sensors = 1 # number of signal measurements

    signal1 = sensor_place + "_Acceleration_x"  # append strings
    sensor_locations = np.array([df_2_data.columns.get_loc(signal1)])

    return sensor_locations, num_sensors

  else:
    print('Signal combination not properly named')




def partition_data(load_X, n, lags):

  """
    Randomly splits data into training, validation, and test sets

  Args:
    load_X (numpy array): Input data
    n (int) = Total number of observations
    lags (int) = Number of time steps used for training

  Return:
    train_indices: Randomly selected indices for training
    valid_indices: Every other remaining index for validation
    test_indices: Remaining indices for testing
  """
  #np.random.seed(0)
  train_indices = np.random.choice(n - lags, size=int(n*0.60), replace=False)
  mask = np.ones(n - lags)
  mask[train_indices] = 0
  valid_test_indices = np.arange(0, n - lags)[np.where(mask!=0)[0]]
  valid_indices = valid_test_indices[::2]
  test_indices = valid_test_indices[1::2]

  return train_indices, valid_indices, test_indices



def partition_data_seq(load_X, n, lags):

    """
      Sequentially splits data into training, validation, and test sets

    Args:
      load_X (numpy array): Input data
      n (int) = Total number of observations
      lags (int) = Number of time steps used for training

    Return:
      train_indices: indices correspoding to first 30% and last 30% of n for training
      valid_indices: idices corresponding to innermost 20% of n for validation
      test_indices: remaining indices for testing
    """
    
    interval10 = int(0.10*n) # 10% of n
    train_indices = np.arange(0, 6*interval10) # train indices are first 60% of n
    valid_indices = np.arange(6*interval10 + 1, 8*interval10 - int(lags/2)) # valid indices are next 20% of n
    test_indices = np.arange(8*interval10 - int(lags/2) + 1, n-lags) # test indices are last 20% of n

    return train_indices, valid_indices, test_indices



def transform_data(load_X, train_indices):
  """
    Normalize input data using MinMaxScaler based on training set statistics

  Args:
    load_X (numpy array): The dataset to be normalized
    train_indices (array): Indices correspinding to the training set

  Return:
    transformed_X: dataset after MinMax scaling
    sc: fitted MinMaxScaler for inverse transformation if needed
  """
  # sklearn's MinMaxScaler is used to preprocess the data for training and
  # we generate input/output pairs for the training, validation, and test sets.
  sc = MinMaxScaler()
  sc = sc.fit(load_X[train_indices])
  transformed_X = sc.transform(load_X)

  return transformed_X, sc



### Generate input sequences to a SHRED model
def train_SHRED_model(transformed_X, sc, train_indices, valid_indices, test_indices, sensor_locations, num_sensors, m, n, lags):
  """
    Trains SHRED model for time series reconstruction

  Args: 
    transformed_X (numpy array): MinMax scaled dataset.
    sc (MinMaxScaler): Fitted MinMax scaler for inverse transformation.
    train_indices (array): Indices for the training set.
    valid_indices (array): Indices for the validation set.
    test_indices (array): Indices for the test set.
    sensor_locations (array): Column indices for sensor data.
    num_sensors (int): Number of signal measurements from sensors (e.g, triaxial = 3)
    m (int): Number of features per timestep
    n (int): Total number of time steps (observations)
    lags (int): length of trajectory
    
  Return:
    test_recons: Reconstructed data from the SHRED model on the test set
    test_ground_truth: Ground truth data from the test set
  """

  all_data_in = np.zeros((n - lags, lags, num_sensors))
  for i in range(len(all_data_in)):
      all_data_in[i] = transformed_X[i:i+lags, sensor_locations]
  ### Generate training validation and test datasets both for reconstruction of states and forecasting sensors
  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  train_data_in = torch.tensor(all_data_in[train_indices], dtype=torch.float32).to(device)
  valid_data_in = torch.tensor(all_data_in[valid_indices], dtype=torch.float32).to(device)
  test_data_in = torch.tensor(all_data_in[test_indices], dtype=torch.float32).to(device)

  ### -1 to have output be at the same time as final sensor measurements
  train_data_out = torch.tensor(transformed_X[train_indices + lags - 1], dtype=torch.float32).to(device)
  valid_data_out = torch.tensor(transformed_X[valid_indices + lags - 1], dtype=torch.float32).to(device)
  test_data_out = torch.tensor(transformed_X[test_indices + lags - 1], dtype=torch.float32).to(device)

  train_dataset = TimeSeriesDataset(train_data_in, train_data_out)
  valid_dataset = TimeSeriesDataset(valid_data_in, valid_data_out)
  test_dataset = TimeSeriesDataset(test_data_in, test_data_out)
  shred = models.SHRED(num_sensors, m, hidden_size=64, hidden_layers=2, l1=350, l2=400, dropout=0.1).to(device)
  validation_errors = models.fit(shred, train_dataset, valid_dataset, batch_size=64, num_epochs=500, lr=1e-3, verbose=True, patience=5)

  # Generate reconstructions from the test set and print mean square error compared to the ground truth
  test_recons = sc.inverse_transform(shred(test_dataset.X).detach().cpu().numpy())
  test_ground_truth = sc.inverse_transform(test_dataset.Y.detach().cpu().numpy())

  return test_recons, test_ground_truth



def rmse_error(Ypred, Ytest):
    """
        Computes the Root Mean Squared Error between two data arrays
        
    Args:
        Ypred (array): Predicted values
        Ytest (array): True values
    
    Return:
        err: RSME value
    """
    err = np.sqrt(((Ypred - Ytest) ** 2).mean()) 

    return err



def mae_error(Ypred, Ytest):
    """
        Computes the Mean Absolute Error between two data arrays
        
    Args:
        Ypred (array): Predicted values
        Ytest (array): True values
    
    Return:
        err: MAE value
    """
    
    err = (abs(Ypred - Ytest)).mean() 

    return err



def mbe_error(Ypred, Ytest):
    """
        Computes the Mean Bias Error between two data arrays
        
    Args:
        Ypred (array): Predicted values
        Ytest (array): True values
    
    Return:
        err: MBE value
    """
    
    err = (Ypred - Ytest).mean()

    return err



def concatRaw(AC_train, sensor_place, sensor_path, start, end, tmp_Ypred, tmp_Ytest, subj):

    """
        Concatenates raw prediction and test data into a DataFrame with detailed sensor and output information.

    Args:
        AC_train (pandas DataFrame): Not used?
        sensor_place (str): Location of the sensor (e.g., "Waist", "Chest").
        sensor_path (str): Path indicating the type of sensor data (e.g., '3acc_Training').
        start (int): Start index for processing subjects.
        end (int): End index for processing subjects.
        tmp_Ypred (DataFrame): Predicted output values.
        tmp_Ytest (DataFrame): True output values for validation.
        subj (str): Subject identifier.

    Returns:
        df_rmse: A DataFrame containing detailed sensor and error information.
    """

    columns = ['Subject', 'Pred', 'True', 'Input Location', 'Sensor Type', 'Output Location', 'Output Signal', 'Output Direction', 'Output Axis', 'Assessment']
    df_rmse = pd.DataFrame(columns=columns)
    num_states = 36 # number of features hardcoded, consider adding input to function def line

    for i in range(start, end+1):
        # if i < 10:
        #     subj = 'P0'+str(i)
        # elif i >= 10:
        #     subj = 'P'+str(i)


        tmp = np.zeros([1,num_states])
        for i in range(0,num_states):
            tmp[:,i] = rmse_error(tmp_Ytest.iloc[:,i],tmp_Ypred.iloc[:,i])

        time_vec = tmp_Ytest['Time']

        tmp_mse = pd.DataFrame(tmp, columns = tmp_Ypred.columns[:num_states])
        #########################################################
        # create df with one error observation per row
        # add columns for subject, value, input location, input signal, output location, output signal, output direction, output axis, Assessment
        
        if sensor_path == '3acc_Training':
            sensor_type = 'Triaxial Accelerometer'
        elif sensor_path == '3acc3gyro_Training':
            sensor_type = 'Triaxial Accelerometer & Gyroscope'
        elif sensor_path == 'Xacc_Training':
            sensor_type = 'Uniaxial Accelerometer'
        else:
            sensor_type = 'Triaxial Gyroscope'

        row_count = 0
        time_ind = -1
        for row_Ypred in range(min(len(tmp_Ypred), 200)):
            for idx, value in tmp_Ypred.iloc[row_Ypred, :36].items():
                tmp_tidy = pd.DataFrame(columns = df_rmse.columns)
                # assign subject
                tmp_tidy.loc[row_count,'Subject'] = subj
                # assign predicted value
                tmp_tidy.loc[row_count,'Pred'] = value
                # assign true value
                tmp_tidy.loc[row_count, 'True'] = tmp_Ytest[idx][row_Ypred]
                # assign input location
                tmp_tidy.loc[row_count,'Input Location'] = sensor_place
                # assign input signal
                tmp_tidy.loc[row_count,'Sensor Type'] = sensor_type
                # assign output location
                if 'Waist' in idx:
                    tmp_tidy.loc[row_count,'Output Location'] = 'Waist'
                elif 'Chest' in idx:
                    tmp_tidy.loc[row_count,'Output Location'] = 'Chest'
                elif 'LeftAnkle' in idx:
                    tmp_tidy.loc[row_count, 'Output Location'] = 'LeftAnkle'
                elif 'RightAnkle' in idx:
                    tmp_tidy.loc[row_count, 'Output Location'] = 'RightAnkle'

                # assign output signal
                if 'Accel' in idx:
                    tmp_tidy.loc[row_count, 'Output Signal'] = 'Acceleration'
                elif 'Angular' in idx:
                    tmp_tidy.loc[row_count, 'Output Signal'] = 'Angular Velocity'
                elif 'Magnetic' in idx:
                    tmp_tidy.loc[row_count, 'Output Signal'] = 'Magnetic Field'

                # assign output direction
                if 'Chest' in idx and '_y' in idx:
                    tmp_tidy.loc[row_count, 'Output Direction'] = 'ML'
                elif 'Chest' in idx and '_z' in idx:
                    tmp_tidy.loc[row_count, 'Output Direction'] = 'AP'
                elif '_y' in idx: # Waist and Ankles
                    tmp_tidy.loc[row_count, 'Output Direction'] = 'AP'
                elif '_z' in idx: # Waist and Ankles
                    tmp_tidy.loc[row_count, 'Output Direction'] = 'ML'
                else: # all others contain '_x'
                    tmp_tidy.loc[row_count, 'Output Direction'] = 'Vertical'

                # assign output axis
                if '_x' in idx:
                    tmp_tidy.loc[row_count, 'Output Axis'] = 'x'
                elif '_y' in idx:
                    tmp_tidy.loc[row_count, 'Output Axis'] = 'y'
                elif '_z' in idx:
                    tmp_tidy.loc[row_count, 'Output Axis'] = 'z'
                
                # assign assessment
                tmp_tidy.loc[row_count, 'Assessment'] = 'Single Speed'

                # assign time stamp
                if row_count % num_states == 0:
                    time_ind = time_ind + 1
                tmp_tidy.loc[row_count, 'Time'] = time_vec[time_ind]

                # concatenate with the previous iterations
                df_rmse = pd.concat([df_rmse, tmp_tidy])
                row_count = row_count + 1


    return df_rmse



def extractSignal(signal_loc, signal_type, signal_ax, df_SHRED):

    """
        Combines the predicted and true sensor signal data for a given location, type, and axis.

    Args:
        signal_loc (str): The location of the sensor (e.g., 'Waist', 'Chest').
        signal_type (str): The type of signal (e.g., 'Acceleration', 'Angular Velocity').
        signal_ax (str): The axis of the sensor signal (e.g., 'x', 'y', 'z').
        df_SHRED (DataFrame): DataFrame containing SHRED model predictions and true values.

    Returns:
        df_signal: A DataFrame containing the time, predicted values (SHRED), and true values for the specified signal.
    """

    df_SHRED_signal = pd.DataFrame(columns = ['Type' , 'Time', 'Value'])
    df_true_signal = pd.DataFrame(columns = df_SHRED_signal.columns)

    # get SHRED data
    df_SHRED_reduced = df_SHRED[(df_SHRED['Output Axis'] == signal_ax) & (df_SHRED['Output Location'] == signal_loc) & (df_SHRED['Output Signal'] == signal_type)]
    df_SHRED_signal['Time'] = df_SHRED_reduced['Time']
    df_SHRED_signal['Value'] = df_SHRED_reduced['Pred']
    df_SHRED_signal['Type'] = 'SHRED'
    max_length = len(df_SHRED_signal)

    # get true data
    df_true_signal['Time'] = df_SHRED_reduced['Time']
    df_true_signal['Value'] = df_SHRED_reduced['True']
    df_true_signal['Type'] = 'True'

    # concatenate SHRED with true data
    df_signal = pd.concat([df_true_signal, df_SHRED_signal])
