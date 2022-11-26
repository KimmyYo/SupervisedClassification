import pandas as pd
import numpy as np
   
def handleDataset(Location):                                                   # 切割分成 data 與 label
    
    csvfile = open(Location)
    df = pd.read_csv(csvfile)
    df = df.fillna(df.mean())                                                  # 將 Nan 填入各欄位均值
    df = df.drop_duplicates()                                                  # 去除重複的資料  
    df_label = df['Outcome']
    df_data = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness' 
                  , 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
    return df_data, df_label

def standardization(data):                                                       # 資料標準化

    data_std = pd.DataFrame()
    for x in range(len(columns)):   
        data_std[columns[x]] = (data[columns[x]] - data.mean()[x]) 
        data_std[columns[x]] = data_std[columns[x]] / data.std()[x]
    return data_std
    
def outlier(data_normal):                                                      # 蓋帽法處理 outlier
       
    for x in range(len(columns)): 
        for y in range(len(data_normal[columns[x]])):
            if abs(data_normal.loc[y,columns[x]]) > 3:
                if data_normal.loc[y,columns[x]] >= 0:
                    data_normal.loc[y,columns[x]] = 3
                else:
                    data_normal.loc[y,columns[x]] = -3
    return data_normal    

def prediction(dist, train_label, k):                                          # 用歐式距離預測 label 
    
    dist_sort = dist.sort_values()
    index = dist_sort.index[0:k]
    
    true = 0
    false = 0
    predict_value = 0
    for x in index:
        if train_label.values[x] == 1:                                         # 比較 1 多還是 0 多
            true += 1
        else:
            false += 1
    if true > false:
        predict_value = 1                                                   
    return predict_value

def KNN(train_label, train_data, test_data, k):                                # KNN  

    test_label_Predict = [] 
    for x in range(len(test_data[columns[0]])):
        distance = []
        for y in range(len(train_data[columns[0]])):
            data_1 = np.array(test_data.loc[[x], :])
            data_2 = np.array(train_data.loc[[y], :])        
            temp = np.linalg.norm(data_1 - data_2)                             # 計算歐式距離 
            distance.append(temp)
        dist = pd.Series(distance)
        predict_value = prediction(dist, train_label, k)  
        test_label_Predict.append(predict_value) 
    test_label_P = pd.Series(test_label_Predict)    
    return test_label_P
    
def correct_rate(test_label, test_label_P):                                    # 計算正確率
    
    error = 0
    for x in range(len(test_label)):
        if test_label_P.values[x] != test_label.values[x]:
            error = error + 1                                              
    errorRate = error / len(test_label)
    return errorRate, error
             
def main():
    
    train_data_location = 'C:\\Users\\Laptop\\OneDrive\\桌面\\data\\class\\data_mining\\experiment_A\\train_data.csv'
    test_data_location = 'C:\\Users\\Laptop\\OneDrive\\桌面\\data\\class\\data_mining\\experiment_A\\test_data.csv'

    global columns
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness' 
                  , 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    
    print("程式運行中 ．．．．．．")
    train_data, train_label = handleDataset(train_data_location)               # 切割分成 data 與 label 
    train_data_std = standardization(train_data)                               # 標準化數據
    train_data_std_out = outlier(train_data_std)                               # 蓋帽法
    
    test_data, test_label = handleDataset(test_data_location)                  # 切割分成 data 與 label 
    test_data_std = standardization(test_data)
    test_data_std_out = outlier(test_data_std)
    
    k = 15
    test_label_P = KNN(train_label, train_data_std_out, test_data_std_out, k)
    errorRate, error = correct_rate(test_label, test_label_P)
    print("errorRate = " + str(errorRate))
    print("error = " + str(error))
    
main()