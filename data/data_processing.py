import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def proccess_train_data(file_path, test_size=0.2, random_state=42):
    """
    Loads Titanic dataset, selects top 5 most correlated features, processes data, then splits into train and validation sets.
    """
    # 讀取數據
    df = pd.read_csv(file_path)

    # **1️⃣ 取前五高相關的變數**
    df = df[['Survived', 'Pclass', 'Fare', 'Sex', 'Embarked', 'Parch']]

    # **2️⃣ 填補缺失值**
    df[['Fare', 'Parch']] = df[['Fare', 'Parch']].apply(lambda x: x.fillna(x.median()))
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # **3️⃣ 類別變數轉換**
    label_encoder = LabelEncoder()
    df['Sex'] = label_encoder.fit_transform(df['Sex'])  # male=1, female=0
    df['Embarked'] = label_encoder.fit_transform(df['Embarked'])  # Embarked 轉換為數值

    # **4️⃣ 分離特徵與標籤**
    features = df[['Pclass', 'Fare', 'Sex', 'Embarked', 'Parch']]
    labels = df['Survived'].values  

    # **5️⃣ 打亂數據**
    np.random.seed(random_state)
    indices = np.arange(len(features))
    np.random.shuffle(indices)

    # **6️⃣ 切分數據**
    split_idx = int(len(features) * (1 - test_size))
    train_data, valid_data = features.iloc[indices[:split_idx]], features.iloc[indices[split_idx:]]
    train_labels, valid_labels = labels[indices[:split_idx]], labels[indices[split_idx:]]

    return train_data, train_labels, valid_data, valid_labels


def proccess_test_data(test_features_path, test_labels_path=None):
    """
    加載測試數據 (test_features) 和 標籤 (test_labels)
    """
    test_features = pd.read_csv(test_features_path)

    # **1️⃣ 選擇前五個相關性最高的變數**
    test_features = test_features[['Pclass', 'Fare', 'Sex', 'Embarked', 'Parch']]

    # **2️⃣ 填補缺失值**
    test_features[['Fare', 'Parch']] = test_features[['Fare', 'Parch']].apply(lambda x: x.fillna(x.median()))
    test_features['Embarked'] = test_features['Embarked'].fillna(test_features['Embarked'].mode()[0])

    # **3️⃣ 轉換類別變數**
    label_encoder = LabelEncoder()
    test_features['Sex'] = label_encoder.fit_transform(test_features['Sex'])  # male=1, female=0
    test_features['Embarked'] = label_encoder.fit_transform(test_features['Embarked'])  # Embarked 類別轉換

    # **4️⃣ 讀取標籤（如果有提供）**
    if test_labels_path:
        test_labels = pd.read_csv(test_labels_path)['Survived'].values  # 只選取 Survived 欄位
    else:
        test_labels = None  # 如果沒有標籤，就設為 None
    
    return test_features, test_labels

'''
train_data	pandas.DataFrame	訓練數據（特徵）
train_labels	numpy.ndarray	訓練數據的標籤(0: 死亡, 1: 生還）
valid_data	pandas.DataFrame	驗證數據（特徵）
valid_labels	numpy.ndarray	驗證數據的標籤(0: 死亡, 1: 生還）
'''


