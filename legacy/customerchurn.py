# import kagglehub
# from kagglehub import KaggleDatasetAdapter

import numpy as np
import pandas as pd
from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder

import pickle

from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb

#đọc dữ liệu từ hai file csv và gộp chúng lại thành một DataFrame duy nhất
base_dir = Path(__file__).resolve().parent
df = pd.read_csv(base_dir / 'customer_churn_dataset-training-master.csv')
# print(df.describe())
# print(df.info())
# print(df.describe(include=[object]))
 #xóa cột CustomerID vì nó không có ý nghĩa trong việc dự đoán churn
df.drop(columns='CustomerID', inplace=True)
#chuyển tên cột thành chữ thường và thay thế khoảng trắng bằng dấu gạch dưới để dễ dàng xử lý dữ liệu
df.columns = [col.lower().replace(' ', '_') for col in df.columns] 
# print(df.shape)
# print(df.isnull().sum())
# print(df[df.isna().any(axis=1)] )
#xóa các hàng có giá trị thiếu
df.dropna(inplace=True)
# print(df.shape)
#chuyển các cột có kiểu dữ liệu object thành kiểu int để dễ dàng xử lý dữ liệu
descrete_col = ['age', 'tenure', 'usage_frequency', 'support_calls', 'payment_delay', 'last_interaction', 'churn']
for col in descrete_col:
    df[col] = df[col].astype(int)
# print(df.info())

# One-hot encode categorical features
print("Original shape:", df.shape)
df_encoded = pd.get_dummies(df, columns=['gender', 'subscription_type', 'contract_length'], drop_first=False, dtype=int)
print("After one-hot encoding shape:", df_encoded.shape)

# Export clean data to CSV
clean_data_path = base_dir / 'clean_data.csv'
df_encoded.to_csv(clean_data_path, index=False)
print(f"\nClean data exported to: {clean_data_path}")
print(f"Columns: {list(df_encoded.columns)}")

# tạo các hàm để trực quan hóa dữ liệu và hiển thị các thống kê cơ bản về phân phối của các đặc trưng trong DataFrame
def make_histogram(df, target_feature, bins = 10, custom_ticks=None, unit='', additional=''):
    plt.figure(figsize=(10, 5))
    plt.hist(df[target_feature], bins=bins)
    if custom_ticks is not None:
        plt.xticks(custom_ticks)
    plt.ylabel('Count')
    plt.xlabel(target_feature)
    plt.title(f"Distribution of {target_feature.lower()}{additional}:\n")
    plt.grid()
    plt.show()
    print(f"Distribution of {target_feature.lower()}{additional}: {df[target_feature].mean():.2f} ± {df[target_feature].median():.2f} {unit}\nMedian: {df[target_feature].median():.2f} {unit}\nMinimum: {df[target_feature].min()} {unit}\nMaximum: {df[target_feature].max()} {unit}\n{df[target_feature].skew():.3f} Skewness\n")

def make_piechart(df, target_feature, additional=''):
    dict_of_val_counts = dict(df[target_feature].value_counts())
    data = list(dict_of_val_counts.values())
    keys = list(dict_of_val_counts.keys())
    
    palette_color = sns.color_palette('bright')
    plt.pie(data, labels=keys, colors=palette_color, autopct='%.0f%%')
    plt.title(f"Distribution of Cutomer's {target_feature}:")
    plt.show()
    print_str = f"Distribution of cutomer's {target_feature.lower()}{additional}:"
    for k, v in zip(keys, data):
        print_str += f"\n{v} {k}"
    print(print_str)

def make_barplot(df, target_feature, custom_ticks=None, unit='', additional=''):
    plt.figure(figsize=(10, 5))
    dict_of_val_counts = dict(df[target_feature].value_counts())
    data = list(dict_of_val_counts.values())
    keys = list(dict_of_val_counts.keys())
    plt.bar(keys, data)
    if custom_ticks is not None:
        plt.xticks(custom_ticks)
    plt.xlabel(f'{target_feature.capitalize()}{additional}')
    plt.ylabel('Frequency')
    plt.title(f"Distribution of cutomer's {target_feature.lower()}{additional}\n")
    plt.grid(axis='y')
    plt.show()
    print(f"Distribution of cutomer's {target_feature.lower()}{additional}: {df[target_feature].mean():.2f} ± {df[target_feature].median():.2f} {unit}\nMedian: {df[target_feature].median():.2f} {unit}\nMinimum: {df[target_feature].min()} {unit}\nMaximum: {df[target_feature].max()} {unit}\n\n{df[target_feature].skew():.3f} Skewness\n")
    
def make_boxplot(df, feature):
    plt.figure(figsize=(10,5))
    sns.boxplot(df, x=feature)
    plt.title(f"Boxplot of {feature}\n")
    plt.xlabel(feature)
    plt.ylabel("Values")
    plt.show()
   
# make_piechart(df, 'gender') 
# make_piechart(df, 'subscription_type')
# make_piechart(df, 'contract_length') 

# filtered = df.copy()
# filtered['churn_category'] = ['Churn' if x == 1.0 else 'Not Churned' for x in df['churn']]
# make_piechart(filtered, 'churn_category')

# make_barplot(df, 'age', custom_ticks=np.arange(0, 66, 5), additional=' (years)', unit='years')
# make_boxplot(df, 'age')

# make_barplot(df, 'tenure', custom_ticks=np.arange(0, 61, 3), additional=' (months)', unit='months')
# make_boxplot(df, 'tenure')

# make_barplot(df, 'usage_frequency', custom_ticks=np.arange(0, 31, 2), unit='times', additional=' (in a month)')
# make_boxplot(df, 'usage_frequency')

# make_barplot(df, 'support_calls', unit='calls', additional=' (in a month)')
# make_boxplot(df, 'support_calls')

# make_barplot(df, 'payment_delay', custom_ticks=np.arange(0, 32, 3), unit='days', additional=' (in days)')
# make_boxplot(df, 'payment_delay')

# make_barplot(df, 'last_interaction', custom_ticks=np.arange(0, 32, 3), unit='days', additional='')
# make_boxplot(df, 'last_interaction')

# make_histogram(df, 'total_spend', bins=25, custom_ticks=np.arange(0, 1001, 100), unit='USD', additional=" on products or services")
# make_boxplot(df, 'total_spend')

# gender_churn = df.groupby(['gender', 'churn']).size().unstack()
#--------------------------------------------------------------------------------------
# X = list(gender_churn.index)
# churn_0 = list(gender_churn.iloc[:, 0])
# churn_1 = list(gender_churn.iloc[:, 1])
  
# X_axis = np.arange(len(X))
  
# plt.bar(X_axis - 0.2, churn_1, 0.4, label = 'Churn')
# plt.bar(X_axis + 0.2, churn_0, 0.4, label = 'Not Churn')
  
# plt.xticks(X_axis, X)
# plt.xlabel('Gender')
# plt.ylabel('Count')
# plt.title("Gender wise churn rate")
# plt.legend(loc='center right')
# plt.grid(axis='y')
# plt.show()
#--------------------------------------------------------------------------------------
# filtered = df.groupby(['payment_delay', 'churn']).size().unstack()
# X = list(filtered.index)
# churn_0 = list(filtered.iloc[:, 0])
# churn_1 = list(filtered.iloc[:, 1])
  
# X_axis = np.arange(len(X))
  
# plt.bar(X_axis - 0.2, churn_1, 0.4, label = 'Churn')
# plt.bar(X_axis + 0.2, churn_0, 0.4, label = 'Not Churn')
  
# plt.xticks(X_axis, X, rotation=90)
# plt.xlabel("Customer payment delays in days")
# plt.ylabel('Count')
# plt.title("Churn rate based on payment delays")
# plt.legend(loc='center right')
# plt.grid(axis='y')
# plt.show()
#--------------------------------------------------------------------------------------
# filtered = df.groupby(['usage_frequency', 'churn']).size().unstack()

# X = list(filtered.index)
# churn_0 = list(filtered.iloc[:, 0])
# churn_1 = list(filtered.iloc[:, 1])
  
# X_axis = np.arange(len(X))
  
# plt.bar(X_axis - 0.2, churn_1, 0.4, label = 'Churn')
# plt.bar(X_axis + 0.2, churn_0, 0.4, label = 'Not Churn')
  
# plt.xticks(X_axis, X, rotation=90)
# plt.xlabel("Customer's company services usage frequency")
# plt.ylabel('Count')
# plt.title("Churn rate based on usage frequency")
# plt.legend(loc='center right')
# plt.grid(axis='y')
# plt.show()
#--------------------------------------------------------------------------------------
# def categorize_age(age):
#     if 0 <= age <= 10:
#         return '0 to 10 months'
#     elif 11 <= age <= 20:
#         return '11 to 20 months'
#     elif 21 <= age <= 30:
#         return '21 to 30 months'
#     elif 31 <= age <= 40:
#         return '31 to 40 months'
#     elif 41 <= age <= 50:
#         return '41 to 50 months'
#     elif 51 <= age <= 60:
#         return '51 to 60 months'
#     else:
#         pass # For nan values

# filtered = df.copy()
# filtered['tenure_segmentation'] = df['tenure'].apply(categorize_age)
# filtered = filtered.groupby(['tenure_segmentation', 'churn']).size().unstack()

# X = list(filtered.index)
# churn_0 = list(filtered.iloc[:, 0])
# churn_1 = list(filtered.iloc[:, 1])
  
# X_axis = np.arange(len(X))
  
# plt.bar(X_axis - 0.2, churn_1, 0.4, label = 'Churn')
# plt.bar(X_axis + 0.2, churn_0, 0.4, label = 'Not Churn')
  
# plt.xticks(X_axis, X, rotation=45)
# plt.xlabel('Tenures')
# plt.ylabel('Count')
# plt.title("Churn rate based on tenures")
# plt.legend(loc='center right')
# plt.grid(axis='y')
# plt.show()
#--------------------------------------------------------------------------------------
# filtered = df.groupby(['support_calls', 'churn']).size().unstack()

# X = list(filtered.index)
# churn_0 = list(filtered.iloc[:, 0])
# churn_1 = list(filtered.iloc[:, 1])
  
# X_axis = np.arange(len(X))
  
# plt.bar(X_axis - 0.2, churn_1, 0.4, label = 'Churn')
# plt.bar(X_axis + 0.2, churn_0, 0.4, label = 'Not Churn')
  
# plt.xticks(X_axis, X, rotation=45)
# plt.xlabel('Customer Support Calls')
# plt.ylabel('Count')
# plt.title("Churn rate based on support calls made by customers")
# plt.legend(loc='center right')
# plt.grid(axis='y')
# plt.show()
#--------------------------------------------------------------------------------------

# filtered = df.groupby(['subscription_type', 'churn']).size().unstack()
# X = list(filtered.index)
# churn_0 = list(filtered.iloc[:, 0])
# churn_1 = list(filtered.iloc[:, 1])
  
# X_axis = np.arange(len(X))
  
# plt.bar(X_axis - 0.2, churn_1, 0.4, label = 'Churn')
# plt.bar(X_axis + 0.2, churn_0, 0.4, label = 'Not Churn')
  
# plt.xticks(X_axis, X, rotation=45)
# plt.xlabel('Subscription Type')
# plt.ylabel('Count')
# plt.title("Churn rate based on subscription type")
# plt.legend(loc='center right')
# plt.grid(axis='y')
# plt.show()
#--------------------------------------------------------------------------------------
# filtered = df.groupby(['contract_length', 'churn']).size().unstack()

# X = list(filtered.index)
# churn_0 = list(filtered.iloc[:, 0])
# churn_1 = list(filtered.iloc[:, 1])
  
# X_axis = np.arange(len(X))
  
# plt.bar(X_axis - 0.2, churn_1, 0.4, label = 'Churn')
# plt.bar(X_axis + 0.2, churn_0, 0.4, label = 'Not Churn')
  
# plt.xticks(X_axis, X, rotation=45)
# plt.xlabel('Contract Length')
# plt.ylabel('Count')
# plt.title("Churn rate based on contract length")
# plt.legend(loc='center right')
# plt.grid(axis='y')
# plt.show()
#--------------------------------------------------------------------------------------
# filtered = df.copy()
# filtered['churn_segment'] = ['Churn' if x == 1.0 else 'Not Churned' for x in df['churn']]

# sns.kdeplot(data=filtered, x="total_spend", hue="churn_segment", multiple="stack")
# plt.show()
#---------------------------------------------------------------------------------------
# independent_features_df = df.select_dtypes(include=['number']).copy().drop(columns=['churn'])
# corr_matrix = independent_features_df.corr()

# # Creating a mask to hide the upper triangle of the heatmap
# mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# plt.figure(figsize=(10, 8))
# sns.set(font_scale=1.2)
# sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", mask=mask)
# plt.title("Independent Features Correlation Heatmap")
# plt.show()
#--------------------------------------------------------------------------------------
# correlation_data = df.select_dtypes(include=['number']).corr().loc[:'last_interaction', 'churn']


# # Create a heatmap
# plt.figure(figsize=(5, 3))
# sns.set(font_scale=1.2)
# sns.heatmap(correlation_data.to_frame(), annot=True, cmap="coolwarm", cbar=True)

# plt.title("Correlation Heatmap between Independent Features and Churn")
# plt.show()
