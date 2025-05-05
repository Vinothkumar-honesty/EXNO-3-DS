## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
df=pd.read_csv('/Encoding Data (2).csv')
df
```
![3 1](https://github.com/user-attachments/assets/6b333eb4-02ae-4c68-8b2e-8ddaba3662a5)

```
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
pm=['Hot',"Warm","Cold"]
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[['ord_2']])
```
![3 2](https://github.com/user-attachments/assets/0856f7c4-9156-41a3-a068-39339ee25492)

```
df['bo2']=e1.fit_transform(df[['ord_2']])
df
```
![3 3](https://github.com/user-attachments/assets/57b95497-5e49-4954-aa1c-c9bd8175c0f9)

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc 
```
![3 4](https://github.com/user-attachments/assets/cbcc581d-a305-47f7-8169-ab277435c906)

```
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
df2=pd.concat([df2,enc],axis=1)
df2
```
![3 5](https://github.com/user-attachments/assets/6da625e2-fb0a-479b-9106-fb72fa64a767)

```
pd.get_dummies(df2,columns=['nom_0'])
```
![3 6](https://github.com/user-attachments/assets/0372a1dd-e637-4ade-b615-5f2a6cc70c46)

```
pip install --upgrade category_encoders
```
![3 7](https://github.com/user-attachments/assets/4bb02837-ee10-4f97-ab15-16fe2c9d0b21)

```
from category_encoders import BinaryEncoder
df=pd.read_csv('/data (2).csv')
df
```
![3 8](https://github.com/user-attachments/assets/2a8d984e-dd9b-43e8-abce-77355f198941)

```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```
![3 9](https://github.com/user-attachments/assets/a4941376-cfdf-49aa-9c9e-4af1a981f759)

```
from category_encoders import TargetEncoder

te = TargetEncoder()
cc = df.copy()

new = te.fit_transform(X=cc["City"],y=cc["Target"])
new.columns = ["City_TargetEncoded"]
cc = pd.concat([cc, new], axis=1)
cc
```
![3 10](https://github.com/user-attachments/assets/2a58bd1d-2e58-4555-a3f8-c7bd68be3f92)

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv('/Data_to_Transform (1).csv')
df
```
![3 11](https://github.com/user-attachments/assets/c1818ca7-facd-4205-95d7-9662a19ca818)

```
df.skew()
```
![3 12](https://github.com/user-attachments/assets/33567a46-8ea4-458b-9ea8-e73dbe07dfd1)

```
np.log(df['Highly Positive Skew'])
```
![3 13](https://github.com/user-attachments/assets/df2fc2a0-0604-4c5e-a4b6-404a81a4b9c8)

```
np.reciprocal(df['Moderate Positive Skew'])
```
![3 14](https://github.com/user-attachments/assets/2bf29299-ba19-459b-84ff-bc36a7d350d2)

```
np.sqrt(df["Highly Positive Skew"])
```
![3 15](https://github.com/user-attachments/assets/e2abe500-85c8-4d89-ae15-98a8664835af)

```
np.square(df["Highly Positive Skew"])
```
![3 16](https://github.com/user-attachments/assets/f3465a37-ec48-4cd5-afae-347db24a85a9)

```
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![3 17](https://github.com/user-attachments/assets/74226f39-07e8-48d8-b424-1188bbb81db8)

```
df.skew()
```
![3 18](https://github.com/user-attachments/assets/a980b64a-b684-449b-a63c-4a883985cfcf)

```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![3 19](https://github.com/user-attachments/assets/8f9db983-90e3-473e-8078-e6a51cafcf62)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
![3 20](https://github.com/user-attachments/assets/6b2dcef7-5358-4ed2-a7ca-711802eab5b6)

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df['Moderate Negative Skew'],line='45')
plt.show()
```
![3 21](https://github.com/user-attachments/assets/5b3f88f3-4fcb-444b-bbd1-3c3f35a52358)

```
sm.qqplot(np.reciprocal(df['Moderate Negative Skew']),line='45')
plt.show()
```
![3 22](https://github.com/user-attachments/assets/eef2a855-fa65-45d7-9271-dc62c5f3abb2)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df['Moderate Negative Skew'],line='45')
plt.show()
```
![3 23](https://github.com/user-attachments/assets/0a70eb78-cde5-4e84-a89f-8014f0bcce08)

```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df['Highly Negative Skew'],line='45')
plt.show()
```
![3 24](https://github.com/user-attachments/assets/ec439c34-df46-4905-a1ce-38a37bcf5146)

```
sm.qqplot(df['Highly Negative Skew_1'],line='45')
plt.show()
```
![3 25](https://github.com/user-attachments/assets/f7c972a2-3906-4707-b92b-d06f26e13575)

```
dt=pd.read_csv('/titanic_dataset (2).csv')
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=861)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45')
plt.show()
```
![3 26](https://github.com/user-attachments/assets/c0978d7b-9fcf-4f27-b675-6a1b64fff643)

```
sm.qqplot(dt['Age_1'],line='45')
plt.show()
```
![3 27](https://github.com/user-attachments/assets/b8e256d9-d8f9-4d76-b396-cf9bee937767)

# RESULT:
Thus the given data  has been read and perform for Feature Encoding and Transformation process are done.

       
