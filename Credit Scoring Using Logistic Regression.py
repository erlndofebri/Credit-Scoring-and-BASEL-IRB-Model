#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Matplolib default parameters
from matplotlib import rcParams
plt.style.use('default')
plt.rcParams['axes.prop_cycle'] = plt.rcParamsDefault['axes.prop_cycle']

# import warning
import warnings
warnings.filterwarnings('ignore')

# Set option max columns display
pd.set_option('max_columns', 150)
pd.set_option('max_rows', 150)


# # Dataset Overview

# In[2]:


df = pd.read_csv('loan_data_2007_2014.csv')
df.head()


# In[6]:


print(f'This dataset contains of {df.shape[0]} rows and {df.shape} columns')


# In[20]:


# show info

list_item = []
for col in df.columns:
    list_item.append([col, df[col].dtype, df[col].isna().sum(), 100*df[col].isna().sum()/len(df[col]), df[col].nunique(), df[col].unique()[:4]])
desc_df = pd.DataFrame(data=list_item, columns='feature data_type null_num null_pct unique_num unique_sample'.split())
desc_df


# 1. There are 2 features that has only 1 value (we will **drop** it all), those are :
# * `policy_code`
# * `application_type`
# 
# 2. **Convert** to Datetime Features:
# * `last_pymnt_d`, `next_pymnt_d`, `last_credit_pull_d`, `earliest_cr_line`
# * `term` (to integer values)
# 
# 3. We will **convert** this `emp_length` to integer
# 
# 4. We will do **feature engineering** on target feature `loan_status`
# 
# 5. We will **drop** categorical feature that unused in modeling, those are:
# * `url`, 
# * `desc`, 
# * `emp_title` (this feature have too many categorical unique values)
# * `Unnamed: 0` because it's index, and we already have it
# * `title` because it has too many unique values and not really importance feature
# * `id` and `member_id`
# * `sub_grade` bcs we already have `grade` feature
# * `zip_code` and `addr_state`
# * `issue_d` bcs we wouldn't know beforehand whether or not a loan would be issued when using our model, so in theory we wouldn't have an issue_date,we'll drop this feature

# ## General Overview

# In[4]:


# check missing values
df.isnull().sum()/len(df)*100


# There are some feature with <= 15% missing values, probably we will do imputation if possible. And if it's not possible, we will drop the missing rows or the whole feature depend on the condition. Those features are:
# * `emp_title`
# * `emp_length`
# * `tot_coll_amt`
# * `tot_cur_bal`
# * `acc_now_delinq`
# * `total_rev_hi_lim`
# 
# And there are huge missing values on some feature (> 20% missing values), high probably we will drop those features. Those are:
# * `mths_since_last_delinq`
# * `mths_since_last_record`
# * `next_pymnt_d`
# * `mths_since_last_major_derog` 
# 
# And those are that have 100% missing values :
# * `annual_inc_joint`, `dti_joint`, `verification_status_joint`, `open_acc_6m`, `open_il_6m`, `open_il_12m`, `open_il_24m`, `mths_since_rcnt_il`, `total_bal_il`, `il_util`, `open_rv_12m`, `open_rv_24m`, `max_bal_bc`, `all_util`, `inq_fi`,`total_cu_tl`,`inq_last_12m`

# In[7]:


# check duplicate
df.duplicated().sum()


# We have no duplicated values

# ## General Preprocessing

# ### Handling Missing Values

# in this section, we will only drop features that have 100% missing values and keep the rest. And we will do missing values handling in the next section while we understanding data

# In[3]:


# define feature that we will drop (this list is a result from the analysis above)
drop_list = ['annual_inc_joint', 'dti_joint', 'verification_status_joint', 'open_acc_6m', 'open_il_6m',
             'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'il_util', 
             'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util', 'inq_fi','total_cu_tl',
             'inq_last_12m', 'Unnamed: 0', 'policy_code', 'application_type', 'url', 'desc',
             'emp_title', 'title', 'member_id', 'id', 'sub_grade', 'zip_code', 'addr_state', 'issue_d']

# drop features
df = df.drop(drop_list, axis=1)

# print length of drop_list and check data shape
print(f'Number of features dropped : {len(drop_list)}')
print(f'For now, this dataset contains of {df.shape[0]} rows and {df.shape[1]} features')


# ### Feature Engineering

# #### Target Feature : `loan_status`

# In[49]:


df['loan_status'].value_counts()


# In[4]:


# make a good_loan list
good_loan = ['Current', 'Fully Paid', 'In Grace Period']

# categorizing whether its a good loan or bad loan
df['loan_status'] = np.where(df['loan_status'].isin(good_loan), 1, 0)
df['loan_status'].value_counts()/len(df)*100


# #### Date Time Features : `last_pymnt_d`

# Assume we're now on 2016, 1 st June

# In[ ]:


# check value_counts
# df['last_pymnt_d'].value_counts()


# In[5]:


# format data type
df['last_pymnt_d'] = pd.to_datetime(df['last_pymnt_d'], format = '%b-%y')

# we will count the distance of last payment month until today (June 1st, 2016)
df['last_pymnt_d'] = round(pd.to_numeric((pd.to_datetime('2016-06-01') - df['last_pymnt_d']) / np.timedelta64(1, 'M')))
df['last_pymnt_d'].describe()


# #### Date Time Features : `next_pymnt_d`

# In[ ]:


# df['next_pymnt_d'].value_counts()


# In[6]:


# format data type
df['next_pymnt_d'] = pd.to_datetime(df['next_pymnt_d'], format = '%b-%y')

# we will count the distance of last payment month until today (June 1st, 2016)
df['next_pymnt_d'] = round(pd.to_numeric((pd.to_datetime('2016-06-01') - df['next_pymnt_d']) / np.timedelta64(1, 'M')))
df['next_pymnt_d'].describe()


# #### Date Time Features : `last_credit_pull_d`

# In[ ]:


# df['last_credit_pull_d'].value_counts()


# In[7]:


# format data type
df['last_credit_pull_d'] = pd.to_datetime(df['last_credit_pull_d'], format = '%b-%y')

# we will count the distance of last payment month until today (June 1st, 2016)
df['last_credit_pull_d'] = round(pd.to_numeric((pd.to_datetime('2016-06-01') - df['last_credit_pull_d']) / np.timedelta64(1, 'M')))
df['last_credit_pull_d'].describe()


# #### Date Time Features : `earliest_cr_line`

# In[ ]:


# df['earliest_cr_line'].value_counts()


# In[8]:


# drop missing values
df = df.dropna(subset=['earliest_cr_line'])
df['earliest_cr_line'].isnull().sum()


# In[9]:


new = df['earliest_cr_line'].str.split('-', n=1, expand=True)
df['earliest_cr_line_month'] = new[0]
df['earliest_cr_line_year'] = new[1]


# In[10]:


new_list = []

for word in df['earliest_cr_line_year']:
    if word == '00':
        new_list.append('20'+word)
    elif word == '01':
        new_list.append('20'+word)
    elif word == '02':
        new_list.append('20'+word)
    elif word == '03':
        new_list.append('20'+word)
    elif word == '04':
        new_list.append('20'+word)
    elif word == '05':
        new_list.append('20'+word)
    elif word == '06':
        new_list.append('20'+word)
    elif word == '07':
        new_list.append('20'+word)
    elif word == '08':
        new_list.append('20'+word)
    elif word == '09':
        new_list.append('20'+word)
    elif word == '10':
        new_list.append('20'+word)
    elif word == '11':
        new_list.append('20'+word)
    else:
        new_list.append('19'+word)
        
df['year_year'] = new_list
df['year_year'].head()


# In[11]:


# assign to original features
df['earliest_cr_line'] = df['earliest_cr_line_month'] + ' ' + df['year_year']


# In[12]:


# drop temporary features
df.drop(['earliest_cr_line_month', 'earliest_cr_line_year', 'year_year'], inplace=True, axis=1)


# In[13]:


# format data type
df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'], format = '%b %Y')

# we will count the distance of last payment month until today (June 1st, 2016)
df['earliest_cr_line'] = round(pd.to_numeric((pd.to_datetime('2016-06-01') - df['earliest_cr_line']) / np.timedelta64(1, 'M')))
df['earliest_cr_line'].describe()


# #### Date Time Features : `term` to integer

# In[65]:


df['term'].value_counts()


# In[14]:


df['term'] = df['term'].apply(lambda term: int(term[:3])) # filter for first 2 character
df['term'].value_counts()


# #### Categorical Feature : `emp_length`

# In[22]:


# plot emp_length
emp_length_order = [ '< 1 year','1 year','2 years','3 years','4 years',
                     '5 years','6 years','7 years','8 years','9 years','10+ years']

plt.figure(figsize=(12,4))
sns.countplot(x='emp_length', data=df, order=emp_length_order,
              hue='loan_status', palette='viridis')


# In[24]:


emp_co = df[df['loan_status']==0].groupby('emp_length').count()['loan_status'] # charge off
emp_fp = df[df['loan_status']==1].groupby('emp_length').count()['loan_status'] # fully paid
emp_len = emp_fp/emp_co
emp_len


# In[25]:


emp_len.plot(kind='bar')


# bacause it has very similar ratio each employment year, we will drop `emp_length`

# In[15]:


df.drop('emp_length', axis=1, inplace=True)


# # Exploratory Data Analysis

# In[27]:


# copy data
eda = df.copy()

# define categorical and numerical features
num = df.select_dtypes(include='number').columns
cat = df.select_dtypes(include='object').columns

# print number of feature each type
print(f'number of categorical features : {len(cat)}')
print(f'number of numeric features : {len(num)}')


# In[194]:


# print shape
print(f'now this dataset contains of {eda.shape[0]} rows and {eda.shape[1]} features')


# ## Univariate Analysis

# ### Descriptive Statistic

# In[195]:


# descriptive statistic for numeric features
eda[num].describe().T


# In[196]:


# descriptive statistic for categorical features
df.describe(exclude=[np.number]).T


# ### Numeric Feature Analysis

# In[197]:


len(num)


# In[28]:


# numeric features analysis
plt.figure(figsize=(24,28))
for i in range(0,len(num)):
    plt.subplot(10,4,i+1)
    sns.boxplot(x=eda[num[i]], palette='viridis')
    plt.title(num[i], fontsize=20)
    plt.xlabel(' ')
    plt.tight_layout()


# In[29]:


# numeric features analysis
plt.figure(figsize=(24,28))
for i in range(0,len(num)):
    plt.subplot(10,4,i+1)
    sns.kdeplot(x=eda[num[i]], palette='viridis', shade=True)
    plt.title(num[i], fontsize=20)
    plt.xlabel(' ')
    plt.tight_layout()


# In[30]:


# numeric features analysis
plt.figure(figsize=(24,28))
for i in range(0,len(num)):
    plt.subplot(10,4,i+1)
    sns.violinplot(x=eda[num[i]], palette='viridis', shade=True)
    plt.title(num[i], fontsize=20)
    plt.xlabel(' ')
    plt.tight_layout()


# ### Categorical Feature Analysis

# In[204]:


len(cat)


# In[31]:


# numeric features analysis
plt.figure(figsize=(14,8))
for i in range(0,len(cat)):
    plt.subplot(2,3,i+1)
    sns.countplot(y=eda[cat[i]], palette='viridis')
    plt.title(cat[i])
    plt.xlabel(' ')
    plt.tight_layout()


# ## Bivariate Analysis

# ### Numerical Feature Analysis

# In[32]:


# numeric features analysis
plt.figure(figsize=(24,28))
for i in range(0,len(num)):
    plt.subplot(10,4,i+1)
    sns.kdeplot(x=eda[num[i]], palette='viridis', shade=True, hue=eda['loan_status'])
    plt.title(num[i], fontsize=20)
    plt.xlabel(' ')
    plt.tight_layout()


# ### Categorical Feature Analysis

# In[33]:


# numeric features analysis
plt.figure(figsize=(14,8))
for i in range(0,len(cat)):
    plt.subplot(2,3,i+1)
    sns.countplot(y=eda[cat[i]], palette='viridis', hue=eda['loan_status'])
    plt.title(cat[i])
    plt.xlabel(' ')
    plt.tight_layout()


# * seems `pymnt_plan` only dominated by single value `n`, we will drop it

# ### Correlation Heatmap

# In[217]:


plt.figure(figsize=(22,22))
sns.heatmap(df.corr(), cmap='Blues', annot=True, fmt='.2f')


# There are some high correlated independent features, we need to remove one of correlated features to prevent multicolinearity, we will use 70% correlation value as a treshold

# # Data Preprocessing

# ## Droping Highly Correlated Features

# In[16]:


# create a square matrix with dimensions equal to the number of features.
cor_matrix = df.corr().abs()

# we are selecting the upper traingular (doesn't matter choose upper/lower, its the same result)
upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))

# create drop list for highly correlated features, we set treshold = 0.7
drop_list = [column for column in upper_tri.columns if any(upper_tri[column] > 0.7)]

# show drop_list
drop_list


# In[35]:


# drop
df = df.drop(drop_list, axis=1)

# print shape
print(f'for now, our dataset contains of {df.shape[0]} rows and {df.shape[1]} features')


# In[36]:


df.corr()['loan_status'].sort_values()


# In[37]:


df.corr()['loan_status'].sort_values().plot(kind='bar')
plt.title('Features Correlation to Target Feature')


# ## Drop Feature that are Dominated by One Value

# In[38]:


# drop feature
df = df.drop('pymnt_plan', axis=1)

# print shape
print(f'for now, our dataset contains of {df.shape[0]} rows and {df.shape[1]} features')


# ## Handling Missing Values

# In[224]:


df.isnull().sum()/len(df)*100


# We will drop high missing values features :
# * `mths_since_last_delinq` and `mths_since_last_record`
# 
# we will impute these feature :
# * `last_pymnt_d` with median
# * `tot_cur_bal` with 0
# * `tot_coll_amt` with 0
# * `revol_util` with 0
# * `collections_12_mths_ex_med` with 0

# In[39]:


# drop features
df = df.drop(['mths_since_last_delinq','mths_since_last_record'],axis=1)

# fill missing values
df['last_pymnt_d'] = df['last_pymnt_d'].fillna(6).astype('int') # median = 6, convert to integer
df['tot_cur_bal'] = df['tot_cur_bal'].fillna(0)
df['tot_coll_amt'] = df['tot_coll_amt'].fillna(0)
df['revol_util'] = df['revol_util'].fillna(0)
df['collections_12_mths_ex_med'] = df['collections_12_mths_ex_med'].fillna(0)


# In[40]:


# check missing values
df.isna().sum()


# # Feature Selection Using Weight of Evidence & Information Value

# In[19]:


#df.to_csv('data_bersih_sebelum_woe.csv')
#df=pd.read_csv('data_bersih_sebelum_woe.csv', index_col=0)


# In[20]:


df.head()


# In[21]:


print(f'this dataset contains of {df.shape[0]} rows and {df.shape[1]} columns')


# In[193]:


from matplotlib import image

img = image.imread('IV.png')
plt.title('Information Value Rule of Thumb', fontsize=15)
plt.imshow(img)
plt.axis('off')


# we will use this rule of thumb to determine which features that will be used on modeling

# ## Categorical Features

# ### WoE : `grade`

# In[47]:


# make probability analysis
grade = df.groupby('grade').agg(num_observation=('loan_status','count'),
                                good_loan_prob=('loan_status','mean')).reset_index()
grade


# In[49]:


# find grade proportion
grade['grade_proportion'] = grade['num_observation']/grade['num_observation'].sum()

# find number of good loan
grade['num_good_loan'] = grade['grade_proportion'] * grade['num_observation']

# find number of bad loan
grade['num_bad_loan'] = (1-grade['grade_proportion']) * grade['num_observation']

# find good loan proportion
grade['good_loan_prop'] = grade['num_good_loan'] / grade['num_good_loan'].sum()

# find bad loan proportion
grade['bad_loan_prop'] = grade['num_bad_loan'] / grade['num_bad_loan'].sum()

# find Weight of Evidence
grade['weight_of_evidence'] = np.log(grade['good_loan_prop'] / grade['bad_loan_prop'])

# sort values by weight of evidence
grade = grade.sort_values('weight_of_evidence').reset_index(drop=True)
grade


# In[50]:


# find difference each good loan proportion
grade['good_loan_prop_diff'] = grade['good_loan_prop'].diff().abs()

# find difference each weight of evidence
grade['diff_woe'] = grade['weight_of_evidence'].diff().abs()
grade


# In[52]:


# find information value
grade['information_value'] = (grade['good_loan_prop']-grade['bad_loan_prop']) * grade['weight_of_evidence']
grade['information_value'] = grade['information_value'].sum()

# show
grade


# In[69]:


plt.figure(figsize=(8, 4))
sns.lineplot(x='grade', y='weight_of_evidence', data=grade,
             marker='o', linestyle='--', color='blue')
plt.title(str('Weight of Evidence by ' + grade.columns[0]))


# In[9]:


# Automate the Code

def woe(df, feature_name):
    # make probability analysis
    feature_name = df.groupby(feature_name).agg(num_observation=('loan_status','count'),
                                                good_loan_prob=('loan_status','mean')).reset_index()
    
    # find grade proportion
    feature_name['grade_proportion'] = feature_name['num_observation']/feature_name['num_observation'].sum()
    
    # find number of good loan
    feature_name['num_good_loan'] = feature_name['grade_proportion'] * feature_name['num_observation']

    # find number of bad loan
    feature_name['num_bad_loan'] = (1-feature_name['grade_proportion']) * feature_name['num_observation']

    # find good loan proportion
    feature_name['good_loan_prop'] = feature_name['num_good_loan'] / feature_name['num_good_loan'].sum()

    # find bad loan proportion
    feature_name['bad_loan_prop'] = feature_name['num_bad_loan'] / feature_name['num_bad_loan'].sum()

    # find Weight of Evidence
    feature_name['weight_of_evidence'] = np.log(feature_name['good_loan_prop'] / feature_name['bad_loan_prop'])

    # sort values by weight of evidence
    feature_name = feature_name.sort_values('weight_of_evidence').reset_index(drop=True)
    
    # find difference each good loan proportion
    feature_name['good_loan_prop_diff'] = feature_name['good_loan_prop'].diff().abs()

    # find difference each weight of evidence
    feature_name['diff_woe'] = feature_name['weight_of_evidence'].diff().abs()
    
    # find information value
    feature_name['information_value'] = (feature_name['good_loan_prop']-feature_name['bad_loan_prop']) * feature_name['weight_of_evidence']
    feature_name['information_value'] = feature_name['information_value'].sum()
    
    return feature_name


# In[10]:


# build plot function
def plot_woe(df, xlabel_rotation=0):
    plt.figure(figsize=(8,4))
    sns.pointplot(x=df.columns[0], y='weight_of_evidence', data=df, marker='o', linestyle='--', color='blue')
    plt.title(str('Weight of Evidence by ' + df.columns[0]))
    plt.xticks(rotation = xlabel_rotation)


# In[87]:


# try find weight of evidence and information value
woe(df,'grade')


# In[414]:


# plot
plot_woe(woe(df,'grade'))


# seems `grade:A` and `grade:D` have similar weight of evidence, but for now, we'll keep it as it is

# ### WoE : `home_ownership`

# In[90]:


df['home_ownership'].value_counts()


# because `home_ownership` has some values that have little frequency, we will combine `ANY`, `NONE`, and `OTHER` into 1 value --> `OTHER`

# In[94]:


df['home_ownership'] = np.where(df['home_ownership']=='ANY','OTHER',
                       np.where(df['home_ownership']=='NONE','OTHER',df['home_ownership']))


# In[95]:


df['home_ownership'].value_counts()


# In[96]:


woe(df, 'home_ownership')


# In[235]:


plot_woe(woe(df, 'home_ownership'))


# every values in `home_ownership` differ by looking at their weight of evidence

# ### WoE : `verification_status`

# In[98]:


woe(df, 'verification_status')


# In[236]:


plot_woe(woe(df, 'verification_status'))


# * every values in `verification_status` differ by looking at their weight of evidence 
# * BUT it has very low information value `0.007677`, and we will drop this feature (later)

# ### WoE : `purpose`

# In[102]:


woe(df,'purpose')


# In[237]:


plot_woe(woe(df,'purpose'), 90)


# there are some similar values on this feature, and we will join as:
# * we will join value `purpose:renewable_energy__educational`
# * we will join value `purpose:house__wedding__vacation__moving`
# * we will join value `purpose:medical__car__small_business__major_purpose`
# * we will join value `purpose:other__home_improvement`
# 
# BUT as a consideration, because it has 1.4 Information value, it's considered as suspicius. we will consider this feature will be dropped or not later

# In[238]:


# plot again after combining some values
plot_woe(woe(df,'purpose').iloc[[1,5,9,11],:], 90)


# ### WoE : `initial_list_status`

# In[113]:


woe(df,'initial_list_status')


# In[239]:


plot_woe(woe(df,'initial_list_status'))


# ## Numeric Features

# In[23]:


# make a function to show data distribution and info
def dist(feature):
    plt.figure(figsize=(4,2))
    sns.violinplot(df[feature],color='orange')
    print('number of unique values :',df[feature].nunique())
    print('Distribution :')
    print(df[feature].describe().T)


# In[26]:


# build new plot function
def plot_num_woe(df,rot=0):
    plt.figure(figsize=(8,4))
    sns.pointplot(x=df.columns[0], y='weight_of_evidence',data=df, color='orange')
    plt.title(str('Weight of Evidence by ' + df.columns[0]))
    plt.xticks(rotation = rot)
    plt.xlabel(' ')


# In[5]:


# make a copy data
data_process = df.copy()


# ### WoE : `loan_amnt`

# In[170]:


dist('loan_amnt')


# we will implement fine-classing using pandas cut method, we split feature into 10 categories

# In[172]:


# fine classing = 10 class
data_process['loan_amnt_fc'] = pd.cut(df['loan_amnt'], 10)


# In[173]:


woe(data_process,'loan_amnt_fc')


# In[240]:


plot_num_woe(woe(data_process,'loan_amnt_fc'),90)


# we will bin these values:
# * `(10850.0, 14300.0]__ (14300.0, 17750.0]__ (17750.0, 21200.0]`
# * `(21200.0, 24650.0]__ (24650.0, 28100.0]`

# ### WoE : `term`

# In[175]:


data_process['term'].value_counts()


# In[176]:


# we will implement label encoding for this feature as 36=0, 60=1
data_process['term'] = np.where(data_process['term'] == 36,0,1)
data_process['term'].value_counts()


# we will run label encoding `term` feature to the original data later

# In[177]:


woe(data_process,'term')


# In[241]:


plot_num_woe(woe(data_process,'term'))


# it has high information value, we will drop `term`

# ### WoE : `int_rate`

# In[184]:


dist('int_rate')


# we will run fine-classing `int_rate` values into 25 categorical values

# In[290]:


# fine classing = 10 class
data_process['int_rate_fc'] = pd.cut(df['int_rate'], 10)


# In[291]:


woe(data_process,'int_rate_fc')


# In[292]:


plot_num_woe(woe(data_process,'int_rate_fc'),90)


# we will bin these values :
# * `(7.484, 9.548]__ (9.548, 11.612]`
# * `(11.612, 13.676]__ (13.676, 15.74]`

# ### WoE : `annual_inc`

# In[247]:


df['annual_inc'] = df['annual_inc'].astype('int')


# In[248]:


dist('annual_inc')


# In[506]:


# fine classing = 4 class
data_process['annual_inc_fc'] = pd.cut(df['int_rate'], 4)


# In[507]:


woe(data_process,'annual_inc_fc')


# In[508]:


plot_num_woe(woe(data_process,'annual_inc_fc'),90)


# we will bin `(9.548, 13.676]__ (13.676, 17.804]`

# ### WoE : `dti`

# In[269]:


dist('dti')


# In[274]:


# fine classing = 10 class
data_process['dti_fc'] = pd.cut(df['dti'], 10)


# In[275]:


woe(data_process,'dti_fc')


# In[276]:


plot_num_woe(woe(data_process,'dti_fc'),90)


# we will bin these value:
# * `(7.998, 11.997]__ (11.997, 15.996]__ (15.996, 19.995]__ (19.995, 23.994]`

# ### WoE : `delinq_2yrs`

# In[278]:


dist('delinq_2yrs')


# In[48]:


# we will encode this feature, if values = 0 return 0, if its greater than 0 return 1, if > 5, return 2
data_process['delinq_2yrs_fc'] = np.where(data_process['delinq_2yrs'] > 3, 3,
                                 np.where(data_process['delinq_2yrs'] == 2, 2,
                                 np.where(data_process['delinq_2yrs'] == 1,1,0)))

data_process['delinq_2yrs_fc'].value_counts()


# In[72]:


# show
woe(data_process,'delinq_2yrs_fc')


# In[283]:


plot_num_woe(woe(data_process,'delinq_2yrs_fc'),90)


# we will drop `delinq_2yrs_fc` because it has high information value greater than 0.5, it's considered as suspicious

# ### WoE : `earliest_cr_line`

# In[294]:


dist('earliest_cr_line')


# In[311]:


# fine classing = 5 class
data_process['earliest_cr_line_fc'] = pd.cut(df['earliest_cr_line'], 5)

# show
woe(data_process,'earliest_cr_line_fc')


# In[312]:


plot_num_woe(woe(data_process,'earliest_cr_line_fc'),90)


# ### WoE : `inq_last_6mths`

# In[301]:


dist('inq_last_6mths')


# In[31]:


data_process['inq_last_6mths_fc'] = np.where(data_process['inq_last_6mths'] == 0,0,
                                    np.where((data_process['inq_last_6mths'] > 0)&(data_process['inq_last_6mths'] <=3),1,
                                    np.where((data_process['inq_last_6mths']>3)&(data_process['inq_last_6mths']<=6),2,
                                    np.where((data_process['inq_last_6mths']>6)&(data_process['inq_last_6mths']<=9),3,4))))

data_process['inq_last_6mths_fc'].value_counts()


# In[32]:


# show
woe(data_process,'inq_last_6mths_fc')


# In[33]:


plot_num_woe(woe(data_process,'inq_last_6mths_fc'))


# ### WoE : `open_acc`

# In[308]:


dist('open_acc')


# In[342]:


# fine classing = 5 class
data_process['open_acc_fc'] = pd.cut(df['open_acc'], 10)

# show
woe(data_process,'open_acc_fc')


# In[343]:


plot_num_woe(woe(data_process,'open_acc_fc'),90)


# ### WoE : `pub_rec`

# In[ ]:





# In[317]:


dist('pub_rec')


# In[12]:


# fine classing = 5 class
data_process['pub_rec_fc'] = pd.cut(df['pub_rec'], 5)

# show
woe(data_process,'pub_rec_fc')


# In[323]:


plot_num_woe(woe(data_process,'pub_rec_fc'),90)


# we will drop `pub_rec_fc`, it's sus to have such a high information value

# ### WoE : `revol_bal`

# In[324]:


dist('revol_bal')


# In[23]:


data_process['revol_bal_fc'] = np.where((data_process['revol_bal']>=0)&(data_process['revol_bal']<=5000),0,
                               np.where((data_process['revol_bal']>5000)&(data_process['revol_bal']<=10000),1,
                               np.where((data_process['revol_bal']>10000)&(data_process['revol_bal']<=15000),2,3)))
                               
                            
data_process['revol_bal_fc'].value_counts()


# In[24]:


# show
woe(data_process,'revol_bal_fc')


# In[28]:


plot_num_woe(woe(data_process,'revol_bal_fc'))


# ### WoE : `revol_util`

# In[327]:


dist('revol_util')


# In[35]:


data_process['revol_util_fc'] = np.where((data_process['revol_util']>=0)&(data_process['revol_util']<=20),0,
                                np.where((data_process['revol_util']>20)&(data_process['revol_util']<=40),1,
                                np.where((data_process['revol_util']>40)&(data_process['revol_util']<=60),2,
                                np.where((data_process['revol_util']>60)&(data_process['revol_util']<=80),3,4))))

data_process['revol_util_fc'].value_counts()


# In[36]:


# show
woe(data_process,'revol_util_fc')


# In[37]:


plot_num_woe(woe(data_process,'revol_util_fc'))


# ### WoE : `total_acc`

# In[329]:


dist('total_acc')


# In[350]:


# fine classing = 6 class
data_process['total_acc_fc'] = pd.cut(df['total_acc'], 6)

# show
woe(data_process,'total_acc_fc')


# In[351]:


plot_num_woe(woe(data_process,'total_acc_fc'),90)


# ### WoE : `out_prncp`

# In[352]:


dist('out_prncp')


# In[59]:


data_process['out_prncp_fc'] = np.where((data_process['out_prncp']>=0)&(data_process['out_prncp']<=1000),0,
                               np.where((data_process['out_prncp']>1000)&(data_process['out_prncp']<=10000),1,
                               np.where((data_process['out_prncp']>10000)&(data_process['out_prncp']<=17000),2,3)))

data_process['out_prncp_fc'].value_counts()


# In[60]:


# show
woe(data_process,'out_prncp_fc')


# ### WoE : `total_rec_late_fee`

# In[359]:


dist('total_rec_late_fee')


# In[64]:


data_process['total_rec_late_fee_fc'] = np.where(data_process['total_rec_late_fee']==0,0,1)
data_process['total_rec_late_fee_fc'].value_counts()


# In[65]:


# show
woe(data_process,'total_rec_late_fee_fc')


# we will drop `total_rec_late_fee`, its sus

# ### WoE : `recoveries`

# In[365]:


dist('recoveries')


# In[68]:


# fine classing = 5 class
data_process['recoveries_fc'] = pd.cut(data_process['recoveries'], 5)

# show
woe(data_process,'recoveries_fc')


# we will drop `recoveries`, its sus

# ### WoE : `last_pymnt_d`

# In[371]:


dist('last_pymnt_d')


# In[ ]:


13,6,15,30


# In[81]:


data_process['last_pymnt_d_fc'] = np.where(data_process['last_pymnt_d']==5,0,
                                  np.where((data_process['last_pymnt_d']>5)&(data_process['last_pymnt_d']<=7),1,
                                  np.where((data_process['last_pymnt_d']>7)&(data_process['last_pymnt_d']<=9),2,
                                  np.where((data_process['last_pymnt_d']>9)&(data_process['last_pymnt_d']<=15),3,4
                                  ))))

data_process['last_pymnt_d_fc'].value_counts()


# In[82]:


# show
woe(data_process,'last_pymnt_d_fc')


# In[83]:


plot_num_woe(woe(data_process,'last_pymnt_d_fc'))


# ### WoE: `collections_12_mths_ex_med`

# In[380]:


dist('collections_12_mths_ex_med')


# In[86]:


# fine classing = 5 class
data_process['collections_12_mths_ex_med_fc'] = pd.cut(df['collections_12_mths_ex_med'], 5)

# show
woe(data_process,'collections_12_mths_ex_med_fc')


# we will drop this `collections_12_mths_ex_med`

# ### WoE : `acc_now_delinq`

# In[384]:


dist('acc_now_delinq')


# In[90]:


# fine classing = 5 class
data_process['acc_now_delinq_fc'] = pd.cut(df['acc_now_delinq'], 5)

# show
woe(data_process,'acc_now_delinq_fc')


# we will drop `acc_now_delinq`

# ### WoE : `tot_coll_amt`

# In[388]:


dist('tot_coll_amt')


# In[93]:


# fine classing = 5 class
data_process['tot_coll_amt_fc'] = pd.cut(df['tot_coll_amt'], 5)

# show
woe(data_process,'tot_coll_amt_fc')


# we will drop `tot_coll_amt`

# ### WoE : `tot_cur_bal`

# In[393]:


# fine classing = 5 class
data_process['tot_cur_bal_fc'] = pd.cut(df['tot_coll_amt'], 5)

# show
woe(data_process,'tot_cur_bal_fc')


# we will drop `tot_cur_bal`

# ## Summary

# In[95]:


print(f'for now, dataset contains of {df.shape[0]} rows and {df.shape[1]} columns')


# ### Drop List

# we will drop these features :
# * `verification_status`
# * `purpose`
# * `delinq_2yrs`
# * `pub_rec`
# * `total_rec_late_fee`
# * `recoveries`
# * `collections_12_mths_ex_med`
# * `acc_now_delinq`
# * `tot_coll_amt`
# * `tot_cur_bal`

# In[22]:


drop_list = ['verification_status', 'purpose', 'delinq_2yrs', 'pub_rec', 
             'total_rec_late_fee','recoveries', 'collections_12_mths_ex_med', 
             'acc_now_delinq','tot_coll_amt','tot_cur_bal']

print('number of features that we will drop :',len(drop_list))


# ### Binning List

# we will bin these features:
# * `grade`
# * `loan_amnt`
# * `annual_inc`
# * `dti`

# ## Feature Encoding

# In[23]:


# copy data
data = df.copy()


# In[24]:


# drop unused features
data = data.drop(drop_list, axis=1)
print(f'for now, dataset contains of {data.shape[0]} rows and {data.shape[1]} columns')


# ### Categorical Feature Encoding

# In[25]:


cat = data.select_dtypes(include='object').columns
num = data.select_dtypes(include='number').columns
cat


# In[26]:


data['home_ownership'] = np.where(data['home_ownership']=='ANY','OTHER',
                         np.where(data['home_ownership']=='NONE','OTHER',data['home_ownership']))


# In[27]:


# we will drop_first later manually
cat_dummies = pd.get_dummies(data[cat])
cat_dummies.head()


# In[28]:


# binning grade A and grade D
cat_dummies['grade_AD'] = cat_dummies['grade_A'] + cat_dummies['grade_D']

# drop grade A and D, also drop grade G in exchange drop_first method
cat_dummies = cat_dummies.drop(['grade_A','grade_D','grade_G'],axis=1)

# drop one encoded feature each features in exhage drop_first method
cat_dummies = cat_dummies.drop(['home_ownership_OTHER', 'initial_list_status_w'],axis=1)

# show
cat_dummies.head(3)


# In[29]:


# shift column 'grade_AD' to first position
first_column = cat_dummies.pop('grade_AD')
  
# insert column using insert(position,column_name,
# first_column) function
cat_dummies.insert(0, 'grade_AD', first_column)


# In[30]:


cat_dummies.head(3)


# ### Numeric Feature Encoding

# In[40]:


# num columns
num = data.select_dtypes(include='number').columns

# define list
manual_encoder_list = ['inq_last_6mths', 'revol_bal', 'revol_util', 'out_prncp', 'last_pymnt_d']
function_encoder_list = num.drop(manual_encoder_list)

# define dataframe for manual and auto encoding feature
num_dummies = data[function_encoder_list]
data_process = data[manual_encoder_list]


# In[35]:


# make a function
def make_bins(df, feature, cut):
    df[feature] = pd.cut(df[feature],cut)
    return df


# In[36]:


# loan amnt
loan_amnt = make_bins(num_dummies, 'loan_amnt',10)
loan_amnt_dum = pd.get_dummies(loan_amnt['loan_amnt'], prefix='loan_amnt')

# int_rate
int_rate = make_bins(num_dummies, 'int_rate',10)
int_rate_dum = pd.get_dummies(int_rate['int_rate'], prefix='int_rate')

# dti
dti = make_bins(num_dummies, 'dti', 10)
dti_dum = pd.get_dummies(dti['dti'], prefix='dti')

# open_acc
open_acc = make_bins(num_dummies,'open_acc',10)
open_acc_dum = pd.get_dummies(open_acc['open_acc'], prefix='open_acc')

# annual_inc
annual_inc = make_bins(num_dummies, 'annual_inc', 4)
annual_inc_dum = pd.get_dummies(annual_inc['annual_inc'], prefix='annual_inc')

# earliest_cr_line
earliest_cr_line = make_bins(num_dummies,'earliest_cr_line',5)
earliest_cr_line_dum = pd.get_dummies(earliest_cr_line['earliest_cr_line'], prefix='earliest_cr_line')

# total_acc
total_acc = make_bins(num_dummies, 'total_acc', 6)
total_acc_dum = pd.get_dummies(total_acc['total_acc'], prefix='total_acc')


# ### Binning Some Values On : `loan_amnt`, `annual_inc`, `dti`

# we will bin these values in `loan_amnt`:
# * `(10850.0, 14300.0]__ (14300.0, 17750.0]__ (17750.0, 21200.0]`
# * `(21200.0, 24650.0]__ (24650.0, 28100.0]`
# 
# we will bin these value in `dti`:
# * `(7.998, 11.997]__ (11.997, 15.996]__ (15.996, 19.995]__ (19.995, 23.994]`

# In[37]:


# loan_amnt
loan_amnt_dum['loan_amnt_(10850.0, 21200.0]'] = sum([loan_amnt_dum['loan_amnt_(10850.0, 14300.0]'],
                                                     loan_amnt_dum['loan_amnt_(14300.0, 17750.0]'],
                                                     loan_amnt_dum['loan_amnt_(17750.0, 21200.0]']])


loan_amnt_dum['loan_amnt_(21200.0, 28100.0]'] = sum([loan_amnt_dum['loan_amnt_(21200.0, 24650.0]'],
                                                     loan_amnt_dum['loan_amnt_(24650.0, 28100.0]']])

# rename column
loan_amnt_dum.rename(columns={'loan_amnt_(465.5, 3950.0]':'loan_amnt_(500, 3950.0]'}, inplace=True)

# drop original features
loan_amnt_dum = loan_amnt_dum.drop(['loan_amnt_(10850.0, 14300.0]',
                                    'loan_amnt_(14300.0, 17750.0]',
                                    'loan_amnt_(17750.0, 21200.0]'], axis=1)

# dti
dti_dum['dti_(7.998, 23.994]'] = sum([dti_dum['dti_(7.998, 11.997]'],
                                      dti_dum['dti_(11.997, 15.996]'],
                                      dti_dum['dti_(15.996, 19.995]'],
                                      dti_dum['dti_(19.995, 23.994]']])

# rename columns
dti_dum.rename(columns={'dti_(-0.04, 3.999]':'dti_(0, 3.999]'}, inplace=True)

# drop original feature
dti_dum = dti_dum.drop(['dti_(7.998, 11.997]',
                        'dti_(11.997, 15.996]',
                        'dti_(15.996, 19.995]',
                        'dti_(19.995, 23.994]'], axis=1)


# In[38]:


# show list
manual_encoder_df = data.copy()
manual_encoder_df = manual_encoder_df[manual_encoder_list]
manual_encoder_df.head(2)


# In[41]:


# inq_last_6mths
manual_encoder_df['inq_last_6mths_(0]'] = np.where(data_process['inq_last_6mths'] == 0,1,0)
manual_encoder_df['inq_last_6mths_(0, 3]'] = np.where((data_process['inq_last_6mths'] > 0)&(data_process['inq_last_6mths'] <=3),1,0)
manual_encoder_df['inq_last_6mths_(3, 6]'] = np.where((data_process['inq_last_6mths']>3)&(data_process['inq_last_6mths']<=6),1,0)
manual_encoder_df['inq_last_6mths_(6, 9]'] = np.where((data_process['inq_last_6mths']>6)&(data_process['inq_last_6mths']<=9),1,0)
manual_encoder_df['inq_last_6mths_(9, 33]'] = np.where(data_process['inq_last_6mths'] > 9,1,0)

# revol_bal
manual_encoder_df['revol_bal_(0, 5000]'] = np.where((data_process['revol_bal']>=0)&(data_process['revol_bal']<=5000),1,0)
manual_encoder_df['revol_bal_(5000, 10000]'] =   np.where((data_process['revol_bal']>5000)&(data_process['revol_bal']<=10000),1,0)
manual_encoder_df['revol_bal_(10000, 15000]'] = np.where((data_process['revol_bal']>10000)&(data_process['revol_bal']<=15000),1,0)
manual_encoder_df['revol_bal_(15000, 250000]'] = np.where(data_process['revol_bal']>15000,1,0)

# revol_util
manual_encoder_df['revol_util_(0, 20]'] = np.where((data_process['revol_util']>=0)&(data_process['revol_util']<=20),1,0)
manual_encoder_df['revol_util_(20, 40]'] = np.where((data_process['revol_util']>20)&(data_process['revol_util']<=40),1,0)
manual_encoder_df['revol_util_(40, 60]'] = np.where((data_process['revol_util']>40)&(data_process['revol_util']<=60),1,0)
manual_encoder_df['revol_util_(60, 80]'] = np.where((data_process['revol_util']>60)&(data_process['revol_util']<=80),1,0)
manual_encoder_df['revol_util_(80, 892]'] = np.where(data_process['revol_util']>80,1,0)

# out_prncp
manual_encoder_df['out_prncp_(0, 1000]'] = np.where((data_process['out_prncp']>=0)&(data_process['out_prncp']<=1000),1,0)
manual_encoder_df['out_prncp_(1000, 10000]'] = np.where((data_process['out_prncp']>=1000)&(data_process['out_prncp']<=10000),1,0)
manual_encoder_df['out_prncp_(10000, 17000]'] = np.where((data_process['out_prncp']>=10000)&(data_process['out_prncp']<=17000),1,0)
manual_encoder_df['out_prncp_(17000, 32160]'] = np.where(data_process['out_prncp']>17000,1,0)

# last_pymnt_d
manual_encoder_df['last_pymnt_d_(5]'] = np.where(data_process['last_pymnt_d']==5,1,0)
manual_encoder_df['last_pymnt_d_(5, 7]'] = np.where((data_process['last_pymnt_d']>5)&(data_process['last_pymnt_d']<=7),1,0)
manual_encoder_df['last_pymnt_d_(7, 9]'] = np.where((data_process['last_pymnt_d']>7)&(data_process['last_pymnt_d']<=9),1,0)
manual_encoder_df['last_pymnt_d_(9, 15]'] = np.where((data_process['last_pymnt_d']>9)&(data_process['last_pymnt_d']<=15),1,0)
manual_encoder_df['last_pymnt_d_(15, 102]'] = np.where(data_process['last_pymnt_d']>15,1,0)

# drop original feature
manual_encoder_df = manual_encoder_df.drop(manual_encoder_list, axis=1)

# show
manual_encoder_df.head()


# ### Concat Numeric and Categorical Encoded Features

# In[43]:


encoded = pd.concat([cat_dummies, loan_amnt_dum,int_rate_dum,dti_dum,open_acc_dum,annual_inc_dum,
                     earliest_cr_line_dum,total_acc_dum,manual_encoder_df,num_dummies['term'],
                     num_dummies['loan_status']],axis=1)

# feature engineering on term feature
encoded['term'] = np.where(encoded['term']==36,0,1)

encoded.head(3)


# In[2]:


#encoded.to_csv('encoded_bf_modeling.csv')
data = pd.read_csv('encoded_bf_modeling.csv', index_col=0)


# In[3]:


data.shape


# # Modeling

# In[48]:


print(f'this encoded dataset contains of {data.shape[0]} rows and {data.shape[1]} columns')


# ## Train Test Split

# In[4]:


from sklearn.model_selection import train_test_split

# let's separate into training and testing set
x_train, x_test, y_train, y_test = train_test_split(data.drop('loan_status', axis=1),
                                                    data['loan_status'],
                                                    test_size=0.3,
                                                    random_state=123)

x_train.shape, x_test.shape, y_train.shape, y_test.shape


# ## Handling Imbalanced Target

# In[130]:


data['loan_status'].value_counts()/len(data)*100


# In[6]:


from imblearn.over_sampling import SMOTE


# In[7]:


# Random Over Sampling
sm = SMOTE(random_state=0)
sm.fit(x_train, y_train)
x_smote, y_smote = sm.fit_resample(x_train, y_train)


# In[51]:


x_smote.shape, x_test.shape, y_smote.shape, y_train.shape


# ## Model Training & Evaluation

# In[8]:


# train model
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression() 
lr.fit(x_smote, y_smote)


# In[9]:


from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc, confusion_matrix

# let's get the predictions
y_pred_proba_train = lr.predict_proba(x_train)[:][:,1]
y_pred_proba_test = lr.predict_proba(x_test)[:][:,1]

print('AUC Train Proba :', roc_auc_score(y_train, y_pred_proba_train))
print('AUC Test Proba :', roc_auc_score(y_test, y_pred_proba_test))


# ### Hyperparameter

# In[54]:


from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

param = {
    'penalty' : ['none', 'l2', 'l1', 'elasticnet'],
    'C' : [float(x) for x in np.linspace(start=0, stop=1, num=75)]
     }

# stratified kfold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=4)
lr = LogisticRegression()

# search
lr_clf = RandomizedSearchCV(lr,
                            param,
                            scoring='roc_auc',
                            cv=skf,
                            refit=True) # refits best model to entire dataset

search_lr = lr_clf.fit(x_smote,y_smote)

# best hyperparameters
search_lr.best_params_


# In[10]:


best_params = search_lr.best_params_
lr_tuning = LogisticRegression(**best_params)
lr_tuning.fit(x_smote,y_smote)

y_train_pred_proba = lr_tuning.predict_proba(x_train)[:][:,1]
y_test_pred_lr_proba = lr_tuning.predict_proba(x_test)[:][:,1]

print('AUC Train Proba :', roc_auc_score(y_train, y_train_pred_proba))
print('AUC Test Proba :', roc_auc_score(y_test, y_test_pred_lr_proba))


# ### Find Pvalues Using StatsModel

# In[11]:


import statsmodels.api as sm

X2 = sm.add_constant(x_smote)
est = sm.OLS(y_smote, X2)
est2 = est.fit()
print(est2.summary())


# ### Feature Selection Using P-Value

# `annual_inc` has high p-value, we will drop it

# In[12]:


drop_list = ['annual_inc_(-5602.104, 1876422.0]', 'annual_inc_(1876422.0, 3750948.0]',
             'annual_inc_(3750948.0, 5625474.0]', 'annual_inc_(5625474.0, 7500000.0]']

x_smote = x_smote.drop(drop_list ,axis=1)
x_train = x_train.drop(drop_list ,axis=1)
x_test = x_test.drop(drop_list ,axis=1)


# ### Re train model

# In[15]:


param = {
    'penalty' : ['none', 'l2', 'l1', 'elasticnet'],
    'C' : [float(x) for x in np.linspace(start=0, stop=1, num=75)]
     }

# stratified kfold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=4)
lr = LogisticRegression()

# search
lr_clf = RandomizedSearchCV(lr,
                            param,
                            scoring='roc_auc',
                            cv=skf,
                            refit=True) # refits best model to entire dataset

search_lr = lr_clf.fit(x_smote,y_smote)

# best hyperparameters
search_lr.best_params_

best_params = search_lr.best_params_
lr_tuning = LogisticRegression(**best_params)
lr_tuning.fit(x_smote,y_smote)

y_train_pred_proba = lr_tuning.predict_proba(x_train)[:][:,1]
y_test_pred_lr_proba = lr_tuning.predict_proba(x_test)[:][:,1]

print('AUC Train Proba :', roc_auc_score(y_train, y_train_pred_proba))
print('AUC Test Proba :', roc_auc_score(y_test, y_test_pred_lr_proba))


# In[16]:


X2 = sm.add_constant(x_smote)
est = sm.OLS(y_smote, X2)
est2 = est.fit()
print(est2.summary())


# Still there's few feature (binned features) that consist of pvalue > 0.05, but majority all those features have pvalue <0.05

# In[71]:


# classification report
y_pred_class = []

for i in y_pred_proba_test:
    if i > 0.5:
        y_pred_class.append(1)
    else:
        y_pred_class.append(0)

print(classification_report(y_test, y_pred_class))


# In[149]:


cf_matrix = confusion_matrix(y_test, y_pred_class)

group_names = ["True Positive", "False Negative", "False Positive", "True Negative"]
group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]

labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)

plt.figure(figsize=(6, 4))
sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')


# ### ROC AUC Plot

# In[150]:


fpr, tpr, tr = roc_curve(y_test, y_pred_proba_test)
auc = roc_auc_score(y_test, y_pred_proba_test)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label='AUC = %0.3f' %auc)
plt.plot(fpr, fpr, linestyle = '--', color='grey')
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('ROC Curve', fontsize=15)
plt.legend()


# ### Kolmogorov-Smirnov

# In[151]:


import scikitplot as skplt
y_pred_proba = lr_tuning.predict_proba(x_test)

skplt.metrics.plot_ks_statistic(y_test, y_pred_proba, figsize=(7,5));


# `KS Statistic: 0.552`

# ## Feature Importance

# ### Top 10 Important Features

# In[17]:


# Converting statsmodels summary object to Pandas Dataframe,
df_importance = pd.read_html(est2.summary().tables[1].as_html(),header=0,index_col=0)[0]

# find odds_ratio
for i in df_importance['coef']:
    if i == 0 :
        df_importance['odds_ratio'] = 0
    else:
        df_importance['odds_ratio'] = np.exp(df_importance['coef'])
        
# show top 5 highest odd ratio
df_importance.iloc[1:,:].sort_values(by='odds_ratio', ascending=False).head(5)


# ### Insights

# * `last_pymnt_d_(5]` : Borrowers who made the last payment in the **past 5 months**, their odds of being good loan borrowers will **increase by 1.9 times**
# * `last_pymnt_d_(5, 7]` : Borrowers who made the last payment in the **past 5-7 months**, their odds of being good loan borrowers will **increase by 1.7 times**
# * `int_rate_(5.399, 7.484]`: Borrowers with an **interest rate of 5.3%-7.4%**, their odds of being good loan borrowers will **increase by 1.5 times**
# * Etc

# # Business Insights

# In[68]:


bi = pd.read_csv('data_bersih_sebelum_woe.csv', index_col=0)
bi.head()


# In[69]:


bi['loan_status_fc'] = np.where(bi['loan_status']==1,'good_loan','bad_loan')
bi['loan_status_fc'].value_counts()


# ## `last_pymnt_d`

# In[4]:


bi['last_pymnt_d_fc'] = np.where(bi['last_pymnt_d']==5,'5 Months',
                        np.where((bi['last_pymnt_d']>5)&(bi['last_pymnt_d']<=7),'5-7 Months',
                        np.where((bi['last_pymnt_d']>7)&(bi['last_pymnt_d']<=9),'7-9 Months',
                        np.where((bi['last_pymnt_d']>9)&(bi['last_pymnt_d']<=15),'9-15 Month','>15 Month'
                                  ))))

bi['last_pymnt_d_fc'].value_counts()


# In[5]:


abc = bi.groupby(['last_pymnt_d_fc','loan_status_fc']).agg(num_cust=('loan_status','count'))
                                                       
abc['loan_status_prop'] = abc['num_cust']/bi.groupby('last_pymnt_d_fc')['loan_status'].count()
abc = abc.reset_index()
abc['loan_status_prop'] = round(abc['loan_status_prop']*100,2)
abc


# In[44]:


order_list = ['5 Months', '5-7 Months', '7-9 Months', '9-15 Month', '>15 Month']
pal = ['#deaaff', '#8e94f2']

fig, ax = plt.subplots(figsize=(10, 6))

sns.barplot(x='last_pymnt_d_fc', y='loan_status_prop', hue='loan_status_fc', 
            data=abc, palette=pal, order=order_list)

plt.bar_label(ax.containers[0], padding=5, fmt='%.2f%%')
plt.bar_label(ax.containers[1], padding=5, fmt='%.2f%%')

sns.regplot(x=np.arange(0, len(abc[abc['loan_status_fc'] == 'bad_loan'])), y='loan_status_prop', color='#4361ee',
            data=abc[abc['loan_status_fc'] == 'good_loan'], scatter=False, truncate=False)

sns.regplot(x=np.arange(0, len(abc[abc['loan_status_fc'] == 'bad_loan'])), y='loan_status_prop', color='#f20089',
            data=abc[abc['loan_status_fc'] == 'bad_loan'], scatter=False, truncate=False)

plt.ylim(0,110)

plt.legend(loc='upper right')
plt.title('Negative Trends on Good Loan Rate of Loan Application\nPer Last Payment Months', pad=35, 
          fontsize=13, weight='extra bold')
plt.text(x=-0.5, y=113, 
         s='The longer last payment month received tend to be more into bad loan borrowers, it means that \nborrowers with recent last payment are more likely to be a good loan borrowers',
         fontstyle='italic', fontsize=11)

plt.ylabel('Good Loan Proportion Rate (%)', fontsize=10)
plt.xlabel('Last Payment Month', fontsize=10)

#plt.savefig('fig/figure11.png', dpi=500, bbox_inches='tight', pad_inches=0.2, transparent=True)


# In[7]:


# feature engineer home ownership feature
bi['home_ownership_fc'] = np.where(bi['home_ownership']=='ANY','OTHER',
                          np.where(bi['home_ownership']=='NONE','OTHER',bi['home_ownership']))

bi['home_ownership_fc'].value_counts()


# In[8]:


# because 'other' vales is too small compared to other, we will considered it as MORTGAGE (most values)
bi['home_ownership_fc'] = np.where(bi['home_ownership_fc']=='OTHER','MORTGAGE',bi['home_ownership_fc'])
bi['home_ownership_fc'].value_counts()


# In[14]:


# create new dataset
df_sum = bi.groupby(['home_ownership_fc', 'last_pymnt_d_fc']).agg(num_good_loan=('loan_status','sum'),
                                                                  num_cust=('loan_status','count')).reset_index()

df_sum['good_borrower_prop'] = df_sum['num_good_loan']/df_sum['num_cust']
df_sum['bad_borrower_prop'] = (df_sum['num_cust']-df_sum['num_good_loan'])/df_sum['num_cust']
df_sum['bad_borrower_prop_pctg'] = round(df_sum['bad_borrower_prop']*100,2)
df_sum


# In[46]:


pal = ['#ffd6ff', '#deaaff', '#8e94f2']

plt.figure(figsize=(10,6))
sns.barplot(x='last_pymnt_d_fc', y='bad_borrower_prop_pctg', data=df_sum, 
            hue='home_ownership_fc', palette=pal)

plt.title('Proportion of Bad Loan Borrowers Per Last Payment Months\nBased on Home Ownership Type',
          fontsize=14, weight='extra bold', pad=45)
plt.text(x=-0.5, y=28, s='Borrowers who have rent home ownership type are more likely to be a bad loan borrowers, also borrowers \nwho have long last payment month tend to be a bad loan borrowers, this is probably because borrowers who \nown a house or martgage have higher buying power or we can say they earn higher than someone who rent a house',
         fontstyle='italic', fontsize=10)

plt.ylabel('Bad Borrowers Rate (%)')
plt.xlabel('Last Payment Month')

#plt.savefig('fig/figure22.png', dpi=500, bbox_inches='tight', pad_inches=0.2, transparent=True)


# ## `total_acc`

# In[187]:


sns.kdeplot(x='total_acc', hue='loan_status', data=bi, shade=True)


# In[189]:


bi['total_acc'].describe()


# In[190]:


# fine classing = 6 class
bi['total_acc_fc'] = pd.cut(bi['total_acc'], 6)
bi['total_acc_fc'].value_counts().sort_index()


# In[190]:


# rename values
bi['total_acc_fc'] = np.where((bi['total_acc']>=1)&(bi['total_acc']<=26),'1-26',
                     np.where((bi['total_acc']>26)&(bi['total_acc']<=52),'26-52',
                     np.where((bi['total_acc']>52)&(bi['total_acc']<=78),'52-78','78-156')))

bi['total_acc_fc'].value_counts()


# In[191]:


acb = bi.groupby(['total_acc_fc','loan_status_fc']).agg(num_cust=('loan_status','count'))
acb['loan_status_prop'] = acb['num_cust']/bi.groupby('total_acc_fc')['loan_status'].count()
acb = acb.reset_index()
acb['loan_status_prop'] = round(acb['loan_status_prop']*100,2)
acb


# In[204]:


# create new dataset
df_acc = bi.groupby(['total_acc_fc', 'last_pymnt_d_fc']).agg(num_good_loan=('loan_status','sum'),
                                                                  num_cust=('loan_status','count')).reset_index()

df_acc['good_borrower_prop'] = df_acc['num_good_loan']/df_acc['num_cust']
df_acc['bad_borrower_prop'] = (df_acc['num_cust']-df_acc['num_good_loan'])/df_acc['num_cust']
df_acc['bad_borrower_prop_pctg'] = round(df_acc['bad_borrower_prop']*100,2)
df_acc


# In[205]:


pal = ['#ffd6ff', '#deaaff', '#8e94f2' ,'#757bc8', '#4f518c']

fig, ax = plt.subplots(figsize=(13, 8))
sns.barplot(x='total_acc_fc', y='bad_borrower_prop_pctg', data=df_acc, 
            hue='last_pymnt_d_fc', palette=pal)

plt.bar_label(ax.containers[0], padding=3, fmt='%.2f%%')
plt.bar_label(ax.containers[1], padding=3, fmt='%.2f%%')
plt.bar_label(ax.containers[2], padding=3, fmt='%.2f%%')
plt.bar_label(ax.containers[3], padding=3, fmt='%.2f%%')
plt.bar_label(ax.containers[4], padding=3, fmt='%.2f%%')

plt.xlabel('Number of Credit Lines',fontsize=12)
plt.ylabel('Good Borrowers Rate (%)',fontsize=12)

plt.title('Proportion of Bad Loan Borrowers Per Credit Lines \nBased on Last Payment Month',
         fontsize=20, weight='extra bold', pad=30)
#plt.text(x=-0.5, y=28, s='High number of credit lines group has slightly higher on Bad Borrowers Rate', 
        # fontsize=12, fontstyle='italic')


# ## `int_rate`

# In[204]:


sns.kdeplot(bi['int_rate'], hue=bi['loan_status'])


# In[17]:


bi['int_rate'].describe()


# In[98]:


bi['int_rate_fc'] = pd.cut(bi['int_rate'], 10)
bi['int_rate_fc'].value_counts()


# In[70]:


bi['int_rate_fc'] = np.where((bi['int_rate']>5.399)&(bi['int_rate']<=7.484),'5.3-7.4',
                    np.where((bi['int_rate']>7.484)&(bi['int_rate']<=9.548),'7.4-9.5',
                    np.where((bi['int_rate']>9.548)&(bi['int_rate']<=11.612),'9.5-11.61',
                    np.where((bi['int_rate']>11.612)&(bi['int_rate']<=13.676),'11.6-13.6',
                    np.where((bi['int_rate']>13.676)&(bi['int_rate']<=15.740),'13.6-15.7',
                    np.where((bi['int_rate']>15.740)&(bi['int_rate']<=17.804),'15.7-17.8',
                    np.where((bi['int_rate']>17.804)&(bi['int_rate']<=19.868),'17.8-19.8',
                    np.where((bi['int_rate']>19.868)&(bi['int_rate']<=21.932),'19.8-21.9',
                    np.where((bi['int_rate']>21.932)&(bi['int_rate']<=23.996),'21.9-23.9','>23.9')))))))))

bi['int_rate_fc'].value_counts()


# In[ ]:


4361ee  f20089


# In[75]:


pal = ['#deaaff', '#8e94f2']
pal2 = ['#f20089', '#4361ee']

order_list = ['5.3-7.4', '7.4-9.5','9.5-11.61','11.6-13.6','13.6-15.7','15.7-17.8','17.8-19.8',
             '19.8-21.9','21.9-23.9','>23.9']

plt.figure(figsize=(15,7))
sns.pointplot(x='int_rate_fc',y='loan_amnt',data=bi,palette=pal2, 
              hue='loan_status_fc',ci=None, order=order_list)

sns.barplot(x='int_rate_fc',y='loan_amnt',data=bi,palette=pal, 
              hue='loan_status_fc',ci=None, order=order_list)

plt.axvline(6.5, ls='--', color='#9984d4')
plt.axvline(9.5, ls='--', color='#9984d4')
plt.stackplot(np.arange(6.49,9.51), [[25000]], color='#baebff', alpha=0.3)
plt.text(x=7.25, y=23000, s='This Type of Borrowes', fontsize=12, 
         color='#431259', va='center', weight='extra bold')
plt.text(x=6.75, y=22000, s='Tend to be a Bad Loan Borrowes', fontsize=12, 
         color='#431259', va='center', weight='extra bold')

plt.xlim(0,9.7)
plt.ylim(0,25000)

plt.title('Loan Amount Per Interest Rate Based On \nBorrowers Loan Status', 
          fontsize=18, weight='extra bold', pad=34)
plt.text(x=0, y=25500, s='The higher loan amount followed by high interest rate, higher interest rate slightly tend to be a bad loan borrowers',
         fontstyle='italic', fontsize=14)

plt.ylabel('Loan Amount', fontsize=14)
plt.xlabel('Interest Rate', fontsize=14)

#plt.savefig('fig/figure33.png', dpi=500, bbox_inches='tight', pad_inches=0.2, transparent=True)


# ## `loan_amnt`

# In[206]:


bi['loan_amnt'].describe()


# In[14]:


bi['loan_amnt_fc'] = pd.cut(bi['loan_amnt'], 10)
bi['loan_amnt_fc'].value_counts()


# In[176]:


bi['loan_amnt_fc'] = np.where((bi['loan_amnt']>=500)&(bi['loan_amnt']<=3950),'500-3950',
                     np.where((bi['loan_amnt']>3950)&(bi['loan_amnt']<=7400),'3950-7400',
                     np.where((bi['loan_amnt']>7400)&(bi['loan_amnt']<=10800),'7400-10800',
                     np.where((bi['loan_amnt']>10800)&(bi['loan_amnt']<=14300),'10800-14300',
                     np.where((bi['loan_amnt']>14300)&(bi['loan_amnt']<=17750),'14300-17750',
                     np.where((bi['loan_amnt']>17750)&(bi['loan_amnt']<=21200),'17750-21200',
                     np.where((bi['loan_amnt']>21200)&(bi['loan_amnt']<=24650),'21200-246500',
                     np.where((bi['loan_amnt']>24650)&(bi['loan_amnt']<=28100),'24650-28100',
                     np.where((bi['loan_amnt']>28100)&(bi['loan_amnt']<=31550),'28100-31550','31550-35000')))))))))

bi['loan_amnt_fc'].value_counts()


# In[177]:


# create new dataset
df_loan = bi.groupby(['loan_amnt_fc', 'total_acc_fc']).agg(num_good_loan=('loan_status','sum'),
                                                           num_cust=('loan_status','count')).reset_index()

df_loan['good_borrower_prop'] = df_loan['num_good_loan']/df_loan['num_cust']
df_loan['bad_borrower_prop'] = (df_loan['num_cust']-df_loan['num_good_loan'])/df_loan['num_cust']
df_loan


# In[184]:


order_list = ['500-3950','3950-7400','10800-14300','14300-17750','17750-21200',
              '21200-246500','24650-28100','28100-31550','31550-35000']

fig, ax = plt.subplots(figsize=(13, 8))

sns.barplot(x='loan_amnt_fc', y='bad_borrower_prop', data=df_loan, 
            hue='total_acc_fc', palette='viridis', order=order_list)

plt.bar_label(ax.containers[0], padding=3, fmt='%.2f%%', fontsize=7)
plt.bar_label(ax.containers[1], padding=3, fmt='%.2f%%', fontsize=7)
plt.bar_label(ax.containers[2], padding=3, fmt='%.2f%%', fontsize=7)

plt.ylim(0,0.17)


# In[54]:


# create new dataset
df_loan2 = bi.groupby(['loan_amnt_fc', 'home_ownership_fc']).agg(num_good_loan=('loan_status','sum'),
                                                                 num_cust=('loan_status','count')).reset_index()

df_loan2['good_borrower_prop'] = df_loan2['num_good_loan']/df_loan2['num_cust']
df_loan2['bad_borrower_prop'] = (df_loan2['num_cust']-df_loan2['num_good_loan'])/df_loan2['num_cust']
df_loan2


# In[64]:


order_list = ['500-3950','3950-7400','10800-14300','14300-17750','17750-21200',
              '21200-246500','24650-28100','28100-31550','31550-35000']

plt.figure(figsize=(15,7))
sns.barplot(x='loan_amnt_fc', y='bad_borrower_prop', data=df_loan2, alpha=0.1,
            hue='home_ownership_fc', palette='viridis', order=order_list)

sns.pointplot(x='loan_amnt_fc', y='bad_borrower_prop', data=df_loan2, 
            hue='home_ownership_fc', palette='viridis', order=order_list)

plt.ylim(0,0.25)


# ## `Top 5 Feature Importance`

# In[185]:


# initialise data of lists.
data_dict = {'Feature':['Last Payment: 5 Month', 'Last Payment: 5-7 Month', 'Total Credit Line: 104-130', 
                        'Interest Rate: 5.3-7.4', 'Interest Rate: 7.4-95'], 
             'Odds Ratio':[1.95, 1.75, 1.64, 1.59, 1.52]}
 
# Create DataFrame
df_important = pd.DataFrame(data_dict)
df_important


# In[207]:


pal = ['#431259', '#60308c', '#805ebf', '#9a99f2', '#ccdcff']
sns.barplot(x='Odds Ratio', y='Feature', data=df_important, palette=pal)
plt.ylabel(' ')
plt.title('Top 5 Important Features', fontsize=15, weight='extra bold')
#plt.savefig('fig/featureimportance2.png', dpi=500, bbox_inches='tight', pad_inches=0.2, transparent=False)


# # Creating Score Card

# In[18]:


# set new index
df_importance = df_importance.reset_index()

# rename columns
df_importance = df_importance.rename(columns = {'index' : 'feature'})

# creat new columns feature_name (stand for original feature name)
df_importance['feature_name'] = df_importance['feature'].str.split('_').str[:-1]
df_importance['feature_name'] = df_importance['feature_name'].str.join('_')
df_importance.at[0,'feature_name']='intercept'
df_importance.at[80,'feature_name']='term'

df_importance


# ## Scoring Each Features

# we are using FICO scale to make credit score card. Read more about the scale : [link](https://www.badcredit.org/how-to/credit-score-range/)
# * min_score = 300
# * max_score = 850

# In[19]:


# copy dataset
df_scorecard = df_importance.copy()

# define max and min score
min_score = 300
max_score = 850


# In[20]:


# aggregate min and sum
min_sum_coef = df_scorecard.groupby('feature_name')['coef'].min().sum()

# aggregate max and sum
max_sum_coef = df_scorecard.groupby('feature_name')['coef'].max().sum()

# define credit score
df_scorecard['Score - Calculation'] = df_scorecard['coef'] * (max_score - min_score) / (max_sum_coef - min_sum_coef)

# adjust intercept values
df_scorecard['Score - Calculation'][0] = ((df_scorecard['coef'][0] - min_sum_coef) / ((max_sum_coef - min_sum_coef)) * (max_score - min_score) + min_score)

# round credit score
df_scorecard['Score - Preliminary'] = df_scorecard['Score - Calculation'].round()


# In[21]:


# check min score, it shoud be = 300
min_sum_score_prel = df_scorecard.groupby('feature_name')['Score - Preliminary'].min().sum()
# check max score, it should be = 850
max_sum_score_prel = df_scorecard.groupby('feature_name')['Score - Preliminary'].max().sum()

print('min score', min_sum_score_prel)
print('max score', max_sum_score_prel)


# this error due to round(), we will adjust the values

# In[22]:


# check difference
df_scorecard['Difference'] = df_scorecard['Score - Preliminary'] - df_scorecard['Score - Calculation']


# In[23]:


df_scorecard['Score - Final'] = df_scorecard['Score - Preliminary']
df_scorecard['Score - Final'][55] = 112 # change value from 113 to 112

# check min score, it shoud be = 300
min_sum_score_prel = df_scorecard.groupby('feature_name')['Score - Final'].min().sum()
# check max score, it should be = 850
max_sum_score_prel = df_scorecard.groupby('feature_name')['Score - Final'].max().sum()

print('min score', min_sum_score_prel)
print('max score', max_sum_score_prel)


# ## Score Card FICO Scale (300-850)

# In[24]:


# define data
data_fico = data[x_smote.columns]

# copy
df = data_fico.copy()
df.head()


# In[25]:


# We insert a column in the dataframe, with an index of 0, that is, in the beginning of the dataframe.
# The name of that column is 'Intercept', and its values are 1s.
df.insert(0, 'Intercept', 1)


# In[26]:


# define score card
scorecard_scores = df_scorecard['Score - Final']

# reshape
scorecard_scores = scorecard_scores.values.reshape(81, 1)


# In[27]:


# multiply the values of each row of the dataframe by the values of each column of the variable using dot
y_scores = df.dot(scorecard_scores)


# In[28]:


# concat
score_card_df = pd.concat([df, y_scores], axis=1)

# rename
score_card_df.rename(columns={0:'Credit Score'}, inplace=True)

# show
score_card_df.head(5)


# In[30]:


# join id based on index
# df = pd.read_csv('loan_data_2007_2014.csv', index_col=0)
df = df[['id','member_id']]


# In[35]:


df.shape, score_card_df.shape


# In[36]:


credit_score_w_id = pd.merge(df, score_card_df, left_index = True, right_index = True)
credit_score_w_id = credit_score_w_id[['id','member_id','Credit Score']]
credit_score_w_id.head()


# In[37]:


credit_score_w_id.sample(5)


# In[56]:


credit_score_w_id.shape


# In[65]:


# export to csv
# credit_score_w_id.to_csv('credit_score_w_id.csv')


# ## Score Card 0-100 Scale

# In[62]:


# converting FICO scale to 100% scale
sum_coef_from_score = ((y_scores - min_score) / (max_score - min_score)) * (max_sum_coef - min_sum_coef) + min_sum_coef

credit_score_pctg_scale = np.exp(sum_coef_from_score) / (np.exp(sum_coef_from_score) + 1)

# concat
score_card_df_pctg = pd.merge(df, credit_score_pctg_scale, left_index = True, right_index = True)

# rename
score_card_df_pctg.rename(columns={0:'Credit Score'}, inplace=True)
score_card_df_pctg.head()


# In[63]:


score_card_df_pctg.shape


# In[66]:


# export to csv
# score_card_df_pctg.to_csv('credit_score_pctg_w_id.csv')


# In[67]:


score_card_df_pctg.sample(5)


# ## Setting Cut-Off

# In[130]:


df_proba = pd.concat([score_card_df_pctg, data['loan_status']], axis=1)
df_proba.head()


# In[131]:


df_proba.shape


# In[132]:


fpr, tpr, thresholds = roc_curve(df_proba['loan_status'], df_proba['Credit Score'])


# In[133]:


# We concatenate 3 dataframes along the columns.
df_cutoff = pd.concat([pd.DataFrame(thresholds), pd.DataFrame(fpr), pd.DataFrame(tpr)], axis = 1)

# rename columns 
df_cutoff.columns = ['thresholds', 'fpr', 'tpr']

# Let the first threshold (the value of the thresholds column with index 0)
# be equal to a number, very close to 1
# but smaller than 1, say 1 - 1 / 10 ^ 16.
df_cutoff['thresholds'][0] = 1 - 1 / np.power(10, 16)

# calculate score, The score corresponsing to each threshold
df_cutoff['Score'] = ((np.log(df_cutoff['thresholds'] / (1 - df_cutoff['thresholds'])) - min_sum_coef) * ((max_score - min_score) / (max_sum_coef - min_sum_coef)) + min_score).round()

# show
df_cutoff.sample(10)


# In[136]:


def n_approved(p):
    return np.where(df_proba['Credit Score'] >= p, 1, 0).sum()


# In[145]:


# calculate number of approved loan
df_cutoff['N Approved'] = df_cutoff['thresholds'].apply(n_approved)

# calculate number of rejected loan
df_cutoff['N Rejected'] = df_proba['Credit Score'].shape[0] - df_cutoff['N Approved']

# calculate approoval rate
df_cutoff['Approval Rate'] = df_cutoff['N Approved'] / df_proba.shape[0]

# # calculate rejection rate
df_cutoff['Rejection Rate'] = 1 - df_cutoff['Approval Rate']

# set max for first rows
df_cutoff['Score'][0] = max_score

# show
df_cutoff.head(20)


# In[147]:


df_cutoff.iloc[150:200,]


# In[148]:


#df_cutoff.to_csv('df_cutoff.csv')

