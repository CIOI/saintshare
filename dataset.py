import config 

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import gc
from sklearn.model_selection import train_test_split 


class DKTDataset(Dataset):
  def __init__(self,group,n_skills,max_seq = 100):
    self.samples = group
    self.n_skills = n_skills
    self.max_seq = max_seq
    self.data = []

    for que,ans,res_time,exe_cat in self.samples:
        if len(que)>=self.max_seq:
            self.data.extend([(que[l:l+self.max_seq],ans[l:l+self.max_seq],res_time[l:l+self.max_seq],exe_cat[l:l+self.max_seq])\
            for l in range(len(que)) if l%self.max_seq==0])
        elif len(que)<self.max_seq and len(que)>10:
            self.data.append((que,ans,res_time,exe_cat))
        else :
            continue
  
  def __len__(self):
    return len(self.data)
  
  def __getitem__(self,idx):
    content_ids,answered_correctly,response_time,exe_category = self.data[idx]
    seq_len = len(content_ids)

    q_ids = np.zeros(self.max_seq,dtype=int)
    ans = np.zeros(self.max_seq,dtype=int)
    r_time = np.zeros(self.max_seq,dtype=int)
    exe_cat = np.zeros(self.max_seq,dtype=int)

    if seq_len>=self.max_seq:
      q_ids[:] = content_ids[-self.max_seq:]
      ans[:] = answered_correctly[-self.max_seq:]
      r_time[:] = response_time[-self.max_seq:]
      exe_cat[:] = exe_category[-self.max_seq:]
    else:
      q_ids[-seq_len:] = content_ids
      ans[-seq_len:] = answered_correctly
      r_time[-seq_len:] = response_time
      exe_cat[-seq_len:] = exe_category
    
    target_qids = q_ids[1:]
    label = ans[1:] 

    input_ids = np.zeros(self.max_seq-1,dtype=int)
    input_ids = q_ids[:-1].copy()

    input_rtime = np.zeros(self.max_seq-1,dtype=int)
    input_rtime = r_time[:-1].copy()

    input_cat = np.zeros(self.max_seq-1,dtype=int)
    input_cat = exe_cat[:-1].copy()

    input = {"input_ids":input_ids,"input_rtime":input_rtime.astype(np.int),"input_cat":input_cat}

    return input,target_qids,label 

def DKT2Riid(df,dtypes):
    '''
    ['userID', 'assessmentItemID', 'testId', 'answerCode', 'Timestamp','KnowledgeTag']
    를 ["user_id","content_id","answered_correctly","prior_question_elapsed_time","task_container_id"]
    Knowledge Tag는 사용하지 않는다.
    '''
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # 시간 차이 계산
    df['Time_difference'] = df['Timestamp'].diff()

    df['Time_difference'] = df['Time_difference'].dt.seconds
    df.loc[df['assessmentItemID'].str.endswith('01'), 'Time_difference'] = 300 # 각 시험지 문제 1번은 결측치로 처리, Nan 대신 300을 넣었는데 이부분 아이디어가 안 떠오름...
    del df['Timestamp']

    # assessmentItemID에서 필요 정보(0을 제외한 숫자)만 가져오기
    df['exerid'] = df['assessmentItemID'].str[2:3] + df['assessmentItemID'].str[-2:]
    del df['assessmentItemID']
    
    # testid 에서 도
    df['testId'] = df['testId'].str[2:3] + df['testId'].str[-2:]
    
    df=df[['userID', 'exerid','answerCode',	'Time_difference','testId' 	]]
    df.columns=["user_id","content_id","answered_correctly","prior_question_elapsed_time","task_container_id"]
    
    df = df.astype(dtypes)
    return df
    
def get_dataloaders():              
    dtypes = { 'user_id': 'int32' ,'content_id': 'int16',
                'answered_correctly':'int8',"prior_question_elapsed_time":"float32","task_container_id":"int16"}
    print("loading csv.....")
    train_df = pd.read_csv('./content/data/train_data.csv')#dtype=dtypes,nrows=20000) 
    print("shape of dataframe :",train_df.shape) 
    train_df = DKT2Riid(train_df,dtypes)
    
    train_df.prior_question_elapsed_time.fillna(300,inplace=True)

    train_df.prior_question_elapsed_time.clip(lower=0,upper=300,inplace=True)# 푸는데 걸리는 시간을 최대 300으로 제한

    train_df.prior_question_elapsed_time = train_df.prior_question_elapsed_time.astype(np.int)

    train_df['prior_question_elapsed_time'].unique()
    
    skills = train_df.content_id.unique()
    n_skills = len(skills)
    n_cats = len(train_df.task_container_id.unique())+100
    print("no. of skills :",n_skills)
    print("no. of categories: ", n_cats)
    print("shape after exlusion:",train_df.shape)

    #grouping based on user_id to get the data supplu
    print("Grouping users...")
    group = train_df[["user_id","content_id","answered_correctly","prior_question_elapsed_time","task_container_id"]]\
                    .groupby("user_id")\
                    .apply(lambda r: (r.content_id.values,r.answered_correctly.values,r.prior_question_elapsed_time.values,r.task_container_id.values))
    del train_df
    gc.collect()

    print("splitting")
    train,val = train_test_split(group,test_size=0.2) 
    print("train size: ",train.shape,"validation size: ",val.shape)
    train_dataset = DKTDataset(train.values,n_skills=n_skills,max_seq = config.MAX_SEQ)
    val_dataset = DKTDataset(val.values,n_skills=n_skills,max_seq = config.MAX_SEQ)
    train_loader = DataLoader(train_dataset,
                          batch_size=config.BATCH_SIZE,
                          num_workers=2,
                          shuffle=True)
    val_loader = DataLoader(val_dataset,
                          batch_size=config.BATCH_SIZE,
                          num_workers=2,
                          shuffle=False)
    del train_dataset,val_dataset
    gc.collect()
    return train_loader, val_loader