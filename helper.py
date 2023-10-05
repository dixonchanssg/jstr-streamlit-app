import pandas as pd
import numpy as np
import math
import re
import ssg_sea
from ssg_sea.extract_skills import extract_skills
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# generate_similar_cwf_kt
# read in job description from uploaded dataset and find top 15 closest CWF clusters for each of the job role
def generate_similar_cwf_kt(index, job_title, job_desc):
    df = pd.read_csv("data_processed/cwf_kt.csv")    
    # Add a new row with the neccessary values required
    df.loc[len(df)] = [df.iloc[:, 0].max() + 1, 99999, "NA", job_desc]
    df['Function ID'] = df['Function ID'].astype(int)
    #model = SentenceTransformer('bert-base-nli-mean-tokens')
    model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
    embeddings = np.load("data_processed/embeddings_distilbert.npy")
    current_embeddings = model.encode([job_desc], batch_size=8)
    # Append the current embedding to the saved embeddings
    embeddings = np.vstack((embeddings, current_embeddings))
    pairwise = pd.DataFrame(cosine_similarity(embeddings))
    long_form = pairwise.unstack()
    long_form.index.rename(['target_id', 'source_id'], inplace=True)
    df_dist = long_form.to_frame('sim_score').reset_index()
    df_sorted = df_dist.sort_values(by=['source_id'])
    df_sorted['rank'] = df_sorted.groupby('source_id')['sim_score'].rank('dense', ascending=False)
    df_sorted['rank'] = df_sorted['rank'].astype(int)
    #locate position of the job description that we wanted to match to CWFs
    pos = len(df.index) - 1
    df_output = df_sorted.loc[df_sorted['source_id'] == pos]
    #choose top 20 cwfs which are most similar to sample job posting
    df_output = df_output.loc[df_output['rank'] < 21]
    df1 = pd.merge(df_output, df[['index', 'Function ID']], left_on='target_id', right_on='index', how ='left')
    df2 = pd.merge(df1, df[['Function ID', 'Critical Work Function', 'Key Tasks']], left_on='Function ID', right_on='Function ID', how ='left').drop(['index', 'Function ID'], axis=1)
    df2 = df2.sort_values(by=['rank'])
    return df2

# Add raw job description for the uploaded job roles in 'cwf_kt.csv'
# find most similar cwf clusters for each job roles and stores output to dataframe list
def populateCWFClusters(df, cwf_kt, df_cwf):
    final = pd.merge(df, cwf_kt[['index', 'Function ID']], left_on = 'target_id', right_on = 'index', how = 'left')
    final = pd.merge(final, df_cwf[['Function ID', 'Sector', 'Track', 'Job Role', 'Key Task ID', 'level_1', 'level_2', 'level_3', 'level_3_refined', 'labels', 'level_2 name']], on = 'Function ID', how = 'left')
    final = final.drop_duplicates(subset ='Function ID')
    final = final.sort_values(by=['rank'])
    final = final.reset_index()
    final = final[['rank', 'sim_score', 'index', 'Sector', 'Track', 'Job Role', 'Critical Work Function',  'Function ID', 'Key Tasks', 'Key Task ID', 'level_1', 'level_2', 'level_3', 'level_3_refined', 'labels', 'level_2 name']]
    return final

# Remove non-printable characters from any text
def clean_text(desc):
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', desc)
    return cleaned_text

# Prepare df_jobrole_cwf to run JSTR
def prepareJobroleCWF (df_jobrole_cwf, populated_df, uploaded_job_titles):
    df_jobrole_cwf['SourceJobWeight'] = 1
    num_job_roles = df_jobrole_cwf['Job Role ID'].nunique()
    
    # create an empty list to store dataframes
    dfs = []
    i = 0
    # loop through each dataframe and read it into a dataframe
    for df in populated_df:
        #global num_job_roles
        # remove rows so that we only take in top 15 CWF clusters for each role
        df = df[df['rank'] != 1]
        df = df[df['rank'] <= 16]
    
        #Use sim_core as SourceJobWeight in the computation of adjacency
        df['SourceJobWeight'] = df['sim_score']
    
        # drop specified columns
        df = df.drop(columns=['rank', 'sim_score'])

        # set the first column to "Job Role ID"
        df = df.rename(columns={df.columns[0]: "Job Role ID"})

        # assign values to the "Sector", "Track", and "Job Role" columns
        df["Sector"] = "New Sector"
        df["Track"] = "None"
        df["Job Role"] = uploaded_job_titles[i]

        # set the value of "Job Role ID" to a string in the format "r" & str(num_job_roles + 1 + index of dfs)"
        df["Job Role ID"] = "r" + str(num_job_roles + i)

        # move last column "Sector" to 2nd column
        last_col = df.pop(df.columns[-1])
        df.insert(1, last_col.name, last_col)
        i = i + 1
   
        dfs.append(df) 
    # concatenate dfs[] to df_jobrole_cwf
    df_jobrole_cwf = pd.concat([df_jobrole_cwf] + dfs)
    df_jobrole_cwf.shape
    return df_jobrole_cwf

# Data preparation to run JSTR
def dataPrepForJSTR(df_job_cwf):
    #Drop unneccessary columns and rename CWF L3 cluster
    df_job_cwf = df_job_cwf.drop(['Track','Function ID','Key Tasks','Key Task ID','level_1','level_2','level_3','labels','level_2 name'], axis=1)
    df_job_cwf.rename(columns={'level_3_refined':'CWF L3 Cluster'}, inplace=True)

    df_source = df_job_cwf.copy()
    df_source["Job Role ID"] = "R: " + df_source["Job Role ID"]
    df_source = df_source[["Job Role ID", "Sector","Job Role", "Critical Work Function","CWF L3 Cluster", "SourceJobWeight"]]

    df_dest = df_job_cwf.copy()
    df_dest["Job Role ID"] = "D: " + df_dest["Job Role ID"]
    df_dest = df_dest[["Job Role ID","Sector","Job Role","Critical Work Function","CWF L3 Cluster", "SourceJobWeight"]]

    JobCWF_df = pd.concat([df_source, df_dest], ignore_index=True)
    # Apply exponential pleateau function on SourceJobWeight in preparation for sim score computation
    JobCWF_df['SourceJobWeight'] = JobCWF_df['SourceJobWeight'].apply(lambda x: 1 - (math.exp(-4 * x)) if (x < 1) else 1  )

    return JobCWF_df


def computeJobAdjacency(JobCWF_df):
    JobCWF_df['Num_Cluster_per_JobRole'] = JobCWF_df.groupby('Job Role ID')['CWF L3 Cluster'].transform('nunique')
    JobCWF_df['Num_CWF_per_JobRole'] = JobCWF_df.groupby('Job Role ID')['Job Role ID'].transform('count')
    JobCWF_df['Num_CWF_per_Cluster'] = JobCWF_df.groupby(['Job Role ID','CWF L3 Cluster'])['CWF L3 Cluster'].transform('count')

    JobCWF_df['CWFClusterWeight'] = JobCWF_df['Num_CWF_per_Cluster'] / JobCWF_df['Num_CWF_per_JobRole']

    #Drop duplicates based on unique Job Role ID and L3 CLuster, retaining the row with highest CWFClusterWeight
    JobCWF_df = JobCWF_df.drop(['Critical Work Function'], axis=1)
    JobCWF_df = JobCWF_df.groupby(['Job Role ID','CWF L3 Cluster'], as_index=False).max()

    # Factor in SourceJobWeight for uploaded roles
    JobCWF_df['CWFClusterWeight'] = JobCWF_df['CWFClusterWeight'] * JobCWF_df['SourceJobWeight']

    # Create dataframe with source and dest roles and their corresponding shared CWF L3 cluster 
    # (Job Role ID	Job Role ID_2	CWF L3 Cluster)
    column_edge = "CWF L3 Cluster"
    Job_Role_ID = "Job Role ID"
    JobJob_CWF_df = JobCWF_df[[Job_Role_ID, column_edge]].dropna(subset=[column_edge]).drop_duplicates()
    JobJob_CWF_df = JobJob_CWF_df.merge(JobJob_CWF_df[[Job_Role_ID, column_edge]].rename(columns={Job_Role_ID:Job_Role_ID+"_2"}), on=column_edge)
    JobJob_CWF_df = JobJob_CWF_df[~(JobJob_CWF_df[Job_Role_ID]==JobJob_CWF_df[Job_Role_ID+"_2"])].dropna()[[Job_Role_ID, Job_Role_ID+"_2", column_edge]]

    JobJob_CWF_df = JobJob_CWF_df.merge(JobCWF_df[["Job Role ID", "CWF L3 Cluster", "SourceJobWeight", "Num_Cluster_per_JobRole", "Num_CWF_per_JobRole", "Num_CWF_per_Cluster" ,"CWFClusterWeight"]], left_on=["Job Role ID_2", "CWF L3 Cluster"], right_on=["Job Role ID", "CWF L3 Cluster"], how='left')
    JobJob_CWF_df = JobJob_CWF_df.drop(['Job Role ID_y'], axis=1)
    JobJob_CWF_df = JobJob_CWF_df.rename(columns={"Job Role ID_x": "Job Role ID"})

    JobJob_CWF_df["Num_Common_Cluster"] = JobJob_CWF_df.groupby(["Job Role ID", "Job Role ID_2"])["Job Role ID"].transform('count')
    JobJob_CWF_df['PercentCommon_byNumCluster'] = JobJob_CWF_df['Num_Common_Cluster']/JobJob_CWF_df['Num_Cluster_per_JobRole']
    JobJob_CWF_df['PercentCommon_bysumClusterweight'] = JobJob_CWF_df.groupby(["Job Role ID", "Job Role ID_2"])["CWFClusterWeight"].transform('sum')

    CWF = JobJob_CWF_df[~JobJob_CWF_df['Job Role ID'].astype(str).str.startswith('D')]
    CWF = CWF[~CWF['Job Role ID_2'].astype(str).str.startswith('R')]

    df_cwf = CWF.copy()
    df_cwf = df_cwf[['Job Role ID', 'Job Role ID_2', 'PercentCommon_bysumClusterweight']]
    df_cwf['Job Role ID'] = df_cwf['Job Role ID'].str[2:]
    df_cwf['Job Role ID_2'] = df_cwf['Job Role ID_2'].str[2:]
    df_cwf = df_cwf.rename(columns={"Job Role ID": "Source Job ID", "Job Role ID_2": "Dest Job ID", "PercentCommon_bysumClusterweight": "CWF Sim Score"})

    #Only keep job pairs with sim score >= 0.7
    df_cwf = df_cwf[df_cwf['CWF Sim Score'] >= 0.70]

    #Remove rows with the same source and dest job ID
    df_cwf = df_cwf[df_cwf['Source Job ID'] != df_cwf['Dest Job ID']]

    return df_cwf

def populateJobPairsWithJobTitle(df_unique_job, df_cwf, sfw_df):
    #Create unique list of job roles
    df_unique_job = df_unique_job[['Job Role ID', 'Sector', 'Track', 'Job Role']].drop_duplicates()

    df_cwf['Source Job ID'] = df_cwf['Source Job ID'].str.strip()
    df_cwf['Dest Job ID'] = df_cwf['Dest Job ID'].str.strip()
    df_unique_job['Job Role ID'] = df_unique_job['Job Role ID'].str.strip()

    #Populate destination job details
    df_cwf = pd.merge(df_cwf, df_unique_job[['Job Role ID', 'Sector', 'Track', 'Job Role']], left_on='Dest Job ID', right_on='Job Role ID')
    df_cwf = df_cwf.drop(columns=['Job Role ID'])
    df_cwf = df_cwf.rename(columns={"Sector": "Dest Sector", "Job Role": "Dest Job Role", "Track": "Dest Track"})

    #Populate source job details
    df_cwf = pd.merge(df_cwf, df_unique_job[['Job Role ID', 'Sector', 'Track', 'Job Role']], left_on='Source Job ID', right_on='Job Role ID')
    df_cwf = df_cwf.drop(columns=['Job Role ID'])
    df_cwf = df_cwf.rename(columns={"Sector": "Source Sector", "Job Role": "Source Job Role", "Track": "Source Track"})

    df_cwf.drop_duplicates(inplace=True)
    df_cwf = df_cwf[['Source Job ID', 'Source Sector', 'Source Track', 'Source Job Role', 'Dest Job ID', 'Dest Sector', 'Dest Track', 'Dest Job Role', 'CWF Sim Score']]

    #Only retain rows with Source Sector="New Sector"
    df_cwf = df_cwf[df_cwf['Source Sector'] == 'New Sector']

    df_cwf = pd.merge(df_cwf, sfw_df[['Job Role ID', 'Average TSC Level']], left_on='Dest Job ID', right_on='Job Role ID')
    df_cwf = df_cwf.drop(columns=['Job Role ID'])
    df_cwf = df_cwf.rename(columns={"Average TSC Level": "Dest Average TSC Level"})

    return df_cwf


def populateSSOCforDestJobs(df_cwf, df_ssoc):
    df_ssoc = df_ssoc[['Job Role ID', 'SSOC 2020']]
    df_ssoc = df_ssoc.drop_duplicates(subset=['Job Role ID'])

    df_cwf = pd.merge(df_cwf, df_ssoc[['Job Role ID', 'SSOC 2020']], how='left', left_on='Dest Job ID', right_on='Job Role ID' )
    del df_cwf['Job Role ID']
    df_cwf.rename(columns={'SSOC 2020':'Dest Job SSOC'}, inplace=True)

    return df_cwf

def populateSalaryforDestJobs(df_cwf, df_salary):
    df_salary['SSOC5dCode'] = df_salary['SSOC5dCode'].astype(str)
    df_cwf['Dest Job SSOC'] = df_cwf['Dest Job SSOC'].astype(str)
    df_cwf = pd.merge(df_cwf, df_salary[['SSOC5dCode', 'finalWageMin', 'finalWageMax']], how='left', left_on='Dest Job SSOC', right_on='SSOC5dCode' )
    del df_cwf['SSOC5dCode']
    df_cwf.rename(columns={'finalWageMin':'Dest Job Min Wage', 'finalWageMax': 'Dest Job Max Wage'}, inplace=True)
    df_cwf[['Dest Job Max Wage', 'Dest Job Min Wage']] = df_cwf[['Dest Job Max Wage', 'Dest Job Min Wage']].fillna(0)
    df_cwf[['Dest Job Max Wage', 'Dest Job Min Wage']] = df_cwf[['Dest Job Max Wage', 'Dest Job Min Wage']].round(0)

    return df_cwf

def populateDemandforDestJobs(df_cwf, df_demand):
    df_demand = df_demand.sort_values(by='count', ascending=False)
    df_demand['Cumulative Count'] = df_demand['count'].cumsum()
    df_demand['Percentage'] = df_demand['Cumulative Count'] / df_demand['count'].sum()

    # add a new column "Demand" and populate values based on percentage
    df_demand.loc[df_demand['Percentage'] >= 0.8, 'Demand'] = 'Very Low'
    df_demand.loc[(df_demand['Percentage'] >= 0.6) & (df_demand['Percentage'] < 0.8), 'Demand'] = 'Low'
    df_demand.loc[(df_demand['Percentage'] >= 0.4) & (df_demand['Percentage'] < 0.6), 'Demand'] = 'Medium'
    df_demand.loc[(df_demand['Percentage'] >= 0.2) & (df_demand['Percentage'] < 0.4), 'Demand'] = 'High'
    df_demand.loc[df_demand['Percentage'] < 0.2, 'Demand'] = 'Very High'
    # rename columns
    df_demand = df_demand.rename(columns={'SSOC5dCode': 'SSOC', 'count': '2022 Job Posting Count'})

    df_demand['SSOC'] = df_demand['SSOC'].astype(str)
    df_cwf = pd.merge(df_cwf, df_demand[['SSOC', '2022 Job Posting Count', 'Demand']], how='left', left_on='Dest Job SSOC', right_on='SSOC' )
    del df_cwf['SSOC']
    df_cwf.rename(columns={'Demand': 'Dest Job Demand'}, inplace=True)
    df_cwf[['2022 Job Posting Count']] = df_cwf[['2022 Job Posting Count']].fillna(0)
    df_cwf[['Dest Job Demand']] = df_cwf[['Dest Job Demand']].fillna("Very Low")

    return df_cwf

def extractSkillsfromUploadedFile (df_uploaded, df_skills):
    id_list = []
    skill_list = []
    skill_form_list = []

    for _id, desc in zip(df_uploaded.iloc[:, 0], df_uploaded.iloc[:, 2]):
        desc = clean_text(str(desc))
        skill_dict = extract_skills(desc)
        if "extractions" in skill_dict and isinstance(skill_dict["extractions"], dict):
            for skill_id, skill_info in skill_dict["extractions"].items():
                id_list.append(_id)
                skill_list.append(skill_info['skill_title'])
                skill_form_list.append(skill_info['skill_type'])
        else:
            id_list.append(_id)
            skill_list.append("no skill extracted")
            skill_form_list.append("NA")

    df_skills['Source Job Role'] = id_list
    df_skills['Source SEA Title'] = skill_list
    df_skills['skill_form'] = skill_form_list
    df_skills = df_skills.dropna()

    return df_skills

def convertTSCtoSEA (df_sea, df_sfw, df_item_jobroles):
    df_sea = df_sea[['11K TSC_CCS Title', '2K Skills Title']].drop_duplicates()
    df_sea = df_sea.rename(columns={'11K TSC_CCS Title': 'TSC Title', '2K Skills Title': 'SEA Title'})
    df_sea = df_sea.apply(lambda x: x.astype(str).str.lower())

    df_sfw['stj'] = df_sfw['Sector'].str.lower() + df_sfw['Track'].str.lower() + df_sfw['Job Role'].str.lower()
    df_sfw['stj'] = df_sfw['stj'].str.replace(" /", "/")
    df_sfw['stj'] = df_sfw['stj'].str.replace("/ ", "/")
    df_sfw = df_sfw[['stj', 'Sector', 'Track', 'Job Role', 'TSC Title']].drop_duplicates()
    df_sfw['TSC Title'] = df_sfw['TSC Title'].str.lower()

    df_item_jobroles['stj'] = df_item_jobroles['sector'] + df_item_jobroles['track'] + df_item_jobroles['job_role']
    df_item_jobroles = df_item_jobroles[['stj', 'role_id','sector','track','job_role']]
    df_item_jobroles.head(3)

    df_sfw = pd.merge(df_sfw, df_item_jobroles[['stj', 'role_id']], on="stj", how="left")
    df_sfw = df_sfw[['role_id', 'Sector', 'Track', 'Job Role', 'TSC Title']]

    df_sfw = pd.merge(df_sfw, df_sea[['TSC Title', 'SEA Title']], on="TSC Title", how="left")
    # fill null values in "SEA Title" with values from "TSC Title"
    df_sfw['SEA Title'] = df_sfw['SEA Title'].fillna(df_sfw['TSC Title'])

    return df_sfw

def mergeOutputwithSEASkills (df_cwf, df_ssoc, df_sfw, df_skills, df_skill_demand):
    df_cwf = pd.merge(df_cwf, df_ssoc[['Job Role ID', 'SSOC 2020']], left_on="Dest Job ID", right_on="Job Role ID", how="left")
    df_cwf = df_cwf.drop(['Job Role ID'], axis=1)
    df_cwf = df_cwf.rename(columns={'SSOC 2020': 'Dest SSOC'})

    df_cwf = pd.merge(df_cwf, df_sfw[['role_id', 'SEA Title']], left_on="Dest Job ID", right_on="role_id", how="left")
    df_cwf = df_cwf.rename(columns={'SEA Title': 'Dest Skill'})
    df_cwf = df_cwf.drop(['role_id'], axis=1)
    df_cwf.head(3)

    # Merge the two dataframes based on 'Source Job Role' column
    df_cwf = pd.merge(df_cwf, df_skills[['Source Job Role', 'Source SEA Title']], left_on=['Source Job Role', 'Dest Skill'], right_on=['Source Job Role','Source SEA Title'], how='left')

    # Replace all NaN values with "Yes"
    df_cwf['Source SEA Title'].fillna("Yes", inplace=True)

    # Replace all non-NaN values with "No"
    df_cwf.loc[df_cwf['Source SEA Title'] != "Yes", 'Source SEA Title'] = "No"

    df_cwf = df_cwf.rename(columns={'Source SEA Title':'Dest Skill Need Top Up?'})

    df_skill_demand = df_skill_demand[['year', '5D-SSOC', 'SSOC 2015 Title', 'skill', 'total', 'percentage', 'rank']]
    df_skill_demand = df_skill_demand[df_skill_demand['year']==2022]

    df_cwf = pd.merge(df_cwf, df_skill_demand[['5D-SSOC', 'skill', 'total']], left_on=["Dest SSOC", "Dest Skill"], right_on=["5D-SSOC", 'skill'], how="left")
    df_cwf['total'] = df_cwf['total'].fillna(0).astype(int)
    df_cwf = df_cwf.drop(['5D-SSOC', 'skill', 'Dest SSOC'], axis=1)
    df_cwf = df_cwf.rename(columns={'total': 'Dest Skill Demand in Dest SSOC in 2022'})

    return df_cwf





