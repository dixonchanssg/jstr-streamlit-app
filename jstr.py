import pandas as pd
import numpy as np
import helper as hp

# df: uploaded dataset from streamlit
def run_jstr(df):

    df_uploaded = df.copy()

    ##########################################################
    # Load all input files required to generated JSTR output #
    ##########################################################
    # All SFw job roles and their associated CWF Clusters 
    df_cwf = pd.read_excel('data_raw/job role_cwf.xlsx')
    # cwf->kt reference table (this is an output from preprocessing step 0.1)
    cwf_kt = pd.read_csv("data_processed/cwf_kt.csv")
    # Read SFw Database - Job Role (Key) (To extract Average TSC Levels - use it to filter unrealistic adjacent pairs)
    sfw_df = pd.read_excel('data_raw\SFw Database_27Jul_2022.xlsx', sheet_name='Job Role (Key)')
    # SSOC->SFw Job Role Reference Table
    df_ssoc = pd.read_excel('data_raw\SFw Database_27Jul_2022.xlsx', sheet_name='Job Role_SSOC')
    # SSOC->Wages Reference Table
    df_salary = pd.read_csv('data_raw\wages_2022.csv')
    # SSOC->Demand Reference Table
    df_demand = pd.read_csv('data_raw\jobdemand_2022.csv')
    # 11K->2K Skills SEA Mapping Table
    df_sea = pd.read_excel("data_raw/2k_to_11k_mapping_include_skills_tag_20220914_SIPD_V02.xlsx", sheet_name="2k_to_11k_mapping_include_skill")
    # SFw Job Role->TSC Mapping Table
    df_sfw = pd.read_excel("data_raw/SFw Database_27Jul_2022.xlsx", sheet_name="Job Role_TSC")
    # Item->job_roles IDs
    df_item_jobroles = pd.read_csv("data_raw/item_job_roles.csv")
    # Skills_demand table
    df_skill_demand = pd.read_csv("data_raw/stats_year_ssoc_tsc.csv")


    # Extract values from the first column in the uploaded data and populate the uploaded_job_titles list
    uploaded_job_titles = []
    uploaded_job_titles = df.iloc[:, 0].tolist()


    # Part 1.3 Add raw job description for the uploaded job roles in 'cwf_kt.csv', find most similar cwf clusters for each job roles and stores output to dataframe list


    # Add raw job description for the uploaded job roles in 'cwf_kt.csv', 
    # find most similar cwf clusters for each job roles and stores output to dataframe list
    # Create an empty list to store cwf_kt DataFrames for all the job titles in uploaded dataset
    all_cwf_kt_df = []

    # Loop through each row and call generate_similar_cwf_kt() for the value in the first column
    for index, row in df.iterrows():
        # Check if job_title is NaN
        # if pd.isna(row.iloc[0]):
        #    break
        job_title = row.iloc[0]
        job_desc = row.iloc[1]
        result_df = hp.generate_similar_cwf_kt(index, job_title, job_desc)
        all_cwf_kt_df.append(result_df)


    # Part 1.4 - Populate all_cwf_kt_df [ ] with Function ID, Sector, Track, Job role, L1/2/3 clusters

    # Populate all_cwf_kt_df [ ] with Function ID, Sector, Track, Job role, L1/2/3 clusters
    # Create an empty list to store the resulting DataFrames
    populated_df = []

    # Iterate over the list of DataFrames and apply the function
    for df in all_cwf_kt_df:
        result_df = hp.populateCWFClusters(df, cwf_kt, df_cwf)
        populated_df.append(result_df) 

    # Part 2 - Prepare df_jobrole_cwf for JSTR
    df_jobrole_cwf = hp.prepareJobroleCWF(df_cwf, populated_df, uploaded_job_titles)

    # Part 3.1 - Data preparation to Run JSTR
    JobCWF_df = hp.dataPrepForJSTR(df_jobrole_cwf)

    # Part 3.2 - Compute Job Adjacency Scores based on CWF Clusters
    df_cwf = hp.computeJobAdjacency(JobCWF_df)

    # Part 3.3 - Populate job pairs with job title
    df_cwf = hp.populateJobPairsWithJobTitle(df_jobrole_cwf, df_cwf, sfw_df)


    # 3.4 Populate SSOC for destination jobs
    df_cwf = hp.populateSSOCforDestJobs(df_cwf, df_ssoc)

    # 3.5 Populate Jobtech salary data
    df_cwf = hp.populateSalaryforDestJobs(df_cwf, df_salary)

    # 3.6 Populate Jobtech job demand data
    df_cwf = hp.populateDemandforDestJobs(df_cwf, df_demand)

    # 4.1 Extract Skills from Uploaded Dataset
    df_skills = pd.DataFrame()
    df_skills = hp.extractSkillsfromUploadedFile(df_uploaded, df_skills)

    # 4.2 Convert SFw Database JobRole-to-TSC to JobRole-to-SEA
    df_sfw = hp.convertTSCtoSEA (df_sea, df_sfw, df_item_jobroles)


    # 4.3 Load JSTR Output and populate the skills data
    # 4.4 Merge JSTR Output with SEA Skills
    df_cwf = hp.mergeOutputwithSEASkills(df_cwf, df_ssoc, df_sfw, df_skills, df_skill_demand)

    return df_cwf