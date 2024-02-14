import streamlit as st
import pandas as pd
import io
from jstr import run_jstr

def main():
    st.title("JSTR Self Help Tool")

    # Sample data for the table
    sample_data = [
        {"Job Title": "Data Analyst", "Job Description": "Analyse data and generate insights...", "Skills": "Advanced Python with Machine Learning"},
    ] 
    # Display sample table
    st.write("Upload a CSV with Job Title, Job Description and Skills to generate JSTR output.")
    st.write("Sample CSV format for reference.")  
    st.table(pd.DataFrame(sample_data))

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read uploaded CSV file
        df = pd.read_csv(uploaded_file, encoding='utf-8', errors='ignore')
        st.write("Uploaded File:")
        st.dataframe(df)

        if st.button("Run JSTR"):
            st.empty()
            st.write("Processing takes about 1 to 2 min per job role")
            # Display the animated GIF while processing
            processing_gif = st.image("walk.gif")

            processed_df = run_jstr(df)

            # Remove the animated GIF and replace it with the result or additional content
            processing_gif.empty()

            # Display processed result
            st.write("JSTR Output File:")
            st.dataframe(processed_df)

            # Provide download link for processed CSV
            csv_download_link(processed_df)

def csv_download_link(dataframe):
    # Create a downloadable link for the processed CSV
    csv_buffer = io.StringIO()
    dataframe.to_csv(csv_buffer, index=False)
    st.download_button("Download Results", data=csv_buffer.getvalue(), file_name='jstr_output.csv')

if __name__ == "__main__":
    main()
