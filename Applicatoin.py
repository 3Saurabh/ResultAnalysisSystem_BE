import streamlit as st
import pandas as pd
import numpy as np
import re
from PyPDF2 import PdfReader
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)


def extract_pdf_data(data, subjects):
    reader = PdfReader(data)
    name, seat, sgpa = [], [], []
    scores = {sub: [] for sub in subjects}

    num_pages = int(st.number_input('Please Enter the Number of Pages:', min_value=1))
    if not num_pages:
        return None

    for page_num in range(num_pages):
        text = reader.pages[page_num].extract_text()
        content = text.splitlines()
        main_index = next((i.find("Tot%") for i in content if "COURSE NAME" in i), None)

        for line in content:
            if 'SEAT ' in line:
                seat.append(line[10:20].strip())
                name.append(line[28:63].strip())
            elif 'SGPA' in line and "TOTAL CREDITS" in line:
                s = [float(n) for n in re.findall(r'-?\d+\.?\d*', line)]
                sgpa.append(s[0] if len(s) > 1 else 0)
            else:
                for sub in subjects:
                    if sub.split()[0] in line:
                        scores[sub].append(line[main_index:main_index + 5])

    df = pd.DataFrame({'SEAT NO.': seat, 'Candidate Name': name, **scores, 'SGPA': sgpa})
    df = df.applymap(lambda x: np.nan if x in ["", "FF"] else x)
    df.fillna(0, inplace=True)
    df[subjects] = df[subjects].astype(float)
    return df


def display_results(df):
    st.subheader("TOP 5 Students")
    st.dataframe(df.nlargest(5, 'SGPA')[['SEAT NO.', 'Candidate Name', 'SGPA']])

    st.subheader("Subject Wise Failure Percentage")
    failures = df.iloc[:, 2:-1].apply(lambda x: (x < 40).sum())
    failure_df = pd.DataFrame(
        {"Subject": failures.index, "Failures": failures.values, "Failure %": (failures / len(df) * 100).round(2)})
    st.dataframe(failure_df)

    st.subheader("Classification of Candidates")
    categories = {"Distinction": 7.75, "First Class": 6.75, "Higher Second Class": 6.25, "Second Class": 5.5,
                  "Pass Class": 4.65}
    classification = {key: (df['SGPA'] >= val).sum() for key, val in categories.items()}
    classification['Fail'] = (df['SGPA'] < 4.65).sum()
    classification['Pass %'] = (sum(classification.values()) / len(df)) * 100
    classification['Fail %'] = 100 - classification['Pass %']
    st.dataframe(pd.DataFrame([classification]))


def visualize_data(df):
    st.subheader("SGPA Distribution")
    sns.histplot(df['SGPA'], kde=True, bins=10)
    st.pyplot()

    st.subheader("Custom Plots")
    cols = st.multiselect("Select Columns", df.columns[2:])
    if cols:
        plot_type = st.selectbox("Plot Type", ["bar", "line", "area"])
        getattr(st, f"{plot_type}_chart")(df[cols])


def main():
    st.sidebar.title("Student Result Analysis")
    sem_options = {"S.E First Semester": ["210241 Discrete Mathematics", "210242 Data Structures"],
                   "T.E First Semester": ["310241 Database Management Systems", "310242 Theory of Computation"],
                   "B.E First Semester": ["410241 Design and Analysis of Algorithms", "410242 Machine Learning"]}

    choice = st.sidebar.radio("Select Option", list(sem_options.keys()))
    data_file = st.file_uploader("Upload PDF", type=["pdf"])

    if data_file:
        df = extract_pdf_data(data_file, sem_options[choice])
        if df is not None:
            display_results(df)
            if st.checkbox("Visualize Data"):
                visualize_data(df)


if __name__ == "__main__":
    main()
