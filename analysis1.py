import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
import streamlit as st
import numpy as np
from io import StringIO
import pandas as pd
from scipy.stats import levene
from itertools import combinations
import plotly.express as px
import psycopg2
import os
import dotenv
from psycopg2.extras import execute_values

def upload_to_db(df):
    if not isinstance(df, pd.DataFrame):
        print("Invalid input: Expected a DataFrame.")
        return
    
    try:
        dotenv.load_dotenv()
        conn = psycopg2.connect(
            host=os.getenv('DATABASE_HOST'),
            dbname=os.getenv('DATABASE_NAME'),
            user=os.getenv('DATABASE_USER'),
            password=os.getenv('DATABASE_PASSWORD'),
            port=os.getenv('DATABASE_PORT')
        )
        cur = conn.cursor()

        insert_spectrum_stmt = """
        INSERT INTO dataframe (sample, wavelengths, absorbances) VALUES %s;
        """
        values = [
            (row["Sample"], df.columns[1:].tolist(), row[1:].tolist())
            for _, row in df.iterrows()
        ]
        execute_values(cur, insert_spectrum_stmt, values)
        conn.commit()
        cur.close()
        conn.close()
        print("Data uploaded successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")




def is_csv(filename: str) -> bool:
    """
    Check if the file is a CSV file.

    Args:
        filename (str): The name of the file.

    Returns:
        bool: True if the file is a CSV file, False otherwise.
    """
    return filename.lower().endswith(".csv")

def levene_test(df):
    df.drop("Controlo", axis=1, inplace=True)

    columns = sorted(df.columns)
    results = []
    for r in range(2, len(columns) + 1):
        column_combinations = combinations(columns, r)
        for combo in column_combinations:
            combo_df = df[list(combo)]
            _, p_value = levene(
                combo_df.iloc[:, 0].dropna().to_numpy(),
                combo_df.iloc[:, 1].dropna().to_numpy(),
                center="mean",
            )

            results.append(
                {
                    "Combination": f"{combo[0]} vs {combo[1]}",
                    "p-value": f"{p_value:.2e}",
                }
            )
    return pd.DataFrame(results)


def import_multiple_csv(files, len_name):
    df = pd.DataFrame()
    names = []
    absorbances = []
    
    for uploaded_file in files:
        if not uploaded_file.name.endswith('.csv'):
            continue


        file_content = uploaded_file.read().decode("ISO-8859-1")
        lines = file_content.split('\n')
        
        header_line = None
        for i, line in enumerate(lines):
            if line.startswith('Wavelength'):
                header_line = i
                break

        if header_line is not None:
            csv_content = '\n'.join(lines[header_line:])
            df_aux = pd.read_csv(StringIO(csv_content), delimiter=",")
            
            name = uploaded_file.name.replace('_', '-')
            if "CONTROLO" in name.upper():
                name = "Controlo"
            else:
                name = "-".join(name.split("-")[:len_name])
            names.append(name)
            
            if 'Absorbance (AU)' in df_aux.columns:
                absorbances.append(df_aux["Absorbance (AU)"].to_list())
            else:

                print(f"Column 'A' not found in file: {uploaded_file.name}")
        else:
            print(f"No 'Wavelength' header found in file: {uploaded_file.name}")

    df["Sample"] = names
    if absorbances:  
        df_spectra = pd.DataFrame(absorbances)
        return pd.concat([df, df_spectra], axis=1)
    else:
        return df
    
def plot(df):

    df_long = pd.melt(df, id_vars=['Sample'], var_name='Measurement', value_name='Absorbance')
    
    df_long['Measurement'] = pd.to_numeric(df_long['Measurement'])
    
    fig = px.line(df_long, x='Measurement', y='Absorbance', color='Sample', title="Spectra")

    st.plotly_chart(fig)


    
def msc(matrix, reference="mean"):
    if reference == "mean":
        matrix_mean = np.mean(matrix, axis=1, keepdims=True)
        matrix_msc = matrix - matrix_mean
    elif reference == "median":
        matrix_median = np.median(matrix, axis=1, keepdims=True)
        matrix_msc = matrix - matrix_median
    else:
        matrix_msc = matrix - reference
    return matrix_msc


def create_boxplot(df: pd.DataFrame):
    categories = df["Sample"].unique().tolist()
    data = []
    labels = ["Controlo"]
    data_preprocessed = []

    i = 0
    for idx, category in enumerate(categories):
        if category.upper() == "CONTROLO":
            continue

        df_sep = df[df["Sample"].isin(["Controlo", category])]

        x = df_sep.iloc[:, 1:].to_numpy()

        x_preprocessed = msc(x[:, :], reference="mean")
        x_preprocessed = msc(x_preprocessed[:, :], reference="mean")
        x_preprocessed = savgol_filter(
            x_preprocessed, window_length=9, polyorder=2, deriv=2
        )

        data_preprocessed.extend(x_preprocessed.tolist())  

        pca = PCA(n_components=2)
        x_pca = pca.fit_transform(x_preprocessed)

        x_pca_reshape = x_pca[: len(df_sep), :]

        df_pca = pd.DataFrame({"PC1": x_pca_reshape[:, 0], "Cat": df_sep["Sample"]})

        coating = df_pca[df_pca["Cat"] != "Controlo"]
        coating_mean = coating["PC1"].mean()
        coating["Sub"] = coating["PC1"] - coating_mean

        if i == 0:
            control = df_pca[df_pca["Cat"] == "Controlo"]
            control_mean = control["PC1"].mean()
            control["Sub"] = control["PC1"] - control_mean
            data.append(control["Sub"].to_numpy())
            i = 2

        data.append(coating["Sub"].to_numpy())
        labels.append(category)

    df_variances = pd.DataFrame()
    for x, label in zip(data, labels):
        aux = pd.DataFrame(x, columns=[label])
        df_variances = pd.concat([df_variances, aux], axis=1)


    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(data, tick_labels=labels)  
    ax.set_ylabel("Intensity")
    ax.set_title("Boxplot of different coatings")
    return fig, df_variances

def main():
    files = st.file_uploader("Upload files", type="csv", accept_multiple_files=True)
    len_name = st.number_input(label="Number of words per sample", min_value=1, value=2, step=1)
    df = import_multiple_csv(files, len_name) 

    if len(df) != 0:
        upload_to_db(df)  
        st.title("Raw Spectra")
        plot(df)

        st.title("Box Plots")
        boxplot_fig, variance_df = create_boxplot(df)
        st.pyplot(boxplot_fig)

        st.title("Statistics Test")
        df_results = levene_test(variance_df)

        df_results["p-value"] = df_results["p-value"].astype(float)
        df_results.drop_duplicates(
            subset=["Combination", "p-value"], inplace=True, ignore_index=True
        )
        
        st.dataframe(df_results)

        significant_combinations = df_results[df_results["p-value"] < 0.05]


        message = f"We can conclude that the variances **are significantly different** at the 5% level of significance for samples:\n\n"
        message += "\n".join(
            [
                f"- {sample}"
                for sample in significant_combinations["Combination"].tolist()
            ]
        )

        st.write(message)

        non_significant_combinations = df_results[df_results["p-value"] >= 0.05]

        message = f"There is **not enough evidence** to conclude that the variances are different at 5% level of significance for samples:\n\n"
        message += "\n".join(
            [
                f"- {sample}"
                for sample in non_significant_combinations["Combination"].tolist()
            ]
        )
        st.write(message)



if __name__ == "__main__":
    main()

