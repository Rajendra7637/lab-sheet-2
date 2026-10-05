# Lab Sheet-02: Data Preprocessing

**Name:** <your name>
**Roll No:** <your roll number>
**Course:** MCA, 3rd Semester (2026-2027)
**University:** COER University, Roorkee

## About this project

This is my work for Lab Sheet-02 (Lab Assessment of Data Preprocessing).
It has 35 programs. They show how to clean and prepare a dataset before
using it in a machine learning model: fix missing values, handle outliers,
scale the numbers, encode text columns, and create new features.

## What is in the folder

| File / Folder | What it does |
|---|---|
| `lab_sheet_02_all_programs.py` | All 35 programs in one file. Each program is its own cell. |
| `datasets/housing_raw.csv` | The original dataset (house prices). It has missing values and outliers. |
| `processed/` | Dataset saved after each major step, and the final cleaned file. |
| `outputs/` | The graphs saved by the programs. |
| `requirements.txt` | The list of libraries to install. |

## Dataset

`housing_raw.csv` has 200 houses and 12 columns, like city, area, bedrooms,
age, distance to metro, listing date and price (in lakh).

## Processed files

- `01_missing_handled.csv` - after filling missing values
- `02_outliers_treated.csv` - after capping outliers
- `03_standardized.csv` - after standard scaling
- `final_preprocessed.csv` - final data, ready for machine learning

## Libraries used

NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, Jupyter Notebook.
Python version: 3.11 or above.

## How to run it

1. Open the project folder in VS Code.
2. Open the terminal and make a virtual environment:

   ```
   python -m venv venv
   venv\Scripts\activate
   ```

   (On Linux or Mac, use `source venv/bin/activate`.)

3. Install the libraries:

   ```
   pip install -r requirements.txt
   ```

4. Run the whole file:

   ```
   python lab_sheet_02_all_programs.py
   ```

   Or open the file in VS Code and click **Run Cell** above any program.
   You can also open the `.ipynb` file and run the cells there.

## What the programs cover

- **Programs 1-10:** Find missing values, remove rows or columns, and fill
  values with mean, median, mode, forward fill and backward fill.
- **Programs 11-18:** Find outliers with IQR and Z-score, show them in graphs,
  and remove, replace or cap them.
- **Programs 19-26:** Min-Max, Standard, Robust and Max Absolute scaling,
  and a comparison of all four.
- **Programs 27-35:** Label, One-Hot and Binary encoding, new features, date
  features, log transform, feature selection, and the final dataset.

## Observations

- Removing every row with a missing value deleted most of the data, because
  one column was missing in more than half of the rows. Dropping that column
  first is better.
- Outliers in area and price pulled the mean up. The median was a safer choice.
- Log transform made the skewed columns more balanced.

## Conclusion

<Write 2-3 lines in your own words about what you learned.>
