import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming your data is in a DataFrame named 'df'
# If not, read your data into a DataFrame using pd.read_csv or another appropriate method
excel_file_path = 'Trader Intern Data Task (2) (1) (1).xlsx'

# Read the Excel file into a DataFrame
df = pd.read_excel(excel_file_path)
print(df.columns)

# Convert the 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Group by date and sum the values for each metric
grouped_df = df.groupby('Date').agg({'Impressions': 'sum', 'Clicks': 'sum', 'Total Conversions': 'sum'}).reset_index()

# Plotting the trend over time
plt.figure(figsize=(12, 6))
sns.lineplot(x='Date', y='Impressions', data=grouped_df, label='Impressions')
sns.lineplot(x='Date', y='Clicks', data=grouped_df, label='Clicks')
sns.lineplot(x='Date', y='Total Conversions', data=grouped_df, label='Total Conversions')

plt.title('Trend in Impressions, Clicks, and Total Conversions Over Time')
plt.xlabel('Date')
plt.ylabel('Count')
plt.legend()
plt.show()
