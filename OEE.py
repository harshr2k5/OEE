import pandas as pd
df = pd.read_excel('Production Shift-Wise.xlsx')

df['Production Time (Min)'] = df['Planned Prod Time (Min)'] - df['Total DownTime (Min)']
df['Availability (%)'] = df['Production Time (Min)']/df['Planned Prod Time (Min)'] * 100
df['Performance (%)'] = df['Parts Produced'] / df['Target Quantity'] * 100
df['OEE (%)'] = df['Availability (%)'] * df['Performance (%)'] * df['Quality (%)'] / 10000


meanOEE = df['OEE (%)'].mean()
stdOEE = df['OEE (%)'].std()
df['Z-scores'] = (df['OEE (%)'] - meanOEE) / stdOEE

print("Mean of OEE (%):", meanOEE)
print("Standard Deviation of OEE (%):", stdOEE)



df.to_excel('Processed Data.xlsx', index=False)





