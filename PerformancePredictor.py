import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import re
import warnings

processed_data = pd.read_excel('Processed Data Final.xlsx')
downtime_data = pd.read_excel('Downtime Durations and Reasons.xlsx')

downtime_agg = downtime_data.groupby(['Date', 'Machine', 'Shift']).agg({
    'Downtime(Minutes)': ['sum', 'mean', 'count', 'std', 'max', 'min'],
    'Reason': (lambda x: x.mode().iat[0] if not x.mode().empty else 'No Reason Selected')
}).reset_index()

downtime_agg.columns = [
    'Date', 'Machine', 'Shift',
    'Total_Downtime_Duration', 'Avg_Downtime_Duration',
    'Downtime_Event_Count', 'Std_Downtime_Duration',
    'Max_Downtime_Duration', 'Min_Downtime_Duration',
    'Most_Common_Reason'
]

if 'Reason' in downtime_data.columns:
    reason_counts = downtime_data['Reason'].value_counts()
    top_reasons = reason_counts.head(10).index.tolist()
    for reason in top_reasons:
        reason_col_name = f"Reason_{reason.replace(' ', '_').replace('/', '_')}"
        reason_counts_df = (downtime_data[downtime_data['Reason'] == reason]
            .groupby(['Date', 'Machine', 'Shift']).size()
            .reset_index(name=reason_col_name))
        downtime_agg = downtime_agg.merge(
            reason_counts_df, on=['Date','Machine','Shift'], how='left'
        )
        downtime_agg[reason_col_name] = downtime_agg[reason_col_name].fillna(0)

merged_data = processed_data.merge(downtime_agg, on=['Date','Machine','Shift'], how='left')


merged_data.to_excel('Processed Data (New).xlsx', index=False)

downtime_cols = [c for c in merged_data.columns if c.startswith('Reason_')] + [
    'Total_Downtime_Duration','Avg_Downtime_Duration','Downtime_Event_Count',
    'Std_Downtime_Duration','Max_Downtime_Duration','Min_Downtime_Duration'
]
for col in downtime_cols:
    if col in merged_data.columns:
        merged_data[col] = merged_data[col].fillna(0)

def infer_family(machine):
    if pd.isna(machine):
        return np.nan
    s = str(machine).strip().upper()
    s = s.replace(' ', '-')
    m = re.match(r'^([A-Z]+)', s)
    return m.group(1) if m else s

if 'Machine' not in merged_data.columns:
    raise ValueError("Column 'Machine' is missing from the merged dataset.")
merged_data['Machine_Family'] = merged_data['Machine'].apply(infer_family)

features = ['Total_Downtime_Duration', 'Avg_Downtime_Duration', 'Target Quantity']
target = 'Performance (%)'

numeric_df = merged_data.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
df = merged_data.loc[numeric_df.dropna(subset=features + [target]).index].copy()

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, df.index, test_size=0.2, random_state=42
)

train_df = df.loc[idx_train]

if 'Machine_Family' not in train_df.columns:
    raise ValueError("Column 'Machine_Family' was not created properly.")

family_stats = (
    train_df
    .groupby('Machine_Family')[['Total_Downtime_Duration','Avg_Downtime_Duration']]
    .mean()
)

overall_means = train_df[['Total_Downtime_Duration','Avg_Downtime_Duration']].mean()

family_stats_dict = family_stats.to_dict('index')
family_set = set(family_stats.index)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n=== Model Performance ===")
print(f"MAE       : {mae:.4f}")
print(f"RMSE      : {rmse:.4f}")
print(f"R² Score  : {r2:.4f}")

print("\n--- Predict Performance ---")
try:
    family_in = input("Enter Machine Family (e.g., CNC or VMC): ").strip().upper().replace(' ', '-')
    target_quantity = float(input("Enter Required Parts (Target Quantity): ").strip())

    choice = input("Use historical averages for this family? [Y/n]: ").strip().lower()
    use_hist = not (choice in ["n", "no"])

    if use_hist:
        fam_key = family_in if family_in in family_set else family_in.rstrip('-')
        if fam_key in family_set:
            td = family_stats_dict[fam_key]['Total_Downtime_Duration']
            ad = family_stats_dict[fam_key]['Avg_Downtime_Duration']
            print(f"Using historical downtime for family '{fam_key}': Total≈{td:.2f}, Avg≈{ad:.2f}")
        else:
            td = overall_means['Total_Downtime_Duration']
            ad = overall_means['Avg_Downtime_Duration']
            print(f"Family '{family_in}' not found in training history. "
                  f"Falling back to overall means: Total≈{td:.2f}, Avg≈{ad:.2f}")
    else:
        td = float(input("Enter expected TOTAL downtime (minutes): ").strip())
        ad = float(input("Enter expected AVERAGE downtime per event (minutes): ").strip())
        if td < 0 or ad < 0:
            raise ValueError("Downtime values must be non-negative.")
        print(f"Using manual downtime inputs: Total={td:.2f}, Avg={ad:.2f}")

    user_input = pd.DataFrame([[td, ad, target_quantity]], columns=features)
    user_input_scaled = scaler.transform(user_input)
    predicted_performance = float(model.predict(user_input_scaled)[0])

    print(f"\nPredicted Performance (%): {predicted_performance:.4f}")

except Exception as e:
    print(f"Error: {e}")


    # === Print average total and average average downtime per machine ===
machine_avg = (
    merged_data
    .groupby('Machine')[['Total_Downtime_Duration', 'Avg_Downtime_Duration']]
    .mean()
    .rename(columns={
        'Total_Downtime_Duration': 'Avg Total Downtime (min)',
        'Avg_Downtime_Duration': 'Avg of Avg Downtime (min)'
    })
    .sort_index()
)

print("\n=== Average Downtime per Machine ===")
print(machine_avg.to_string(formatters={
    'Avg Total Downtime (min)': '{:.2f}'.format,
    'Avg of Avg Downtime (min)': '{:.2f}'.format
}))


import matplotlib.pyplot as plt

corr = merged_data[['Performance (%)',
                    'Total_Downtime_Duration',
                    'Avg_Downtime_Duration',
                    'Target Quantity',
                    'Downtime_Event_Count',
                    'Max_Downtime_Duration']].corr()

plt.figure(figsize=(8,6))
plt.imshow(corr, cmap='coolwarm', interpolation='none')
plt.colorbar(label='Correlation')
plt.xticks(range(len(corr)), corr.columns, rotation=45, ha='right')
plt.yticks(range(len(corr)), corr.columns)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()
