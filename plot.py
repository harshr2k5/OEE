import matplotlib.pyplot as plt
import pandas as pd 

avg_calc_sheet = pd.read_excel('Downtime Durations and Reasons.xlsx', sheet_name='Sheet1')
avg_by_reason = avg_calc_sheet.groupby('Reason')['Downtime(Minutes)'].mean().sort_values(ascending=True)
print(avg_by_reason)

plt.figure(figsize=(12, 8))
plt.barh(avg_by_reason.index, avg_by_reason.values, color='steelblue')
plt.xlabel('Average Downtime (Minutes)', fontweight='bold')
plt.ylabel('Reason', fontweight='bold')
plt.title('Average Downtime Duration by Reason', fontweight='bold', fontsize=14)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('avg downtime by reason.png', dpi=300, bbox_inches='tight')
plt.show()