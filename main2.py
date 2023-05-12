import pandas as pd
import seaborn as sns
from fpgrowth import fpgrowth
from association_rules import association_rules
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

Hepatitis_df = pd.read_csv('hepatitis.csv')
Lung_cancer_df=pd.read_csv('lung_cancer.csv')

Hepatitis_df['histology'] = (Hepatitis_df['histology'] == 2)
columns = ['sex', 'steroid', 'antivirals', 'fatigue', 'malaise', 'anorexia', 'liver_big', 'liver_firm', 'spleen_palable', 'spiders', 'ascites', 'varices']
Hepatitis_df[columns] = (Hepatitis_df[columns] == 2)
Hepatitis_frequent_itemsets = fpgrowth(Hepatitis_df[columns], min_support=0.7, use_colnames=True)
Hepatitis_association_rules = association_rules(Hepatitis_frequent_itemsets, metric="confidence", min_threshold=0.7)
print(Hepatitis_association_rules.to_string())
print("Below is the report for the lung cancer")
# print(Hepatitis_frequent_itemsets.to_string())
# plt.barh(range(len(Hepatitis_frequent_itemsets)), Hepatitis_frequent_itemsets.support, tick_label=Hepatitis_frequent_itemsets.itemsets)
# plt.xlabel('Support')
# plt.ylabel('Itemsets')
# plt.title('Frequent Itemsets')
# plt.show()
# plt.barh(Hepatitis_association_rules['antecedents'].astype(str) + '->' +  Hepatitis_association_rules['consequents'].astype(str), Hepatitis_association_rules['confidence'],color='green')
# plt.xlabel('Confidence')
# plt.ylabel('Association Rule')
# plt.title('Association Rules for Hepatitis Disease Prediction')
# plt.show()

# if 'LUNG_CANCER' in df.columns:
Lung_cancer_df['LUNG_CANCER']=(Lung_cancer_df['LUNG_CANCER']==2)
# Extract the relevant columns for mining
columns =['SMOKING','YELLOW_FINGERS','ANXIETY','PEER_PRESSURE','CHRONIC_DISEASE','FATIGUE ','ALLERGY ','WHEEZING','ALCOHOL_CONSUMING','COUGHING','SHORTNESS_OF_BREATH','SWALLOWING_DIFFICULTY','CHEST_PAIN']
Lung_cancer_df[columns] = (Lung_cancer_df[columns] == 2)
Lung_cancer_frequent_itemsets = fpgrowth(Lung_cancer_df[columns], min_support=0.4, use_colnames=True)
Lung_cancer_association_rules = association_rules(Lung_cancer_frequent_itemsets, metric="confidence", min_threshold=0.7)
print(Lung_cancer_association_rules.to_string())
# print(Lung_cancer_frequent_itemsets[:30].to_string())
# plt.barh(Lung_cancer_association_rules['antecedents'].astype(str) + '->' +  Lung_cancer_association_rules['consequents'].astype(str), Lung_cancer_association_rules['confidence'],color='green')
# plt.xlabel('Confidence')
# plt.ylabel('Association Rule')
# plt.title('Association Rules for Lung Cancer Disease Prediction')
# plt.show()
# plt.barh(range(len(Lung_cancer_frequent_itemsets)), Lung_cancer_frequent_itemsets.support, tick_label=Lung_cancer_frequent_itemsets.itemsets)
# plt.xlabel('Support')
# plt.ylabel('Itemsets')
# plt.title('Frequent Itemsets')
# plt.show()
# fig, ax = plt.subplots(figsize=(12, 8))
# scatter = ax.scatter(association_rules['support'],association_rules['confidence'], alpha=0.8, c=association_rules['lift'], cmap='viridis')
# ax.set_xlabel('Support')
# ax.set_ylabel('Confidence')
# ax.set_title('Association Rules - Hepatitis Disease Prediction')
# plt.colorbar(scatter)
# plt.show()

# scatter_matrix(Hepatitis_association_rules[['support', 'confidence', 'lift']], alpha=0.2, figsize=(6, 6), diagonal='hist')
# plt.show()
#
# scatter_matrix(Lung_cancer_association_rules[['support', 'confidence', 'lift']], alpha=0.2, figsize=(6, 6), diagonal='hist')
# plt.show()

# fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
# sns.heatmap(hepatitis_matrix, annot=True, fmt=".2f", ax=ax[0])
# sns.heatmap(lung_cancer_matrix, annot=True, fmt=".2f", ax=ax[1])
# ax[0].set_title("Hepatitis dataset")
# ax[1].set_title("Lung Cancer dataset")
# plt.show()

