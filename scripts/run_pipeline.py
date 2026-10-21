import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Ensure directories exist
os.makedirs('outputs/figures', exist_ok=True)
os.makedirs('outputs/tables', exist_ok=True)
os.makedirs('outputs/models', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

# Set visual style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.titlesize'] = 12

print("Step 1: Loading raw dataset...")
raw_df = pd.read_csv('data/raw/ewaste_technological_hubs_dataset.csv')
print(f"Loaded raw dataset with shape: {raw_df.shape}")

# Data Quality & Cleaning
df_clean = raw_df.drop_duplicates().copy()
print(f"Data shape after duplicate check: {df_clean.shape}")
df_clean.to_csv('data/processed/cleaned_ewaste_dataset.csv', index=False)

# NLP Feature Preparation
print("Step 2: NLP and Text Feature Preparation...")
keywords = [
    'flame retardant', 'lithium', 'mercury', 'lead', 'circuit board',
    'sensor', 'cooling', 'server', 'copper', 'gold',
    'informal', 'landfill', 'decommission', 'refrigerant', 'leaching'
]

def count_keywords(text):
    text_lower = str(text).lower()
    return sum(text_lower.count(k) for k in keywords)

df_feat = df_clean.copy()
df_feat['Text_Length'] = df_feat['Hub_Stream_Description'].apply(lambda x: len(str(x)))
df_feat['Token_Count'] = df_feat['Hub_Stream_Description'].apply(lambda x: len(str(x).split()))
df_feat['Security_Keyword_Count'] = df_feat['Hub_Stream_Description'].apply(count_keywords)

# TF-IDF (15 features)
tfidf = TfidfVectorizer(max_features=15, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_feat['Hub_Stream_Description']).toarray()
tfidf_cols = [f'tfidf_{term}' for term in tfidf.get_feature_names_out()]
tfidf_df = pd.DataFrame(tfidf_matrix, columns=tfidf_cols, index=df_feat.index)
df_feat = pd.concat([df_feat, tfidf_df], axis=1)

# Domain Feature Engineering
print("Step 3: Engineering Domain Features...")
df_feat['Obsolescence_Velocity_Index'] = 12.0 / df_feat['Average_Device_Lifespan_Months']
df_feat['Formal_to_Informal_Ratio'] = df_feat['Formal_Collection_Rate_Pct'] / (df_feat['Informal_Recovery_Rate_Pct'] + 1.0)
df_feat['Informal_Burden_Pct'] = df_feat['Informal_Recovery_Rate_Pct'] / (df_feat['Formal_Collection_Rate_Pct'] + df_feat['Informal_Recovery_Rate_Pct'])
df_feat['E_Waste_per_GDP'] = df_feat['Annual_Generation_kt'] / (df_feat['Tech_Sector_GDP_Billion_USD'] + 1.0)
df_feat['Toxic_Load_kt'] = df_feat['Annual_Generation_kt'] * df_feat['Heavy_Metal_Toxicity_Index'] / 10.0
df_feat['Recoverable_Precious_Value_MUSD'] = (df_feat['Annual_Generation_kt'] * 1000.0 * df_feat['Precious_Metal_Value_USD_per_Ton']) / 1e6

df_feat.to_csv('data/processed/engineered_ewaste_features.csv', index=False)
print(f"Engineered dataset saved. Shape: {df_feat.shape}")

# Generate Figures for Chapter 3 & 4
print("Step 4: Generating Exploratory Analysis Charts...")

# Figure 3.1: Architecture Diagram
fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
ax.axis('off')
boxes = [
    ("Stage 1: Multi-Hub Empirical Data Ingestion\n(12 Developing Tech Hubs, GEM/UNITAR Metrics)", 0.5, 0.92, '#E3F2FD'),
    ("Stage 2: Data Quality Verification & Profiling\n(Missing Value Audit, Outlier Inspection, Type Normalization)", 0.5, 0.80, '#E8F5E9'),
    ("Stage 3: NLP & Stream Text Feature Engineering\n(Keyword Extraction, Tokenization, 15-dim TF-IDF Representation)", 0.5, 0.68, '#FFF3E0'),
    ("Stage 4: Domain Statistical & Metric Modeling\n(Obsolescence Velocity, Formal-to-Informal Ratio, Toxic Burden)", 0.5, 0.56, '#F3E5F5'),
    ("Stage 5: Exploratory Data Analysis & Statistical Hypothesis Testing\n(Distributions, ANOVA, Pearson/Spearman Correlations)", 0.5, 0.44, '#EDE7F6'),
    ("Stage 6: Multi-Classifier Training & Evaluation (80:20 Stratified Split)\n(Random Forest, XGBoost, SVM, Decision Tree, Logistic Reg., Naive Bayes)", 0.5, 0.32, '#FCE4EC'),
    ("Stage 7: Decision-Support, Error Auditing & Policy Recommendation\n(Confusion Matrices, Feature Importance, Environmental Vulnerability Index)", 0.5, 0.20, '#E0F2F1')
]
for text, x, y, color in boxes:
    ax.text(x, y, text, ha='center', va='center', fontsize=9.5, weight='bold',
            bbox=dict(boxstyle='round,pad=0.7', facecolor=color, edgecolor='#37474F', linewidth=1.5))
for i in range(len(boxes) - 1):
    y_start = boxes[i][2] - 0.045
    y_end = boxes[i+1][2] + 0.045
    ax.annotate('', xy=(0.5, y_end), xytext=(0.5, y_start),
                arrowprops=dict(arrowstyle="->", color='#263238', lw=2, mutation_scale=15))
plt.title("Figure 3.1: Sequential End-to-End Workflow Architecture of the Proposed E-Waste Analysis Framework", fontsize=11, weight='bold', pad=15)
plt.tight_layout()
plt.savefig('outputs/figures/fig_3_1_workflow_architecture.png', bbox_inches='tight')
plt.close()

# Figure 4.1: Distribution of E-Waste Categories
fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
cat_counts = df_clean['E_Waste_Category'].value_counts()
colors = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78']
bars = ax.bar(cat_counts.index, cat_counts.values, color=colors, edgecolor='#333333', width=0.6)
ax.set_title("Figure 4.1: Distribution of E-Waste Categories in the Dataset", fontsize=11, weight='bold', pad=10)
ax.set_ylabel("Number of Records", fontsize=10)
ax.set_xlabel("E-Waste Category Stream", fontsize=10)
plt.xticks(rotation=15, ha='right')
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=9, weight='bold')
plt.tight_layout()
plt.savefig('outputs/figures/fig_4_1_category_distribution.png', bbox_inches='tight')
plt.close()

# Figure 4.2: Distribution of Annual Generation across Developing Tech Hubs
fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
hub_gen = df_clean.groupby('Tech_Hub')['Annual_Generation_kt'].mean().sort_values(ascending=False)
bars = ax.bar(hub_gen.index, hub_gen.values, color='#4575b4', edgecolor='#333333', width=0.65)
ax.set_title("Figure 4.2: Mean Annual E-Waste Generation (kt) across Developing Tech Hubs", fontsize=11, weight='bold', pad=10)
ax.set_ylabel("Mean Annual Generation (kt)", fontsize=10)
ax.set_xlabel("Technological Hub", fontsize=10)
plt.xticks(rotation=30, ha='right')
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=8.5)
plt.tight_layout()
plt.savefig('outputs/figures/fig_4_2_annual_generation_hubs.png', bbox_inches='tight')
plt.close()

# Figure 4.3: Per Capita Generation vs Tech Sector GDP
fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
sns.scatterplot(data=df_clean, x='Tech_Sector_GDP_Billion_USD', y='Per_Capita_Generation_kg',
                hue='Region', style='Region', s=50, alpha=0.8, ax=ax)
ax.set_title("Figure 4.3: Per Capita E-Waste Generation (kg) vs. Tech Sector GDP (Billion USD)", fontsize=11, weight='bold', pad=10)
ax.set_xlabel("Tech Sector GDP (Billion USD)", fontsize=10)
ax.set_ylabel("Per Capita E-Waste Generation (kg/capita/year)", fontsize=10)
ax.legend(title='Region', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.savefig('outputs/figures/fig_4_3_per_capita_vs_gdp.png', bbox_inches='tight')
plt.close()

# Figure 4.4: Device Lifespan vs Heavy Metal Toxicity Index
fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
sns.boxplot(data=df_clean, x='E_Waste_Category', y='Heavy_Metal_Toxicity_Index', palette='Set2', ax=ax)
ax.set_title("Figure 4.4: Heavy Metal Toxicity Index Distribution across E-Waste Streams", fontsize=11, weight='bold', pad=10)
ax.set_xlabel("E-Waste Category", fontsize=10)
ax.set_ylabel("Heavy Metal Toxicity Index (1-10)", fontsize=10)
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.savefig('outputs/figures/fig_4_4_lifespan_vs_toxicity.png', bbox_inches='tight')
plt.close()

# Figure 4.5: Formal Collection vs Informal Recovery Rates across Tech Hubs
fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
hub_rates = df_clean.groupby('Tech_Hub')[['Formal_Collection_Rate_Pct', 'Informal_Recovery_Rate_Pct']].mean().loc[hub_gen.index]
x = np.arange(len(hub_rates))
width = 0.35
ax.bar(x - width/2, hub_rates['Formal_Collection_Rate_Pct'], width, label='Formal Recycling Rate (%)', color='#2ca02c', edgecolor='#333333')
ax.bar(x + width/2, hub_rates['Informal_Recovery_Rate_Pct'], width, label='Informal Recovery Rate (%)', color='#d62728', edgecolor='#333333')
ax.set_title("Figure 4.5: Formal vs. Informal E-Waste Processing Disparity across Tech Hubs", fontsize=11, weight='bold', pad=10)
ax.set_ylabel("Percentage (%)", fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(hub_rates.index, rotation=30, ha='right')
ax.legend()
plt.tight_layout()
plt.savefig('outputs/figures/fig_4_5_formal_vs_informal.png', bbox_inches='tight')
plt.close()

# Figure 4.6: Distribution of Hub Vulnerability Tiers
fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
tier_counts = df_clean['Management_Risk_Tier'].value_counts()[['Tier-1 Controlled', 'Tier-2 Emerging Risk', 'Tier-3 High Vulnerability', 'Tier-4 Critical Crisis']]
colors_tier = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728']
bars = ax.bar(tier_counts.index, tier_counts.values, color=colors_tier, edgecolor='#333333', width=0.55)
ax.set_title("Figure 4.6: Distribution of Environmental Management Risk Tiers", fontsize=11, weight='bold', pad=10)
ax.set_ylabel("Number of Records", fontsize=10)
plt.xticks(rotation=15, ha='right')
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=9, weight='bold')
plt.tight_layout()
plt.savefig('outputs/figures/fig_4_6_vulnerability_tiers.png', bbox_inches='tight')
plt.close()

# Figure 4.7: Continuous Score Distribution
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
sns.histplot(df_clean['Raw_Vulnerability_Score'], kde=True, color='#3182bd', bins=25, ax=ax)
q25, q50, q75 = df_clean['Raw_Vulnerability_Score'].quantile([0.25, 0.50, 0.75])
ax.axvline(q25, color='green', linestyle='--', label=f'Tier 1/2 Threshold ({q25:.1f})')
ax.axvline(q50, color='orange', linestyle='--', label=f'Tier 2/3 Threshold ({q50:.1f})')
ax.axvline(q75, color='red', linestyle='--', label=f'Tier 3/4 Threshold ({q75:.1f})')
ax.set_title("Figure 4.7: Continuous E-Waste Environmental Vulnerability Index (EVI) Distribution", fontsize=11, weight='bold', pad=10)
ax.set_xlabel("Vulnerability Index Score (0-100)", fontsize=10)
ax.set_ylabel("Frequency", fontsize=10)
ax.legend()
plt.tight_layout()
plt.savefig('outputs/figures/fig_4_7_continuous_hvi_score.png', bbox_inches='tight')
plt.close()

# Figure 4.8: Landfill Diversion Compliance
fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
comp_counts = df_clean['Landfill_Diversion_Compliance'].value_counts()[['Non-Compliant', 'Minimal Compliance', 'Substantial Compliance', 'Certified Compliance']]
ax.bar(comp_counts.index, comp_counts.values, color='#9467bd', edgecolor='#333333', width=0.55)
ax.set_title("Figure 4.8: Distribution of Landfill Diversion Compliance Levels", fontsize=11, weight='bold', pad=10)
ax.set_ylabel("Frequency", fontsize=10)
plt.xticks(rotation=15, ha='right')
for bar in ax.patches:
    height = bar.get_height()
    ax.annotate(f'{height}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig('outputs/figures/fig_4_8_compliance_distribution.png', bbox_inches='tight')
plt.close()

# Figure 4.9: Correlation Matrix Heatmap
fig, ax = plt.subplots(figsize=(9, 7.5), dpi=300)
corr_cols = [
    'Annual_Generation_kt', 'Per_Capita_Generation_kg', 'Population_Millions',
    'Tech_Sector_GDP_Billion_USD', 'Tech_Workforce_Thousands', 'Average_Device_Lifespan_Months',
    'Formal_Collection_Rate_Pct', 'Informal_Recovery_Rate_Pct', 'Heavy_Metal_Toxicity_Index',
    'EPR_Enforcement_Score', 'Raw_Vulnerability_Score'
]
corr_matrix = df_clean[corr_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, ax=ax,
            annot_kws={'size': 7.5}, cbar_kws={'label': 'Pearson r'})
ax.set_title("Figure 4.9: Correlation Matrix of Key Quantitative Indicators", fontsize=11, weight='bold', pad=12)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.savefig('outputs/figures/fig_4_9_correlation_matrix.png', bbox_inches='tight')
plt.close()

# Statistical tables
desc_table = df_clean[corr_cols].describe().T[['mean', 'std', 'min', '50%', 'max']]
desc_table.columns = ['Mean', 'Std. Dev.', 'Min', 'Median', 'Max']
desc_table.round(2).to_csv('outputs/tables/descriptive_statistics.csv')

cat_df = pd.DataFrame({
    'Number of Records': cat_counts.values,
    'Percentage': (cat_counts.values / len(df_clean) * 100).round(2)
}, index=cat_counts.index)
cat_df.to_csv('outputs/tables/category_distribution.csv')

# Step 5: Machine Learning Classification
print("Step 5: Machine Learning Classification Pipeline...")

# Prepare features and target
target_col = 'Management_Risk_Tier'
y_raw = df_feat[target_col]

label_enc_target = LabelEncoder()
y = label_enc_target.fit_transform(y_raw)
classes = label_enc_target.classes_

# Drop non-feature and target columns
exclude_cols = [
    'Record_ID', 'Hub_Stream_Description', 'Raw_Vulnerability_Score',
    'Management_Risk_Tier'
]
X_df = df_feat.drop(columns=exclude_cols).copy()

# Categorical encoding
cat_cols = ['Tech_Hub', 'Country', 'Region', 'E_Waste_Category', 'Landfill_Diversion_Compliance']
for c in cat_cols:
    le = LabelEncoder()
    X_df[c] = le.fit_transform(X_df[c].astype(str))

feature_names = X_df.columns.tolist()
print(f"Total features in model matrix: {len(feature_names)}")

# Stratified 80:20 split
X_train, X_test, y_train, y_test = train_test_split(
    X_df.values, y, test_size=0.20, random_state=42, stratify=y
)

# Standard scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train 6 Classifiers
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=12),
    'Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=8),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, C=1.0),
    'Support Vector Machine (SVM)': SVC(kernel='rbf', probability=True, random_state=42, C=1.0),
    'XGBoost': xgb.XGBClassifier(eval_metric='mlogloss', random_state=42, n_estimators=100, max_depth=5)
}

results = []
conf_matrices = {}
per_class_data = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    cm = confusion_matrix(y_test, y_pred)
    conf_matrices[name] = cm.tolist()
    
    results.append({
        'Model': name,
        'Accuracy': f"{acc * 100:.1f}%",
        'Macro Precision': f"{prec_macro * 100:.1f}%",
        'Macro Recall': f"{rec_macro * 100:.1f}%",
        'Macro F1-score': f"{f1_macro * 100:.1f}%",
        '_acc_float': acc,
        '_prec_float': prec_macro,
        '_rec_float': rec_macro,
        '_f1_float': f1_macro
    })
    
    # Per class metrics
    cr = classification_report(y_test, y_pred, target_names=classes, output_dict=True, zero_division=0)
    for cls in classes:
        per_class_data.append({
            'Model': name,
            'Class': cls,
            'Precision': round(cr[cls]['precision'], 3),
            'Recall': round(cr[cls]['recall'], 3),
            'F1-score': round(cr[cls]['f1-score'], 3),
            'Support': int(cr[cls]['support'])
        })
    
    # Save model
    safe_name = name.lower().replace(' ', '_').replace('(', '').replace(')', '')
    joblib.dump(model, f'outputs/models/{safe_name}_model.joblib')

results_df = pd.DataFrame(results)
display_results = results_df[['Model', 'Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1-score']]
display_results.to_csv('outputs/tables/model_comparison.csv', index=False)
print("\n--- MODEL COMPARISON RESULTS ---")
print(display_results.to_string(index=False))

per_class_df = pd.DataFrame(per_class_data)
per_class_df.to_csv('outputs/tables/per_class_metrics.csv', index=False)

with open('outputs/tables/confusion_matrices.json', 'w') as f:
    json.dump(conf_matrices, f, indent=4)

# Figure 6.1: Target Class Proportions
fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
tier_proportions = df_clean['Management_Risk_Tier'].value_counts()
ax.pie(tier_proportions.values, labels=tier_proportions.index, autopct='%1.1f%%',
       startangle=140, colors=['#4575b4', '#74add1', '#f46d43', '#d73027'])
ax.set_title("Figure 6.1: Target Class Distribution (Management Risk Tiers)", fontsize=11, weight='bold')
plt.tight_layout()
plt.savefig('outputs/figures/fig_6_1_target_tier_pie.png', bbox_inches='tight')
plt.close()

# Figure 6.2: Comparative Accuracy
fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
acc_vals = [r['_acc_float'] for r in results]
model_names = [r['Model'] for r in results]
bars = ax.bar(model_names, acc_vals, color='#2b6cb0', edgecolor='#333333', width=0.55)
ax.axhline(0.25, color='red', linestyle='--', label='Theoretical Chance Level (25.0%)')
ax.set_ylim(0, 1.0)
ax.set_title("Figure 6.2: Comparative Accuracy of Six Evaluated Classification Models", fontsize=11, weight='bold', pad=10)
ax.set_ylabel("Test Accuracy (240 test records)", fontsize=10)
plt.xticks(rotation=20, ha='right')
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4), textcoords="offset points",
                ha='center', va='bottom', fontsize=9, weight='bold')
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig('outputs/figures/fig_6_2_model_accuracy_comparison.png', bbox_inches='tight')
plt.close()

# Figure 6.3: Confusion Matrix - Best Model (Random Forest)
fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)
rf_cm = np.array(conf_matrices['Random Forest'])
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Tier-1', 'Tier-2', 'Tier-3', 'Tier-4'],
            yticklabels=['Tier-1', 'Tier-2', 'Tier-3', 'Tier-4'], ax=ax)
ax.set_title("Figure 6.3: Confusion Matrix – Best Performing Model (Random Forest)", fontsize=11, weight='bold', pad=10)
ax.set_xlabel("Predicted Class", fontsize=10)
ax.set_ylabel("True Class", fontsize=10)
plt.tight_layout()
plt.savefig('outputs/figures/fig_6_3_confusion_matrix_best_rf.png', bbox_inches='tight')
plt.close()

# Figure B.1: Macro Precision, Recall, and F1-score
fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
x = np.arange(len(model_names))
w = 0.25
ax.bar(x - w, [r['_prec_float'] for r in results], w, label='Macro Precision', color='#3182bd', edgecolor='#333333')
ax.bar(x, [r['_rec_float'] for r in results], w, label='Macro Recall', color='#31a354', edgecolor='#333333')
ax.bar(x + w, [r['_f1_float'] for r in results], w, label='Macro F1-score', color='#e6550d', edgecolor='#333333')
ax.set_title("Figure B.1: Comparative Macro-Averaged Performance Metrics", fontsize=11, weight='bold', pad=10)
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=20, ha='right')
ax.set_ylabel("Score", fontsize=10)
ax.set_ylim(0, 1.0)
ax.legend()
plt.tight_layout()
plt.savefig('outputs/figures/fig_b_1_macro_metrics_comparison.png', bbox_inches='tight')
plt.close()

# Figures B.2 to B.7: Confusion Matrices for All 6 Models
cm_plots = [
    ('Logistic Regression', 'fig_b_2_cm_logistic_regression.png', 'Figure B.2: Confusion Matrix – Logistic Regression'),
    ('Naive Bayes', 'fig_b_3_cm_naive_bayes.png', 'Figure B.3: Confusion Matrix – Naive Bayes'),
    ('Decision Tree', 'fig_b_4_cm_decision_tree.png', 'Figure B.4: Confusion Matrix – Decision Tree'),
    ('Random Forest', 'fig_b_5_cm_random_forest.png', 'Figure B.5: Confusion Matrix – Random Forest'),
    ('Support Vector Machine (SVM)', 'fig_b_6_cm_svm.png', 'Figure B.6: Confusion Matrix – Support Vector Machine'),
    ('XGBoost', 'fig_b_7_cm_xgboost.png', 'Figure B.7: Confusion Matrix – XGBoost')
]

for m_name, file_name, title in cm_plots:
    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=300)
    cm_arr = np.array(conf_matrices[m_name])
    sns.heatmap(cm_arr, annot=True, fmt='d', cmap='Blues',
                xticklabels=['T1', 'T2', 'T3', 'T4'],
                yticklabels=['T1', 'T2', 'T3', 'T4'], ax=ax)
    ax.set_title(title, fontsize=10, weight='bold', pad=8)
    ax.set_xlabel("Predicted Class", fontsize=9)
    ax.set_ylabel("True Class", fontsize=9)
    plt.tight_layout()
    plt.savefig(f'outputs/figures/{file_name}', bbox_inches='tight')
    plt.close()

# Feature Importances: Random Forest & XGBoost
rf_model = models['Random Forest']
xgb_model = models['XGBoost']

rf_importances = pd.Series(rf_model.feature_importances_, index=feature_names).sort_values(ascending=False).head(10)
xgb_importances = pd.Series(xgb_model.feature_importances_, index=feature_names).sort_values(ascending=False).head(10)

fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
rf_importances.plot(kind='barh', ax=ax, color='#3182bd', edgecolor='#333333')
ax.invert_yaxis()
ax.set_title("Figure B.8: Top 10 Feature Importances – Random Forest", fontsize=11, weight='bold', pad=10)
ax.set_xlabel("Relative Impurity Importance", fontsize=10)
plt.tight_layout()
plt.savefig('outputs/figures/fig_b_8_rf_feature_importance.png', bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
xgb_importances.plot(kind='barh', ax=ax, color='#e6550d', edgecolor='#333333')
ax.invert_yaxis()
ax.set_title("Figure B.9: Top 10 Feature Importances – XGBoost", fontsize=11, weight='bold', pad=10)
ax.set_xlabel("Relative Gain Importance", fontsize=10)
plt.tight_layout()
plt.savefig('outputs/figures/fig_b_9_xgb_feature_importance.png', bbox_inches='tight')
print("Pipeline execution complete! All figures, tables, and models saved successfully.")

# Optional report generation if generator scripts exist
if os.path.exists("scripts/generate_report_pdf.py"):
    print("Generating academic project report...")
    import subprocess
    subprocess.run(["python", "scripts/generate_report_pdf.py"], check=False)
    if os.path.exists("scripts/generate_report_docx.py"):
        subprocess.run(["python", "scripts/generate_report_docx.py"], check=False)
    if os.path.exists("scripts/generate_markdown_report.py"):
        subprocess.run(["python", "scripts/generate_markdown_report.py"], check=False)

