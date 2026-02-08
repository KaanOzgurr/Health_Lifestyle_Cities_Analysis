import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Style settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Load dataset - CORRECT FILENAME
df = pd.read_csv('healthy_lifestyle_cities_report_2021.csv')

print("=" * 80)
print("HEALTHY LIFESTYLE CITIES ANALYSIS - 2021")
print("=" * 80)
print("\n1. DATASET OVERVIEW")
print("-" * 80)
print(f"Total cities: {len(df)}")
print(f"Total features: {len(df.columns)}")
print("\nFirst 5 rows:")
print(df.head())

print("\n\nDataset columns:")
print(df.columns.tolist())

print("\n\nData types:")
print(df.dtypes)

print("\n\nMissing data analysis:")
print(df.isnull().sum())

print("\n\nBasic statistics:")
print(df.describe())

# Data cleaning - convert object type numeric columns
# Replace '-' with NaN and remove special characters
df['Sunshine hours(City)'] = df['Sunshine hours(City)'].str.replace(',', '').str.replace('-', '').replace('', np.nan)
df['Sunshine hours(City)'] = pd.to_numeric(df['Sunshine hours(City)'], errors='coerce')

df['Obesity levels(Country)'] = df['Obesity levels(Country)'].str.replace('%', '').str.replace('-', '').replace('', np.nan)
df['Obesity levels(Country)'] = pd.to_numeric(df['Obesity levels(Country)'], errors='coerce')

df['Pollution(Index score) (City)'] = df['Pollution(Index score) (City)'].str.replace(',', '').str.replace('-', '').replace('', np.nan)
df['Pollution(Index score) (City)'] = pd.to_numeric(df['Pollution(Index score) (City)'], errors='coerce')

df['Annual avg. hours worked'] = df['Annual avg. hours worked'].str.replace(',', '').str.replace('-', '').replace('', np.nan)
df['Annual avg. hours worked'] = pd.to_numeric(df['Annual avg. hours worked'], errors='coerce')

df['Cost of a monthly gym membership(City)'] = df['Cost of a monthly gym membership(City)'].str.replace('£', '').str.replace('-', '').replace('', np.nan)
df['Cost of a monthly gym membership(City)'] = pd.to_numeric(df['Cost of a monthly gym membership(City)'], errors='coerce')

df['Cost of a bottle of water(City)'] = df['Cost of a bottle of water(City)'].str.replace('£', '').str.replace('-', '').replace('', np.nan)
df['Cost of a bottle of water(City)'] = pd.to_numeric(df['Cost of a bottle of water(City)'], errors='coerce')

print("\n\n✓ Data cleaning completed!")

# ============================================================================
# CHART 1: Top 15 Happiest Cities
# ============================================================================
plt.figure(figsize=(14, 8))
top_15_happy = df.nlargest(15, 'Happiness levels(Country)')
colors = plt.cm.viridis(np.linspace(0, 1, 15))
bars = plt.barh(range(15), top_15_happy['Happiness levels(Country)'], color=colors)
plt.yticks(range(15), top_15_happy['City'])
plt.xlabel('Happiness Level', fontsize=12, fontweight='bold')
plt.title('Top 15 Happiest Cities', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()

for i, (idx, row) in enumerate(top_15_happy.iterrows()):
    plt.text(row['Happiness levels(Country)'], i, f" {row['Happiness levels(Country)']:.2f}", 
             va='center', fontsize=9)

plt.tight_layout()
plt.savefig('1_top_cities_happiness.png', dpi=300, bbox_inches='tight')
print("\n✓ Chart 1 saved: 1_top_cities_happiness.png")

# ============================================================================
# CHART 2: Sunshine Hours vs Happiness
# ============================================================================
plt.figure(figsize=(12, 8))
plt.scatter(df['Sunshine hours(City)'], df['Happiness levels(Country)'], 
            alpha=0.6, s=100, c=df['Cost of a monthly gym membership(City)'], 
            cmap='YlOrRd', edgecolors='black', linewidth=0.5)
plt.colorbar(label='Gym Membership Cost (£)')
plt.xlabel('Sunshine Hours (Annual)', fontsize=12, fontweight='bold')
plt.ylabel('Happiness Level', fontsize=12, fontweight='bold')
plt.title('Sunshine Hours vs Happiness Level\n(Color: Gym Membership Cost)', 
          fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

corr = df['Sunshine hours(City)'].corr(df['Happiness levels(Country)'])
plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
         transform=plt.gca().transAxes, fontsize=11, 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
         verticalalignment='top')

plt.tight_layout()
plt.savefig('2_sunshine_vs_happiness.png', dpi=300, bbox_inches='tight')
print("✓ Chart 2 saved: 2_sunshine_vs_happiness.png")

# ============================================================================
# CHART 3: Gym Cost Comparison
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 8))
top_20_cities = df.nlargest(20, 'Happiness levels(Country)')

x = np.arange(len(top_20_cities))
bars = ax.bar(x, top_20_cities['Cost of a monthly gym membership(City)'], 
              color='#3498db', alpha=0.8)

ax.set_xlabel('Cities', fontsize=12, fontweight='bold')
ax.set_ylabel('Monthly Gym Membership Cost (£)', fontsize=12, fontweight='bold')
ax.set_title('Gym Membership Costs in Top 20 Happiest Cities', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(top_20_cities['City'], rotation=45, ha='right')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('3_gym_cost_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Chart 3 saved: 3_gym_cost_comparison.png")

# ============================================================================
# CHART 4: Correlation Matrix
# ============================================================================
plt.figure(figsize=(14, 12))

numeric_cols = ['Rank', 'Sunshine hours(City)', 'Cost of a bottle of water(City)',
                'Obesity levels(Country)', 'Life expectancy(years) (Country)',
                'Pollution(Index score) (City)', 'Annual avg. hours worked',
                'Happiness levels(Country)', 'Outdoor activities(City)',
                'Number of take out places(City)', 'Cost of a monthly gym membership(City)']

correlation_matrix = df[numeric_cols].corr()

mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', 
            cmap='coolwarm', center=0, square=True, linewidths=1,
            cbar_kws={"shrink": 0.8})

plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout()
plt.savefig('4_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Chart 4 saved: 4_correlation_heatmap.png")

# ============================================================================
# CHART 5: Activities vs Obesity
# ============================================================================
plt.figure(figsize=(12, 8))

plt.scatter(df['Outdoor activities(City)'], df['Obesity levels(Country)'], 
            alpha=0.6, s=100, c=df['Happiness levels(Country)'], 
            cmap='RdYlGn', edgecolors='black', linewidth=0.5)
plt.colorbar(label='Happiness Level')
plt.xlabel('Outdoor Activities (Count)', fontsize=12, fontweight='bold')
plt.ylabel('Obesity Rate (%)', fontsize=12, fontweight='bold')
plt.title('Outdoor Activities vs Obesity Rate\n(Color: Happiness Level)', 
          fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

corr = df['Outdoor activities(City)'].corr(df['Obesity levels(Country)'])
plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
         transform=plt.gca().transAxes, fontsize=11, 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
         verticalalignment='top')

plt.tight_layout()
plt.savefig('5_activities_vs_obesity.png', dpi=300, bbox_inches='tight')
print("✓ Chart 5 saved: 5_activities_vs_obesity.png")

# ============================================================================
# CHART 6: Air Quality Distribution
# ============================================================================
plt.figure(figsize=(12, 8))

df['Air_Quality_Category'] = pd.cut(df['Pollution(Index score) (City)'], 
                                     bins=[0, 25, 50, 75, 100], 
                                     labels=['Excellent', 'Good', 'Moderate', 'Poor'])

category_counts = df['Air_Quality_Category'].value_counts()
colors_pie = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']

plt.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%',
        startangle=90, colors=colors_pie, explode=(0.1, 0, 0, 0.1),
        textprops={'fontsize': 12, 'fontweight': 'bold'})
plt.title('Air Quality Distribution Across Cities', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('6_air_quality_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Chart 6 saved: 6_air_quality_distribution.png")

# ============================================================================
# CHART 7: Life Expectancy Analysis
# ============================================================================
fig, axes = plt.subplots(2, 1, figsize=(14, 12))

top_15_life = df.nlargest(15, 'Life expectancy(years) (Country)')
colors = plt.cm.Greens(np.linspace(0.4, 1, 15))

axes[0].barh(range(15), top_15_life['Life expectancy(years) (Country)'], color=colors)
axes[0].set_yticks(range(15))
axes[0].set_yticklabels(top_15_life['City'])
axes[0].set_xlabel('Life Expectancy (Years)', fontsize=12, fontweight='bold')
axes[0].set_title('Highest Life Expectancy - Top 15 Cities', fontsize=13, fontweight='bold')
axes[0].invert_yaxis()
axes[0].grid(True, alpha=0.3, axis='x')

for i, (idx, row) in enumerate(top_15_life.iterrows()):
    axes[0].text(row['Life expectancy(years) (Country)'], i, 
                 f" {row['Life expectancy(years) (Country)']:.1f}", 
                 va='center', fontsize=9)

bottom_15_life = df.nsmallest(15, 'Life expectancy(years) (Country)')
colors = plt.cm.Reds(np.linspace(0.4, 1, 15))

axes[1].barh(range(15), bottom_15_life['Life expectancy(years) (Country)'], color=colors)
axes[1].set_yticks(range(15))
axes[1].set_yticklabels(bottom_15_life['City'])
axes[1].set_xlabel('Life Expectancy (Years)', fontsize=12, fontweight='bold')
axes[1].set_title('Lowest Life Expectancy - Bottom 15 Cities', fontsize=13, fontweight='bold')
axes[1].invert_yaxis()
axes[1].grid(True, alpha=0.3, axis='x')

for i, (idx, row) in enumerate(bottom_15_life.iterrows()):
    axes[1].text(row['Life expectancy(years) (Country)'], i, 
                 f" {row['Life expectancy(years) (Country)']:.1f}", 
                 va='center', fontsize=9)

plt.tight_layout()
plt.savefig('7_life_expectancy_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Chart 7 saved: 7_life_expectancy_analysis.png")

# ============================================================================
# CHART 8: PCA Analysis
# ============================================================================
plt.figure(figsize=(12, 8))

features_for_pca = ['Happiness levels(Country)', 'Sunshine hours(City)', 
                    'Life expectancy(years) (Country)', 'Obesity levels(Country)',
                    'Pollution(Index score) (City)', 'Outdoor activities(City)']

df_pca = df[features_for_pca + ['City']].dropna()

scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_pca[features_for_pca])

pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_features)

plt.scatter(pca_components[:, 0], pca_components[:, 1], 
            alpha=0.6, s=100, c=df_pca['Happiness levels(Country)'], 
            cmap='viridis', edgecolors='black', linewidth=0.5)
plt.colorbar(label='Happiness Level')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', 
           fontsize=12, fontweight='bold')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', 
           fontsize=12, fontweight='bold')
plt.title('PCA Analysis - Cities Distribution by Principal Components', 
          fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

for i in range(len(pca_components)):
    if abs(pca_components[i, 0]) > 2.5 or abs(pca_components[i, 1]) > 2.5:
        plt.annotate(df_pca.iloc[i]['City'], 
                    (pca_components[i, 0], pca_components[i, 1]),
                    fontsize=8, alpha=0.7)

plt.tight_layout()
plt.savefig('8_pca_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Chart 8 saved: 8_pca_analysis.png")

# ============================================================================
# CHART 9: Radar Chart - Top 5 Cities
# ============================================================================
from math import pi

fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))

top_5 = df.nlargest(5, 'Happiness levels(Country)')

categories = ['Happiness', 'Sunshine\nHours', 'Life\nExpectancy', 
              'Air\nQuality', 'Activities']
N = len(categories)

colors_radar = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
for idx, (i, city_row) in enumerate(top_5.iterrows()):
    values = [
        city_row['Happiness levels(Country)'] * 15,
        city_row['Sunshine hours(City)'] / 30,
        city_row['Life expectancy(years) (Country)'] * 1.2,
        100 - city_row['Pollution(Index score) (City)'],
        city_row['Outdoor activities(City)'] / 6
    ]
    
    values += values[:1]
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    ax.plot(angles, values, 'o-', linewidth=2, label=city_row['City'], 
            color=colors_radar[idx])
    ax.fill(angles, values, alpha=0.15, color=colors_radar[idx])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11)
ax.set_ylim(0, 100)
ax.set_title('Top 5 Cities - Multidimensional Comparison\n(Normalized Values)', 
             fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
ax.grid(True)

plt.tight_layout()
plt.savefig('9_radar_top5_cities.png', dpi=300, bbox_inches='tight')
print("✓ Chart 9 saved: 9_radar_top5_cities.png")

# ============================================================================
# CHART 10: Comprehensive Dashboard
# ============================================================================
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

ax1 = fig.add_subplot(gs[0, :])
ax1.hist(df['Happiness levels(Country)'], bins=30, color='skyblue', 
         edgecolor='black', alpha=0.7)
ax1.set_xlabel('Happiness Level', fontweight='bold')
ax1.set_ylabel('Number of Cities', fontweight='bold')
ax1.set_title('Happiness Level Distribution', fontweight='bold', fontsize=12)
ax1.axvline(df['Happiness levels(Country)'].mean(), color='red', 
            linestyle='dashed', linewidth=2, 
            label=f'Average: {df["Happiness levels(Country)"].mean():.2f}')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(gs[1, 0])
ax2.scatter(df['Cost of a monthly gym membership(City)'], 
            df['Happiness levels(Country)'], alpha=0.5, s=30)
ax2.set_xlabel('Gym Cost (£)', fontweight='bold', fontsize=9)
ax2.set_ylabel('Happiness', fontweight='bold', fontsize=9)
ax2.set_title('Cost vs Happiness', fontweight='bold', fontsize=10)
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(gs[1, 1])
ax3.boxplot([df['Sunshine hours(City)'].dropna()], labels=['Sunshine Hours'])
ax3.set_ylabel('Hours', fontweight='bold', fontsize=9)
ax3.set_title('Sunshine Hours Distribution', fontweight='bold', fontsize=10)
ax3.grid(True, alpha=0.3, axis='y')

ax4 = fig.add_subplot(gs[1, 2])
ax4.boxplot([df['Life expectancy(years) (Country)'].dropna()], 
            labels=['Life Expectancy'])
ax4.set_ylabel('Years', fontweight='bold', fontsize=9)
ax4.set_title('Life Expectancy Distribution', fontweight='bold', fontsize=10)
ax4.grid(True, alpha=0.3, axis='y')

ax5 = fig.add_subplot(gs[2, :2])
top_10 = df.nlargest(10, 'Happiness levels(Country)')
colors_bar = plt.cm.plasma(np.linspace(0, 1, 10))
bars = ax5.barh(range(10), top_10['Happiness levels(Country)'], color=colors_bar)
ax5.set_yticks(range(10))
ax5.set_yticklabels(top_10['City'], fontsize=9)
ax5.set_xlabel('Happiness Level', fontweight='bold', fontsize=9)
ax5.set_title('Top 10 Happiest Cities', fontweight='bold', fontsize=10)
ax5.invert_yaxis()
ax5.grid(True, alpha=0.3, axis='x')

ax6 = fig.add_subplot(gs[2, 2])
ax6.axis('off')
stats_data = [
    ['Total Cities:', f'{len(df)}'],
    ['Avg. Happiness:', f'{df["Happiness levels(Country)"].mean():.2f}'],
    ['Avg. Life Exp.:', f'{df["Life expectancy(years) (Country)"].mean():.1f}'],
    ['Avg. Obesity:', f'{df["Obesity levels(Country)"].mean():.1f}%'],
    ['Avg. Sunshine:', f'{df["Sunshine hours(City)"].mean():.0f}'],
]
table = ax6.table(cellText=stats_data, cellLoc='left', loc='center',
                  colWidths=[0.6, 0.4])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)
ax6.set_title('General Statistics', fontweight='bold', fontsize=10, pad=20)

fig.suptitle('HEALTHY LIFESTYLE CITIES - COMPREHENSIVE ANALYSIS DASHBOARD', 
             fontsize=16, fontweight='bold', y=0.98)

plt.savefig('10_comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
print("✓ Chart 10 saved: 10_comprehensive_dashboard.png")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETED!")
print("=" * 80)
print("\nAll charts saved to project folder.")
print("\nGenerated charts:")
print("  1. Top 15 Happiest Cities")
print("  2. Sunshine Hours vs Happiness")
print("  3. Gym Membership Cost Comparison")
print("  4. Correlation Matrix")
print("  5. Activities vs Obesity")
print("  6. Air Quality Distribution")
print("  7. Life Expectancy Analysis")
print("  8. PCA Analysis")
print("  9. Top 5 Cities Radar Chart")
print(" 10. Comprehensive Dashboard")
print("=" * 80)

    







    













    







                                    






















    













