import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create folder if it doesn't exist
os.makedirs("outputs/metrics", exist_ok=True)

# Load data
with open('outputs/metrics/batch_results.json', 'r', encoding='utf-8') as f:
    results = json.load(f)

records = []
all_classes = []
for k, v in results.items():
    records.append({
        'num_objects': v['num_objects'],
        'terrain': v['scene']['terrain'],
        'activity': v['scene']['activity'],
        'priority_score': v['priority']['score'],
        'priority_tier': v['priority']['tier'],
    })
    for d in v.get('detections', []):
        all_classes.append(d['label'])
        
df = pd.DataFrame(records)

# 1. Number of Objects
plt.figure(figsize=(8,5))
sns.countplot(data=df, x='num_objects', palette='viridis')
plt.title('Distribution of Object Detections (People/Animals)')
plt.xlabel('Number of Objects in Image')
plt.ylabel('Count of Images')
plt.savefig('outputs/metrics/1_object_distributions.png', bbox_inches='tight')
plt.close()

# 2. Priority Hist
plt.figure(figsize=(8,5))
sns.histplot(df['priority_score'], bins=20, kde=True, color='red')
plt.title('Histogram of Priority Scores (0 - 100)')
plt.xlabel('Priority Score')
plt.savefig('outputs/metrics/2_priority_scores.png', bbox_inches='tight')
plt.close()

# 3. Priority Tiers
plt.figure(figsize=(8,5))
sns.countplot(data=df, x='priority_tier', order=['P1', 'P2', 'P3', 'P4'], palette='OrRd_r')
plt.title('Images per Priority Tier')
plt.savefig('outputs/metrics/3_priority_tiers.png', bbox_inches='tight')
plt.close()

# 4. Terrain Pie
plt.figure(figsize=(8,8))
terrain_counts = df['terrain'].value_counts()
plt.pie(terrain_counts, labels=terrain_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
plt.title('Terrain Classification Across Dataset')
plt.savefig('outputs/metrics/4_terrain_pie.png', bbox_inches='tight')
plt.close()

# 5. Activity Count
plt.figure(figsize=(10,6))
sns.countplot(data=df, y='activity', order=df['activity'].value_counts().index, palette='magma')
plt.title('Dominant Activities Detected')
plt.savefig('outputs/metrics/5_activities.png', bbox_inches='tight')
plt.close()

# 6. Detections Count
if all_classes:
    plt.figure(figsize=(8,5))
    sns.countplot(y=all_classes, order=pd.Series(all_classes).value_counts().index, palette='cubehelix')
    plt.title('Instances of Different Objects Detected by YOLO')
    plt.xlabel('Total Detected Instances')
    plt.savefig('outputs/metrics/6_detected_classes.png', bbox_inches='tight')
    plt.close()

print("Graphs successfully exported!")
