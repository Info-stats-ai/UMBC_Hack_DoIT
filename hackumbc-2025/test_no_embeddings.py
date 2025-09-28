#!/usr/bin/env python3
"""
Quick test script to compare accuracy with and without embeddings
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

print("🧪 TESTING: With vs Without Embeddings")
print("=" * 50)

# Load the comprehensive datasets
train_df = pd.read_csv("ml/data/train_processed_comprehensive.csv")
test_df = pd.read_csv("ml/data/test_processed_comprehensive.csv")

print(f"📊 Original dataset shape: {train_df.shape}")

# Convert GPA to grade categories
def gpa_to_grade_category(gpa):
    if gpa >= 3.7: return 'A'
    elif gpa >= 3.0: return 'B'
    elif gpa >= 2.0: return 'C'
    elif gpa >= 1.0: return 'D'
    else: return 'F'

train_df['grade_category'] = train_df['gpa'].apply(gpa_to_grade_category)
test_df['grade_category'] = test_df['gpa'].apply(gpa_to_grade_category)

# Prepare features
id_cols = ["student_id", "course_id"]
target_col = "grade_category"
all_feature_cols = [c for c in train_df.columns if c not in id_cols + [target_col, 'gpa']]

# Filter out non-numerical features
numerical_features = train_df[all_feature_cols].select_dtypes(include=[np.number]).columns.tolist()
print(f"📊 All available features: {len(all_feature_cols)}")
print(f"📊 Numerical features: {len(numerical_features)}")

# Identify embedding features
embedding_features = [col for col in numerical_features if 'emb_' in col]
academic_features = [col for col in numerical_features if 'emb_' not in col]

print(f"📊 Embedding features: {len(embedding_features)}")
print(f"📊 Academic features: {len(academic_features)}")

# Test 1: WITH embeddings (original approach)
print(f"\n🔬 TEST 1: WITH EMBEDDINGS ({len(numerical_features)} features)")
X_train_all = train_df[numerical_features]
y_train_all = train_df[target_col]
X_test_all = test_df[numerical_features]
y_test_all = test_df[target_col]

# Encode labels
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train_all)
y_test_encoded = le.transform(y_test_all)

# Train models
rf_all = RandomForestClassifier(n_estimators=100, random_state=42)
rf_all.fit(X_train_all, y_train_encoded)
y_pred_all = rf_all.predict(X_test_all)

accuracy_all = accuracy_score(y_test_encoded, y_pred_all)
f1_all = f1_score(y_test_encoded, y_pred_all, average='weighted')

print(f"   Random Forest Accuracy: {accuracy_all:.4f}")
print(f"   Random Forest F1: {f1_all:.4f}")

# Test 2: WITHOUT embeddings (academic features only)
print(f"\n🔬 TEST 2: WITHOUT EMBEDDINGS ({len(academic_features)} features)")
X_train_academic = train_df[academic_features]
y_train_academic = train_df[target_col]
X_test_academic = test_df[academic_features]
y_test_academic = test_df[target_col]

# Encode labels
y_train_encoded_academic = le.fit_transform(y_train_academic)
y_test_encoded_academic = le.transform(y_test_academic)

# Train models
rf_academic = RandomForestClassifier(n_estimators=100, random_state=42)
rf_academic.fit(X_train_academic, y_train_encoded_academic)
y_pred_academic = rf_academic.predict(X_test_academic)

accuracy_academic = accuracy_score(y_test_encoded_academic, y_pred_academic)
f1_academic = f1_score(y_test_encoded_academic, y_pred_academic, average='weighted')

print(f"   Random Forest Accuracy: {accuracy_academic:.4f}")
print(f"   Random Forest F1: {f1_academic:.4f}")

# Compare results
print(f"\n📊 COMPARISON:")
print(f"   With Embeddings:    {accuracy_all:.4f} accuracy, {f1_all:.4f} F1")
print(f"   Without Embeddings: {accuracy_academic:.4f} accuracy, {f1_academic:.4f} F1")

improvement = ((accuracy_academic - accuracy_all) / accuracy_all) * 100
print(f"   Improvement: {improvement:+.2f}%")

if accuracy_academic > accuracy_all:
    print(f"   ✅ REMOVING EMBEDDINGS IMPROVES ACCURACY!")
else:
    print(f"   ⚠️ Embeddings might be helpful, but let's try other approaches")

print(f"\n📊 Feature Analysis:")
print(f"   Academic features used: {academic_features}")
print(f"   Feature-to-sample ratio (no embeddings): {len(academic_features)}/{len(X_train_academic)} = 1:{len(X_train_academic)//len(academic_features)}")
