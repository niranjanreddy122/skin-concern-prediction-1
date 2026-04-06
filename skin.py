# ==============================
# STEP 1: IMPORT LIBRARIES
# ==============================
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# ==============================
# STEP 2: LOAD DATASET
# ==============================
file_name = input("Enter file path (example: C:/Users/YourName/Downloads/file.xlsx): ")

df = pd.read_excel(file_name)

print("\n✅ Dataset Loaded Successfully!\n")
print(df.head())

# ==============================
# STEP 3: SELECT FEATURES & TARGET
# ==============================
print("\nColumns:\n", list(df.columns))

target = input("\nEnter target column EXACTLY as shown: ").strip()

features = input("Enter feature columns (comma separated): ").split(',')
features = [f.strip() for f in features]

# Validation
for col in features + [target]:
    if col not in df.columns:
        print(f"❌ Error: {col} not found in dataset")
        exit()

print("\n✅ Features:", features)
print("✅ Target:", target)

# ==============================
# STEP 4: PREPROCESSING
# ==============================
X = df[features].copy()
y = df[target].copy()

le_dict = {}

# Encode features
for col in X.columns:
    le = LabelEncoder()
    X.loc[:, col] = le.fit_transform(X[col])
    le_dict[col] = le

# Encode target
le_target = LabelEncoder()
y = le_target.fit_transform(y)

# ==============================
# STEP 5: SPLIT + TRAIN MODEL
# ==============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

print("\n✅ Model Trained Successfully!")

accuracy = model.score(X_test, y_test)
print("📊 Accuracy:", accuracy)

# ==============================
# STEP 6: ACCURACY GRAPH
# ==============================
plt.figure()
plt.bar(['Accuracy'], [accuracy])
plt.title("Model Accuracy")
plt.xlabel("Model")
plt.ylabel("Score")
plt.show()

# ==============================
# STEP 7: FEATURE IMPORTANCE GRAPH
# ==============================
importance = model.feature_importances_

plt.figure(figsize=(6,4))
plt.bar(features, importance)
plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.show()

# ==============================
# STEP 8: USER INPUT + PREDICTION
# ==============================
print("\n🔮 Enter values for prediction:\n")

user_data = {}
encoded_input = []

for col in features:
    print(f"\nOptions for {col}:")
    print(list(le_dict[col].classes_))
    
    val = input(f"Enter {col}: ")
    user_data[col] = val
    
    val_encoded = le_dict[col].transform([val])[0]
    encoded_input.append(val_encoded)

# Predict
user_df = pd.DataFrame([encoded_input], columns=features)
prediction = model.predict(user_df)
predicted_concern = le_target.inverse_transform(prediction)[0]

print("\n🎯 Predicted Concern:", predicted_concern)

# ==============================
# STEP 9: SHOW INGREDIENTS
# ==============================
result = df.copy()

for col in features:
    result = result[result[col] == user_data[col]]

result = result[result[target] == predicted_concern]

if not result.empty:
    print("\n🧪 Ingredients:", result.iloc[0]["Ingredients"])
    print("📊 Concentrations:", result.iloc[0]["Concentrations"])
    print("✨ Effects:", result.iloc[0]["Effects"])
else:
    print("\n⚠️ No exact match found, showing predicted concern only.")