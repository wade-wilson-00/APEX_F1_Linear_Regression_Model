# 🏎️ APEX — F1 Lap Time Prediction using Linear Regression

APEX is a learning-focused Machine Learning project where **Linear Regression is implemented from scratch using Gradient Descent** to predict Formula 1 lap times.  
The project emphasizes **understanding fundamentals** rather than maximizing accuracy.

The custom model is later **validated against sklearn’s LinearRegression** to ensure correctness.

---

## 📌 Project Objective

The main objectives of this project are:

- Understand Linear Regression mathematically
- Implement Gradient Descent from scratch
- Observe convergence and learning rate behavior
- Learn the importance of feature scaling
- Analyze underfitting using real-world data
- Validate a custom model using sklearn

This project focuses on **learning how models behave**, not on building a production-ready predictor.

---

## 📊 Dataset

- Formula 1 lap time dataset
- Each row represents a single lap in a race

### Key Columns Used
- `raceId` — Race identifier
- `driverId` — Driver identifier
- `lap` — Lap number
- `milliseconds` — Lap time in milliseconds

### Data Filtering
To maintain interpretability:
- A **single race** was selected
- A **single driver** was selected
- Extreme outliers (e.g., pit stops) were removed

---

## 🧠 Problem Formulation

This is a **single-variable linear regression** problem:

Lap Time = w × Lap Number + b

- **Feature (X):** Lap number  
- **Target (Y):** Lap time in milliseconds  

The use of a single feature is intentional to study model limitations and underfitting.

---

## ⚖️ Feature Scaling

Gradient Descent is sensitive to feature scale.

- Lap numbers are small (1–70)
- Lap times are large (~80,000 ms)

To stabilize training, the feature was standardized:

X_scaled = (X − mean) / standard deviation

All training and predictions were performed using `X_scaled`.  
Raw lap numbers were used only for visualization.

---

## 🧮 Cost Function

The model uses **Mean Squared Error (MSE)**:

J(w, b) = (1 / 2m) × Σ (ŷ − y)²

MSE is:
- Differentiable
- Sensitive to large errors
- Suitable for Gradient Descent

---

## 🔁 Gradient Descent Implementation

Linear Regression was optimized using Gradient Descent implemented from scratch.

At each iteration:
1. Predictions are generated
2. Gradients are computed
3. Parameters are updated

Update rules:

w := w − α × ∂J/∂w  
b := b − α × ∂J/∂b  

Where α is the learning rate.

---

## 📉 Model Convergence

- Cost decreases rapidly in early iterations
- Gradually flattens as convergence is reached
- No oscillation or divergence observed

This confirms correct gradient computation and stable learning.

---

## 🔍 Predictions & Model Behavior

- Predictions cluster near the average lap time
- The model captures the overall trend
- Individual lap fluctuations are not captured

This behavior indicates **underfitting**, which is expected given the single feature.

---

## 🚨 Underfitting Explained

Lap time depends on many factors:
- Tyre degradation
- Fuel load
- Traffic
- DRS
- Driver strategy

Using only lap number is insufficient to model this complexity.

This project intentionally demonstrates:
**Underfitting due to limited features, not algorithm failure.**

---

## 🔍 sklearn Validation

- sklearn’s `LinearRegression` was trained on the same scaled features
- Predictions from both models were compared

### Result
- Custom Gradient Descent predictions closely match sklearn
- Confirms correctness of the custom implementation

This highlights an important ML principle:
**Models must be trained and evaluated on identical feature representations.**

---

## 📊 Visualizations

The project includes:
- Cost vs Iterations plot
- Actual vs Predicted lap times
- Custom model vs sklearn predictions

These plots help visualize learning behavior and model limitations.

---

## 🧪 Tech Stack

- Python
- NumPy
- Pandas
- Matplotlib
- scikit-learn (validation only)
- Jupyter Notebook

---

## 🧠 Key Learnings

- Linear Regression models average trends
- Feature scaling is critical for Gradient Descent
- Underfitting is a feature problem, not an algorithm problem
- Validation builds confidence in custom implementations
- Fundamentals matter more than libraries

---

## 🏁 Final Note

This project was built with a **learn-by-building** approach  
and serves as a foundation for deeper exploration into Machine Learning.

---

## ✍️ Author

**Aman**
