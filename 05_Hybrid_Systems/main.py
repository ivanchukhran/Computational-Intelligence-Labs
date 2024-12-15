import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import torch.nn as nn
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt


def prepare_data(filepath):
    try:
        data = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error reading file: {e}")
        return None, None, None

    categorical_cols = [
        "sex",
        "chest_pain_type",
        "fasting_blood_sugar",
        "rest_ecg",
        "exercise_induced_angina",
        "slope",
        "vessels_colored_by_flourosopy",
        "thalassemia",
    ]

    for col in categorical_cols:
        if data[col].dtype == "object":
            le = LabelEncoder()
            data[col] = data[col].fillna(data[col].mode()[0])
            data[col] = le.fit_transform(data[col].astype(str))

    numerical_cols = [
        "age",
        "resting_blood_pressure",
        "cholestoral",
        "max_heart_rate",
        "oldpeak",
    ]

    for col in numerical_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
        data[col] = data[col].fillna(data[col].median())

    scaler = StandardScaler()
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

    data["target"] = pd.to_numeric(data["target"], errors="coerce")
    data["target"] = data["target"].fillna(0)

    return data, scaler, numerical_cols


class HeartDiseaseNN(nn.Module):
    def __init__(self, input_size):
        super(HeartDiseaseNN, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout(self.relu(self.layer1(x)))
        x = self.dropout(self.relu(self.layer2(x)))
        x = self.sigmoid(self.layer3(x))
        return x


def train_nn_model(X_train, y_train, input_size, epochs=100):
    model = HeartDiseaseNN(input_size)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    X_train_tensor = torch.FloatTensor(X_train.values)
    y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1)

    losses = []

    for epoch in range(epochs):
        model.train()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    return model, losses


def create_fuzzy_system():

    age = ctrl.Antecedent(np.arange(20, 81, 1), "age")
    blood_pressure = ctrl.Antecedent(np.arange(90, 201, 1), "blood_pressure")
    cholesterol = ctrl.Antecedent(np.arange(100, 501, 1), "cholesterol")
    heart_rate = ctrl.Antecedent(np.arange(60, 201, 1), "heart_rate")
    nn_prediction = ctrl.Antecedent(np.arange(0, 1.01, 0.01), "nn_prediction")

    risk = ctrl.Consequent(np.arange(0, 1.01, 0.01), "risk")

    age["young"] = fuzz.trimf(age.universe, [20, 35, 50])
    age["middle"] = fuzz.trimf(age.universe, [40, 55, 70])
    age["elderly"] = fuzz.trimf(age.universe, [60, 70, 80])

    blood_pressure["low"] = fuzz.trimf(blood_pressure.universe, [90, 100, 120])
    blood_pressure["normal"] = fuzz.trimf(blood_pressure.universe, [110, 130, 150])
    blood_pressure["high"] = fuzz.trimf(blood_pressure.universe, [140, 160, 200])

    cholesterol["low"] = fuzz.trimf(cholesterol.universe, [100, 150, 200])
    cholesterol["normal"] = fuzz.trimf(cholesterol.universe, [150, 250, 350])
    cholesterol["high"] = fuzz.trimf(cholesterol.universe, [300, 400, 500])

    heart_rate["low"] = fuzz.trimf(heart_rate.universe, [60, 80, 100])
    heart_rate["normal"] = fuzz.trimf(heart_rate.universe, [90, 120, 150])
    heart_rate["high"] = fuzz.trimf(heart_rate.universe, [140, 170, 200])

    nn_prediction["low"] = fuzz.trimf(nn_prediction.universe, [0, 0.2, 0.4])
    nn_prediction["medium"] = fuzz.trimf(nn_prediction.universe, [0.3, 0.5, 0.7])
    nn_prediction["high"] = fuzz.trimf(nn_prediction.universe, [0.6, 0.8, 1.0])

    risk["low"] = fuzz.trimf(risk.universe, [0, 0.2, 0.4])
    risk["medium"] = fuzz.trimf(risk.universe, [0.3, 0.5, 0.7])
    risk["high"] = fuzz.trimf(risk.universe, [0.6, 0.8, 1.0])

    rule1 = ctrl.Rule(nn_prediction["high"], risk["high"])
    rule2 = ctrl.Rule(age["elderly"] & blood_pressure["high"], risk["high"])
    rule3 = ctrl.Rule(cholesterol["high"] & heart_rate["high"], risk["high"])
    rule4 = ctrl.Rule(
        age["elderly"] & cholesterol["high"] & heart_rate["high"], risk["high"]
    )
    rule5 = ctrl.Rule(
        blood_pressure["high"] & cholesterol["high"] & nn_prediction["high"],
        risk["high"],
    )

    rule6 = ctrl.Rule(
        age["middle"] & blood_pressure["normal"] & cholesterol["normal"], risk["medium"]
    )
    rule7 = ctrl.Rule(
        age["elderly"] & blood_pressure["normal"] & heart_rate["normal"], risk["medium"]
    )
    rule8 = ctrl.Rule(
        age["middle"] & cholesterol["high"] & heart_rate["normal"], risk["medium"]
    )
    rule9 = ctrl.Rule(
        blood_pressure["high"] & cholesterol["normal"] & nn_prediction["medium"],
        risk["medium"],
    )
    rule10 = ctrl.Rule(
        age["young"] & blood_pressure["high"] & heart_rate["normal"], risk["medium"]
    )

    rule11 = ctrl.Rule(
        age["young"] & blood_pressure["normal"] & nn_prediction["low"], risk["low"]
    )
    rule12 = ctrl.Rule(
        age["young"] & cholesterol["normal"] & heart_rate["normal"], risk["low"]
    )
    rule13 = ctrl.Rule(
        blood_pressure["normal"]
        & cholesterol["normal"]
        & heart_rate["normal"]
        & nn_prediction["low"],
        risk["low"],
    )
    rule14 = ctrl.Rule(
        age["middle"]
        & blood_pressure["normal"]
        & cholesterol["normal"]
        & heart_rate["normal"]
        & nn_prediction["low"],
        risk["low"],
    )
    rule15 = ctrl.Rule(
        age["young"]
        & blood_pressure["normal"]
        & cholesterol["low"]
        & heart_rate["normal"],
        risk["low"],
    )

    rule16 = ctrl.Rule(age["elderly"] & blood_pressure["low"], risk["medium"])
    rule17 = ctrl.Rule(heart_rate["high"] & blood_pressure["low"], risk["medium"])
    rule18 = ctrl.Rule(
        age["young"] & cholesterol["high"] & nn_prediction["medium"], risk["medium"]
    )
    rule19 = ctrl.Rule(
        age["middle"] & heart_rate["low"] & blood_pressure["normal"], risk["medium"]
    )
    rule20 = ctrl.Rule(
        cholesterol["high"] & blood_pressure["high"] & heart_rate["low"], risk["high"]
    )

    system = ctrl.ControlSystem(
        [
            rule1,
            rule2,
            rule3,
            rule4,
            rule5,
            rule6,
            rule7,
            rule8,
            rule9,
            rule10,
            rule11,
            rule12,
            rule13,
            rule14,
            rule15,
            rule16,
            rule17,
            rule18,
            rule19,
            rule20,
        ]
    )
    return ctrl.ControlSystemSimulation(system)


class HybridSystem:
    def __init__(self, nn_model, fuzzy_system, scaler, numerical_cols):
        self.nn_model = nn_model
        self.fuzzy_system = fuzzy_system
        self.scaler = scaler
        self.numerical_cols = numerical_cols

    def predict(self, patient_data):
        try:
            self.nn_model.eval()
            with torch.no_grad():
                patient_tensor = torch.FloatTensor(patient_data.values.reshape(1, -1))
                nn_pred = self.nn_model(patient_tensor).item()

            numerical_data = patient_data[self.numerical_cols]
            original_values = self.scaler.inverse_transform(
                numerical_data.values.reshape(1, -1)
            )[0]

            print("Original values:", original_values)
            print("NN prediction:", nn_pred)

            self.fuzzy_system.input["age"] = float(original_values[0])
            self.fuzzy_system.input["blood_pressure"] = float(original_values[1])
            self.fuzzy_system.input["cholesterol"] = float(original_values[2])
            self.fuzzy_system.input["heart_rate"] = float(original_values[3])
            self.fuzzy_system.input["nn_prediction"] = float(nn_pred)

            print("Computing fuzzy system...")
            self.fuzzy_system.compute()

            final_risk = self.fuzzy_system.output["risk"]
            print(f"Fuzzy system output: {final_risk}")

            return final_risk

        except Exception as e:
            print(f"Error in prediction: {e}")
            return nn_pred


def plot_training_history(losses):
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title("Neural Network Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()


def plot_fuzzy_memberships(fuzzy_system):
    universes = {
        "age": np.arange(20, 81, 1),
        "blood_pressure": np.arange(90, 201, 1),
        "cholesterol": np.arange(100, 501, 1),
        "heart_rate": np.arange(60, 201, 1),
    }

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    for idx, (var, universe) in enumerate(universes.items()):
        ax = axes[idx // 2, idx % 2]

        if var == "age":
            ax.plot(universe, fuzz.trimf(universe, [20, 20, 40]), label="young")
            ax.plot(universe, fuzz.trimf(universe, [35, 50, 65]), label="middle")
            ax.plot(universe, fuzz.trimf(universe, [60, 80, 80]), label="elderly")
        elif var == "blood_pressure":
            ax.plot(universe, fuzz.trimf(universe, [90, 90, 120]), label="low")
            ax.plot(universe, fuzz.trimf(universe, [110, 130, 150]), label="normal")
            ax.plot(universe, fuzz.trimf(universe, [140, 200, 200]), label="high")
        elif var == "cholesterol":
            ax.plot(universe, fuzz.trimf(universe, [100, 100, 200]), label="low")
            ax.plot(universe, fuzz.trimf(universe, [150, 250, 350]), label="normal")
            ax.plot(universe, fuzz.trimf(universe, [300, 500, 500]), label="high")
        elif var == "heart_rate":
            ax.plot(universe, fuzz.trimf(universe, [60, 60, 100]), label="low")
            ax.plot(universe, fuzz.trimf(universe, [90, 120, 150]), label="normal")
            ax.plot(universe, fuzz.trimf(universe, [140, 200, 200]), label="high")

        ax.set_title(f'{var.replace("_", " ").title()} Membership Functions')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()


def plot_predictions(y_true, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(y_true)), y_true, c="b", label="Actual", alpha=0.5)
    plt.scatter(range(len(y_pred)), y_pred, c="r", label="Predicted", alpha=0.5)
    plt.title("Actual vs Predicted Risk")
    plt.xlabel("Patient")
    plt.ylabel("Risk Level")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_predictions(y_true, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(y_true)), y_true, c="b", label="Actual", alpha=0.5)
    plt.scatter(range(len(y_pred)), y_pred, c="r", label="Predicted", alpha=0.5)
    plt.title("Actual vs Predicted Risk")
    plt.xlabel("Patient")
    plt.ylabel("Risk Level")
    plt.legend()
    plt.grid(True)
    plt.show()


def evaluate_models(X_test, y_test, hybrid_system):
    """Evaluate and compare neural network and hybrid system performance."""
    nn_predictions = []
    hybrid_predictions = []

    for idx, patient in X_test.iterrows():

        patient_tensor = torch.FloatTensor(patient.values.reshape(1, -1))
        with torch.no_grad():
            hybrid_system.nn_model.eval()
            nn_pred = hybrid_system.nn_model(patient_tensor).item()
        nn_predictions.append(nn_pred)

        hybrid_pred = hybrid_system.predict(patient)
        hybrid_predictions.append(hybrid_pred)

    nn_binary = [1 if pred >= 0.5 else 0 for pred in nn_predictions]
    hybrid_binary = [1 if pred >= 0.5 else 0 for pred in hybrid_predictions]

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    metrics = {
        "Neural Network": {
            "accuracy": accuracy_score(y_test, nn_binary),
            "precision": precision_score(y_test, nn_binary),
            "recall": recall_score(y_test, nn_binary),
            "f1": f1_score(y_test, nn_binary),
        },
        "Hybrid System": {
            "accuracy": accuracy_score(y_test, hybrid_binary),
            "precision": precision_score(y_test, hybrid_binary),
            "recall": recall_score(y_test, hybrid_binary),
            "f1": f1_score(y_test, hybrid_binary),
        },
    }

    return metrics, nn_predictions, hybrid_predictions


def plot_comparison(y_true, nn_preds, hybrid_preds):
    """Visualize predictions from both models compared to true values."""
    plt.figure(figsize=(12, 6))

    indices = range(len(y_true))
    plt.scatter(indices, y_true, c="blue", label="Actual", alpha=0.6, s=100)
    plt.scatter(indices, nn_preds, c="green", label="Neural Network", alpha=0.6, s=100)
    plt.scatter(indices, hybrid_preds, c="red", label="Hybrid System", alpha=0.6, s=100)

    plt.title("Comparison of Model Predictions")
    plt.xlabel("Patient Index")
    plt.ylabel("Risk Score")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_confusion_matrices(y_true, nn_preds, hybrid_preds):
    """Plot confusion matrices for both models."""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    nn_binary = [1 if pred >= 0.5 else 0 for pred in nn_preds]
    hybrid_binary = [1 if pred >= 0.5 else 0 for pred in hybrid_preds]

    nn_cm = confusion_matrix(y_true, nn_binary)
    hybrid_cm = confusion_matrix(y_true, hybrid_binary)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(nn_cm, annot=True, fmt="d", ax=ax1, cmap="Blues")
    ax1.set_title("Neural Network\nConfusion Matrix")
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")

    sns.heatmap(hybrid_cm, annot=True, fmt="d", ax=ax2, cmap="Blues")
    ax2.set_title("Hybrid System\nConfusion Matrix")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")

    plt.tight_layout()
    plt.show()


def analyze_predictions(data, nn_preds, hybrid_preds, scaler, numerical_cols):
    """Analyze how different input factors affect predictions."""

    original_values = scaler.inverse_transform(data[numerical_cols])

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    factors = ["Age", "Blood Pressure", "Cholesterol", "Heart Rate"]

    for idx, (factor, values) in enumerate(zip(factors, original_values.T)):
        ax = axes[idx // 2, idx % 2]

        scatter = ax.scatter(
            values, hybrid_preds, c=nn_preds, cmap="viridis", alpha=0.6
        )

        ax.set_xlabel(factor)
        ax.set_ylabel("Hybrid System Prediction")
        ax.set_title(f"{factor} vs Predictions")

        plt.colorbar(scatter, ax=ax, label="Neural Network Prediction")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    data, scaler, numerical_cols = prepare_data(
        "/home/chukhran/datasets/Heart Disease Dataset UCI/HeartDiseaseTrain-Test.csv"
    )
    X = data.drop("target", axis=1)
    y = data["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    nn_model, losses = train_nn_model(X_train, y_train, X_train.shape[1])
    plot_training_history(losses)

    fuzzy_system = create_fuzzy_system()
    plot_fuzzy_memberships(fuzzy_system)

    hybrid_system = HybridSystem(nn_model, fuzzy_system, scaler, numerical_cols)

    predictions = []
    for idx, patient in X_test.iterrows():
        risk = hybrid_system.predict(patient)
        predictions.append(risk)

    plot_predictions(y_test, predictions)

    metrics, nn_predictions, hybrid_predictions = evaluate_models(
        X_test, y_test, hybrid_system
    )

    print("\nModel Performance Metrics:")
    print("-" * 50)
    for model, metric in metrics.items():
        print(f"\n{model}:")
        for name, value in metric.items():
            print(f"{name}: {value:.3f}")

    plot_comparison(y_test, nn_predictions, hybrid_predictions)
    plot_confusion_matrices(y_test, nn_predictions, hybrid_predictions)
    analyze_predictions(
        X_test, nn_predictions, hybrid_predictions, scaler, numerical_cols
    )
