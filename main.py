# Make sure you have ran the Models_and_price_prediction.ipynb
# pip install pyqt5 joblib scikit-learn lightgbm pandas

# This script launches a PyQt5 GUI application for car price prediction.
# It loads trained models (RandomForest and LightGBM) and metadata from the 'models' folder.
# The user can input car attributes (model, year, mileage, fuel type, transmission, engine size, manufacturer),
# and the app will predict the price using both models.
import sys
import os
import joblib
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QComboBox, QMessageBox
)
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QCompleter

# Find project root folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load both models
rf = joblib.load(os.path.join(BASE_DIR, "models", "randomForest.pkl"))
lgb = joblib.load(os.path.join(BASE_DIR, "models", "lgbModel.pkl"))

# Load training X.columns
X_columns = joblib.load(os.path.join(BASE_DIR, "models", "X_columns.pkl"))

# Load dropdown values
model_grouped_values = joblib.load(os.path.join(BASE_DIR, "models", "model_grouped_values.pkl"))
fuelType_values = joblib.load(os.path.join(BASE_DIR, "models", "fuelType_values.pkl"))
transmission_values = joblib.load(os.path.join(BASE_DIR, "models", "transmission_values.pkl"))
engineSize_values = joblib.load(os.path.join(BASE_DIR, "models", "engineSize_values.pkl"))
engineSize_values = sorted(engineSize_values)
manufacturer_values = joblib.load(os.path.join(BASE_DIR, "models", "manufacturer_values.pkl"))

class CarPredictionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Car Prediction App")
        self.setGeometry(200, 200, 500, 450)

        # Styling
        self.setStyleSheet("""
            QWidget { background-color: #f0f2f5; }
            QLabel { font-size: 14px; font-weight: bold; }
            QLineEdit, QComboBox {
                padding: 6px;
                border: 1px solid #ccc;
                border-radius: 4px;
            }
            QPushButton {
                background-color: #0078d7;
                color: white;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #005a9e; }
        """)

        layout = QVBoxLayout()

        # Model Grouped dropdown
        self.model_grouped = QComboBox()
        self.model_grouped.addItems(model_grouped_values)
        layout.addWidget(QLabel("Models"))
        layout.addWidget(self.model_grouped)

        completer = QCompleter(model_grouped_values)
        completer.setCaseSensitivity(False)
        self.model_grouped.setEditable(True)
        self.model_grouped.setCompleter(completer)

        # Year input
        self.year = QLineEdit()
        layout.addWidget(QLabel("Year"))
        layout.addWidget(self.year)

        # Mileage input
        self.mileage = QLineEdit()
        layout.addWidget(QLabel("Mileage"))
        layout.addWidget(self.mileage)

        # Fuel Type dropdown
        self.fuelType = QComboBox()
        self.fuelType.addItems(fuelType_values)
        layout.addWidget(QLabel("Fuel Type"))
        layout.addWidget(self.fuelType)

        # Transmission dropdown
        self.transmission = QComboBox()
        self.transmission.addItems(transmission_values)
        layout.addWidget(QLabel("Transmission"))
        layout.addWidget(self.transmission)

        # Engine Size dropdown
        self.engineSize = QComboBox()
        self.engineSize.addItems([str(v) for v in engineSize_values])
        layout.addWidget(QLabel("Engine Size"))
        layout.addWidget(self.engineSize)

        # Manufacturer dropdown
        self.manufacturer = QComboBox()
        self.manufacturer.addItems(manufacturer_values)
        layout.addWidget(QLabel("Manufacturer"))
        layout.addWidget(self.manufacturer)

        # Prediction button
        self.predict_button = QPushButton("Predict Price")
        self.predict_button.clicked.connect(self.predict)
        layout.addWidget(self.predict_button)

        # Result label
        self.result_label = QLabel("")
        self.result_label.setFont(QFont("Arial", 12))
        layout.addWidget(self.result_label)

        self.setLayout(layout)

    def predict(self):
        try:
            # Make a dataframe out of the inputs
            uus_auto = pd.DataFrame({
                'model_grouped':[self.model_grouped.currentText()],
                'year':[int(self.year.text())],
                'mileage':[int(self.mileage.text())],
                'fuelType':[self.fuelType.currentText()],
                'transmission':[self.transmission.currentText()],
                'engineSize':[float(self.engineSize.currentText())],
                'Manufacturer':[self.manufacturer.currentText()]
            })

            # Convert categorical variables into dummy/one-hot encoded columns
            # Ensure the DataFrame has the same structure as the training data
            uus_auto = pd.get_dummies(uus_auto)
            uus_auto = uus_auto.reindex(columns=X_columns, fill_value=0)

            # Predict the price with both models
            hind_rf = rf.predict(uus_auto)[0]
            hind_lgb = lgb.predict(uus_auto)[0]

            self.result_label.setText(
                f"RandomForest: {hind_rf:.2f}\nLightGBM: {hind_lgb:.2f}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CarPredictionApp()
    window.show()
    sys.exit(app.exec_())
