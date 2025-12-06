# pip install pyqt5 joblib scikit-learn lightgbm pandas
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

# Leia projekti juurkaust
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Lae mudelid
rf = joblib.load(os.path.join(BASE_DIR, "models", "randomForest.pkl"))
lgb = joblib.load(os.path.join(BASE_DIR, "models", "lgbModel.pkl"))

# Lae treeningu X.columns
X_columns = joblib.load(os.path.join(BASE_DIR, "models", "X_columns.pkl"))

# Lae dropdown väärtused
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

        # Year (sisestus)
        self.year = QLineEdit()
        layout.addWidget(QLabel("Year"))
        layout.addWidget(self.year)

        # Mileage (sisestus)
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

        # Ennustuse nupp
        self.predict_button = QPushButton("Predict Price")
        self.predict_button.clicked.connect(self.predict)
        layout.addWidget(self.predict_button)

        # Tulemuse label
        self.result_label = QLabel("")
        self.result_label.setFont(QFont("Arial", 12))
        layout.addWidget(self.result_label)

        self.setLayout(layout)

    def predict(self):
        try:
            # Tee DataFrame sisendist
            uus_auto = pd.DataFrame({
                'model_grouped':[self.model_grouped.currentText()],
                'year':[int(self.year.text())],
                'mileage':[int(self.mileage.text())],
                'fuelType':[self.fuelType.currentText()],
                'transmission':[self.transmission.currentText()],
                'engineSize':[float(self.engineSize.currentText())],
                'Manufacturer':[self.manufacturer.currentText()]
            })

            # Tee dummies ja reindexi
            uus_auto = pd.get_dummies(uus_auto)
            uus_auto = uus_auto.reindex(columns=X_columns, fill_value=0)

            # Ennusta mõlema mudeliga
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
