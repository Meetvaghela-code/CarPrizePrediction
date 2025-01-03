# Car Price Prediction Project

## Overview
This repository contains a machine learning project designed to predict the price of used cars based on their attributes such as name, company, year of manufacture, kilometers driven, and fuel type. The dataset is sourced from Quikr, and extensive data cleaning and preprocessing steps have been performed to prepare the data for modeling.

---

## Features
- **Data Cleaning**: Removal of inconsistent and irrelevant data.
- **Feature Engineering**: Encoding categorical variables and preparing the dataset.
- **Modeling**: Linear Regression model integrated with a preprocessing pipeline.
- **Evaluation**: Performance measured using R-squared (²) score.
- **Deployment Ready**: The model is saved as a `.pkl` file for easy reuse.

---

## Dataset
The dataset `quikr_car.csv` contains the following columns:
- **name**: Car name (e.g., Maruti Suzuki Swift).
- **year**: Year of manufacture.
- **Price**: Car price (target variable).
- **kms_driven**: Kilometers driven.
- **fuel_type**: Type of fuel (e.g., Petrol, Diesel).
- **company**: Car brand.

---

## Steps

### 1. Data Cleaning
- Convert non-numeric year values to integers.
- Remove rows with "Ask For Price" in the `Price` column and convert to integers.
- Extract numeric values from `kms_driven` and convert to integers.
- Drop rows with missing `fuel_type` values.
- Simplify the `name` column to retain the first three words.
- Filter out cars with prices above 6 million.

### 2. Data Preprocessing
- Split the dataset into features (`X`) and target (`y`).
- Use `OneHotEncoder` for encoding categorical columns (`name`, `company`, `fuel_type`).

### 3. Model Training
- Use Linear Regression as the predictive model.
- Create a pipeline to automate preprocessing and training:
  - `ColumnTransformer` for encoding categorical features.
  - Linear Regression for prediction.

### 4. Evaluation
- Evaluate the model using the R-squared score on test data.

### 5. Saving the Model
- Save the trained model pipeline using `pickle` for future use.

---

## Installation
### Prerequisites
Ensure the following Python libraries are installed:
- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `pickle`

### Steps
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/car-price-prediction.git
    ```
2. Navigate to the project directory:
    ```bash
    cd car-price-prediction
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Place the `quikr_car.csv` file in the project directory.
5. Run the script:
    ```bash
    python car_price_prediction.py
    ```

---

## Results
The Linear Regression model achieved a high R-squared score, demonstrating its effectiveness in predicting car prices based on the given features.

---

## Usage
Use the saved model `LinearRegression.pkl` to predict car prices for new data:
```python
import pickle
model = pickle.load(open("LinearRegression.pkl", 'rb'))
predicted_price = model.predict(new_data)
```

---

## Future Work
- Incorporate additional features like car condition and location.
- Experiment with advanced models such as Random Forest and Gradient Boosting.
- Deploy the model as a web application for real-time predictions.

---

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

---

## License
This project is licensed under the MIT License.

---

## Acknowledgments
- Dataset sourced from Quikr.
- Thanks to the open-source community for the tools and libraries used in this project.

---

## Contact
For any questions or issues, please contact:
- **Email**: vaghelameet765@gmail.com
- **GitHub**: (https://github.com/Meetvaghela-code)

