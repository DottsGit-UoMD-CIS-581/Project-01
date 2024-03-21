"""
CIS 581 - Computational Learning
Author: Nicholas Butzke
Project 1
"""
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

def load_data(filename):
    data = np.loadtxt(filename)
    x, y = data[:, 0], data[:, 1]
    return x, y

def transform_features(X, degree):
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X.reshape(-1, 1))
    # X_poly_scaled = StandardScaler().fit_transform(X_poly)
    return X_poly

def cross_validate(X, y, degrees, lambdas, folds=12):
    # Set beginning variable, shuffle keeps the data in the same order
    kf = KFold(n_splits=folds, shuffle=False)
    best_val_rmse = np.inf
    best_degree = 0
    best_lambda = 0
    best_model = None
    model_28 = None
    lambda_vals = [0]
    flag_28 = False
    avg_train_rmse_per_degree = []
    avg_val_rmse_per_degree = []
    degree_28_cv_data:list[list, list, list, float] = [[], [], [], np.inf]
    setup_output_files()

    # Loop through each polynomial degree
    for degree in degrees:
        if degree == 29:
            lambda_vals = lambdas
            degree = 28
            flag_28 = True
        # Loop through each lambda value
        for lambda_val in lambda_vals:
            avg_validation_rmse = 0
            avg_train_rmse = 0

            # Main cross validation section
            for train_index, val_index in kf.split(X):
                X_scaled = StandardScaler().fit_transform(X.reshape(-1,1))
                X_poly = PolynomialFeatures(degree=degree).fit_transform(X_scaled)
                X_train, X_val = X_poly[train_index], X_poly[val_index]                
                y_train, y_val = y[train_index], y[val_index]
                
                # ONLY SCALE TRAINING DATA NOT VALIDATION DATA
                x_scaler = StandardScaler()
                X_train_scaled = x_scaler.fit_transform(X_train)
                X_val_scaled = x_scaler.transform(X_val)
                y_scaler = StandardScaler()
                y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

                # Train the model
                model = Ridge(alpha=lambda_val, solver='cholesky')
                model.fit(X_train_scaled, y_train_scaled)

                # Make predictions on the validation set
                y_val_pred_scaled = model.predict(X_val_scaled)
                y_train_pred_scaled = model.predict(X_train_scaled)

                y_val_pred = y_scaler.inverse_transform(y_val_pred_scaled.reshape(-1,1)).ravel()
                y_train_pred = y_scaler.inverse_transform(y_train_pred_scaled.reshape(-1,1)).ravel()

                avg_validation_rmse += np.sqrt(mean_squared_error(y_val, y_val_pred))
                avg_train_rmse += np.sqrt(mean_squared_error(y_train, y_train_pred))
            avg_validation_rmse /= folds
            avg_train_rmse /= folds

            # Storing the best output
            if not flag_28 and avg_validation_rmse < best_val_rmse:
                best_val_rmse = avg_validation_rmse
                best_train_rmse = avg_train_rmse
                best_degree = degree
                best_model = model
            
            # Storing the output for degree 28
            if flag_28 and avg_validation_rmse < degree_28_cv_data[3]:
                #degree_28_cv_data = [lambda_val, avg_validation_rmse, avg_train_rmse]
                degree_28_cv_data[3] = avg_validation_rmse
                best_lambda = lambda_val
                model_28 = model
            if flag_28:
                degree_28_cv_data[0].append(lambda_val)
                degree_28_cv_data[1].append(avg_validation_rmse)
                degree_28_cv_data[2].append(avg_train_rmse)

            # Recording the data at every lambda and degree combination
            record_cv_data(degree, lambda_val,  avg_train_rmse, avg_validation_rmse, level='full')
        # Recording the best data at every degree
        record_cv_data(degree, best_lambda, avg_train_rmse, avg_validation_rmse, level='partial')
        if not flag_28:
            avg_train_rmse_per_degree.append(avg_train_rmse)
            avg_val_rmse_per_degree.append(avg_validation_rmse)

    return best_model, model_28, best_degree, best_lambda, avg_train_rmse_per_degree, avg_val_rmse_per_degree, degree_28_cv_data

def setup_output_files():
    fieldnames = ['Degree', 'Lambda', 'CV Train RMSE', 'CV Test RMSE']
    with open('cv_data_full.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    with open('cv_data_partial.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

def record_cv_data(degree, lambda_val, avg_train_rmse, avg_validation_rmse, level):
    filename = f'cv_data_{level}.csv'
    fieldnames = ['Degree', 'Lambda', 'CV Train RMSE', 'CV Test RMSE']
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)        
        writer.writerow({'Degree': degree, 'Lambda': lambda_val, 'CV Train RMSE': avg_train_rmse, 'CV Test RMSE': avg_validation_rmse})

# Load the data
X_train, y_train = load_data('Data/train.dat')
X_test, y_test = load_data('Data/test.dat')

# Scale the input and output
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train.reshape(-1, 1)).ravel()
X_test_scaled = scaler_X.transform(X_test.reshape(-1, 1)).ravel()
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

# Set the parameters and cross validate
# As a bandaid to a bug I am overshooting the range by 1 and then using 29 to signal to loop through lambdas
degrees = range(0, 30)
lambdas = np.exp(np.arange(-30, 11, 2))
lambdas = np.insert(lambdas, 0, 0)
best_model, model_28, best_degree, best_lambda, avg_train_rmse_per_degree, avg_val_rmse_per_degree, degree_28_cv_data = cross_validate(X_train, y_train, degrees, lambdas)

# Transform the data
X_train_poly = transform_features(X_train_scaled, best_degree)
X_test_poly = transform_features(X_test_scaled, best_degree)
X_train_poly_28 = transform_features(X_train_scaled, 28)
X_test_poly_28 = transform_features(X_test_scaled, 28)


# Set, fit, and predict the output in scaled space
#best_model = Ridge(alpha=0)
best_model.fit(X_train_poly, y_train_scaled)
y_train_pred_scaled = best_model.predict(X_train_poly)
y_test_pred_scaled = best_model.predict(X_test_poly)

# Do it also for the 28 degree polynomial with its lambda
#model_28 = Ridge(alpha=best_lambda)
model_28.fit(X_train_poly_28, y_train_scaled)
y_28_train_pred_scaled = model_28.predict(X_train_poly_28)
y_28_test_pred_scaled = model_28.predict(X_test_poly_28)

# Unscale the predicted output
y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).ravel()
y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).ravel()
# Do it also for the 28 degree polynomial
y_28_train_pred = scaler_y.inverse_transform(y_28_train_pred_scaled.reshape(-1, 1)).ravel()
y_28_test_pred = scaler_y.inverse_transform(y_28_test_pred_scaled.reshape(-1, 1)).ravel()
# Optain the RMSE values in original space
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
# Do it also for the 28 degree polynomial
train_28_rmse = np.sqrt(mean_squared_error(y_train, y_28_train_pred))
test_28_rmse = np.sqrt(mean_squared_error(y_test, y_28_test_pred))

print("-"*75)
print("d* Model")
print("-"*75)
print(f"Degree: {best_degree}")
print(f"Lambda: {0}")
print(f"Train RMSE: {train_rmse}")
print(f"Test RMSE: {test_rmse}")
print(f"Coefficient-weights:\n{best_model.coef_}")
print("-"*75)
print("λ* Model")
print("-"*75)
print("Degree: 28")
print(f"Lambda: {best_lambda}")
print(f"Train RMSE: {degree_28_cv_data[3]}")
print(f"Test RMSE: {test_28_rmse}")
print(f"Coefficient-weights:\n{model_28.coef_}")

# Generate predictions in the scaled space for the full line
range_x = np.linspace(X_train_scaled.min(), X_train_scaled.max(), 1000)
range_x_poly = transform_features(range_x, best_degree)
predictions_scaled = best_model.predict(range_x_poly)

# Generate predictions in the scaled space for the full line on degree 28
range_x_28 = np.linspace(X_train_scaled.min(), X_train_scaled.max(), 1000)
range_x_poly_28 = transform_features(range_x_28, 28)
predictions_scaled_28 = model_28.predict(range_x_poly_28)

# Inverse transform the predictions to the original scale
predictions_original = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).ravel()
predictions_original_28 = scaler_y.inverse_transform(predictions_scaled_28.reshape(-1, 1)).ravel()

fig, axs = plt.subplots(3, 2, figsize=(15, 12))

# Plotting
range_x_original = scaler_X.inverse_transform(range_x.reshape(-1, 1)).ravel()
range_x_original_28 = scaler_X.inverse_transform(range_x_28.reshape(-1, 1)).ravel()
axs[0][0].plot(range_x_original, predictions_original, color='red', label=f'Polynomial curve (Degree {best_degree})')
axs[0][0].scatter(X_train, y_train, color='orange', label='Training data')
axs[0][0].scatter(X_test, y_test, color='blue', label='Test data')
axs[0][0].legend()
axs[0][0].set_xlabel('Week number')
axs[0][0].set_ylabel('Number of cases')
axs[0][0].set_title('COVID-19 Weekly Cases Polynomial Regression (d*)')

axs[0][1].plot(range_x_original_28, predictions_original_28, color='red', label='Polynomial curve (Degree 28)')
axs[0][1].scatter(X_train, y_train, color='orange', label='Training data')
axs[0][1].scatter(X_test, y_test, color='blue', label='Test data')
axs[0][1].legend()
axs[0][1].set_xlabel('Week number')
axs[0][1].set_ylabel('Number of cases')
axs[0][1].set_title('COVID-19 Weekly Cases Polynomial Regression (λ*)')

axs[1][0].plot(range(len(avg_train_rmse_per_degree)), avg_train_rmse_per_degree, marker='o', linestyle='-')
axs[1][0].set_xlabel('Degree of Polynomial')
axs[1][0].set_ylabel('Average Train RMSE')
axs[1][0].set_title('Average Train RMSE vs. Degree of Polynomial (d*)')
axs[1][0].grid(True)

axs[1][1].semilogx(degree_28_cv_data[0], degree_28_cv_data[2], marker='o', linestyle='-')
axs[1][1].set_xlabel('Lambda Value')
axs[1][1].set_ylabel('Average Train RMSE')
axs[1][1].set_title('Average Train RMSE vs. Lambda Value (λ*)')
axs[1][1].grid(True)

axs[2][0].plot(range(len(avg_val_rmse_per_degree)), avg_val_rmse_per_degree, marker='o', linestyle='-')
axs[2][0].set_xlabel('Degree of Polynomial')
axs[2][0].set_ylabel('Average Validation RMSE')
axs[2][0].set_title('Average Validation RMSE vs. Degree of Polynomial (d*)')
axs[2][0].grid(True)

axs[2][1].semilogx(degree_28_cv_data[0], degree_28_cv_data[1], marker='o', linestyle='-')
axs[2][1].set_xlabel('Lambda Value')
axs[2][1].set_ylabel('Average Validation RMSE')
axs[2][1].set_title('Average Validation RMSE vs. Lambda Value (λ*)')
axs[2][1].grid(True)

plt.tight_layout()
plt.show()
