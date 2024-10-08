from sklearn.linear_model import Lasso
from Linear_Regression import X_train, y_train, X_test, y_test
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np 
import joblib
from sklearn.preprocessing import StandardScaler


# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
# Lưu scaler vào file
joblib.dump(scaler, 'scaler.pkl')

# Train Lasso regression model (choose an appropriate alpha value)

alpha = 0.01
model_lasso = Lasso(alpha=alpha)
model_lasso.fit(X_train_scaled, y_train)

# Predict on test set for both models
y_pred_lasso = model_lasso.predict(X_test)



# Evaluate Lasso regression model
mae_lasso = mean_absolute_error(y_test, y_pred_lasso)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)  


data = {
    'Metric': ['MAE', 'MSE', 'R²'],
    'Value': [mae_lasso, mse_lasso, r2_lasso]
}
model_evaluation_table = pd.DataFrame(data)

# Print model evaluation table
print("\n\nModel Evaluation Table:")
print(model_evaluation_table.to_string(index=False))
print("\n\n")
# # #Lưu mô hình
model_filename = 'trained_LS.pkl'
joblib.dump(model_lasso, model_filename)
print(f"Model save to {model_filename}")

# # Tạo một danh sách các giá trị alpha muốn thử nghiệm
# alpha_values = [0.01, 0.1, 1, 10]

# # Tạo một đối tượng Lasso
# lasso = Lasso()

# # Tạo một đối tượng GridSearchCV để tìm giá trị alpha tốt nhất
# param_grid = {'alpha': alpha_values}
# grid = GridSearchCV(lasso, param_grid, cv=5)

# # Thực hiện tìm kiếm
# grid.fit(X_train, y_train)

# # In ra giá trị alpha tốt nhất
# print("Best alpha:", grid.best_params_['alpha'])

# best_model = grid.best_estimator_
# y_pred = best_model.predict(X_test)

# Vẽ biểu đồ dự đoán so với thực tế
# plt.figure(figsize=(8,6))
# plt.scatter(y_test, y_pred_lasso, color='black')
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
# plt.xlabel("Giá trị thực tế")
# plt.ylabel("Giá trị dự đoán")
# plt.title("Prediction vs Actual")
# plt.show()



# from sklearn.model_selection import learning_curve
# import matplotlib.pyplot as plt

# # # ... (các bước huấn luyện mô hình như đã thực hiện trước đó)

# train_sizes, train_scores, test_scores = learning_curve(model_lasso, X_train, y_train, cv=5)

# train_mean = np.mean(train_scores, axis=1)
# train_std = np.std(train_scores, axis=1)
# test_mean = np.mean(test_scores, axis=1)
# test_std = np.std(test_scores, axis=1)

# plt.plot(train_sizes, train_mean, label='Training score')
# plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
# plt.plot(train_sizes, test_mean, label='Cross-validation score')
# plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
# plt.title('Learning Curve')
# plt.xlabel('Training set size')
# plt.ylabel('Accuracy')
# plt.legend(loc="best")
# plt.show()

