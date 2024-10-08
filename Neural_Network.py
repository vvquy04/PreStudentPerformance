from sklearn.neural_network import MLPRegressor
from Linear_Regression import X_train, y_train, X_test, y_test
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np 


# # Xác định các tham số cần thử nghiệm
# param_grid = {
#     'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100)],  # số lượng neuron và tầng ẩn
#     'activation': ['relu', 'tanh'],   # hàm kích hoạt
#     'solver': ['adam', 'sgd'],        # thuật toán tối ưu
#     'learning_rate': ['constant', 'adaptive'],  # tốc độ học
#     'max_iter': [500, 1000, 2000],  # số lần lặp (epoch)
#     'alpha': [0.0001, 0.001, 0.01]  # tham số điều chuẩn (regularization)
# }

# Sử dụng GridSearchCV để thử nghiệm với các tổ hợp tham số
# mlp = MLPRegressor()
# grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
# grid_search.fit(X_train, y_train)

# Huấn luyện mô hình MLP Regressor
mlp_model = MLPRegressor(hidden_layer_sizes=(50,), max_iter=2000, random_state=42, activation='relu', alpha=0.01, learning_rate='constant', solver='adam')
mlp_model.fit(X_train, y_train)

# Hiển thị tham số tốt nhất
# print("Best parameters found: ", grid_search.best_params_)

# Sử dụng mô hình với tham số tốt nhất để dự đoán
# best_model = grid_search.best_estimator_
# y_pred_mpl = best_model.predict(X_test)

# Dự đoán với tập dữ liệu kiểm tra
y_pred_mlp = mlp_model.predict(X_test)


# Đánh giá mô hình
mae_mlp = mean_absolute_error(y_test, y_pred_mlp)
mse_mlp = mean_squared_error(y_test, y_pred_mlp)
r2_mlp = r2_score(y_test, y_pred_mlp)

data = {
    'Metric': ['MAE', 'MSE', 'R²'],
    'Value': [mae_mlp, mse_mlp, r2_mlp]
}
model_evaluation_table = pd.DataFrame(data)

# Print model evaluation table
print("\n\nModel Evaluation Table:")
print(model_evaluation_table.to_string(index=False))
print("\n\n")


# # Vẽ biểu đồ dự đoán so với thực tế
# plt.figure(figsize=(8,6))
# plt.scatter(y_test, y_pred_mpl, color='blue')
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
# plt.xlabel("Giá trị thực tế")
# plt.ylabel("Giá trị dự đoán")
# plt.title("Prediction vs Actual")
# plt.show()

# from sklearn.model_selection import learning_curve

# train_sizes, train_scores, test_scores = learning_curve(mlp_model, X_train, y_train, cv=5)

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