from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
from Linear_Regression import model
from Lasso_Regression import model_lasso
from Neural_Network import mlp_model
from Linear_Regression import X_train, y_train, X_test, y_test
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np 


# Xây dựng stacking model với LinearRegression là mô hình meta (meta model)
stacking_model = StackingRegressor(
    estimators=[('mlp', mlp_model), ('lasso', model_lasso), ('linear', model)],
    final_estimator=Lasso(alpha=0.01)
)

# Huấn luyện mô hình stacking
stacking_model.fit(X_train, y_train)

# Dự đoán với tập kiểm tra
y_pred_stack = stacking_model.predict(X_test)

# Đánh giá hiệu suất của mô hình stacking
mae_stack = mean_absolute_error(y_test, y_pred_stack)
mse_stack = mean_squared_error(y_test, y_pred_stack)
r2_stack = r2_score(y_test, y_pred_stack)

data = {
    'Metric': ['MAE', 'MSE', 'R²'],
    'Value': [mae_stack, mse_stack, r2_stack]
}
model_evaluation_table = pd.DataFrame(data)

# # Print model evaluation table
# print("\n\nModel Evaluation Table:")
# print(model_evaluation_table.to_string(index=False))
# print("\n\n")

# # Vẽ biểu đồ dự đoán so với thực tế
# plt.figure(figsize=(8,6))
# plt.scatter(y_test, y_pred_stack, color='brown')
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
# plt.xlabel("Giá trị thực tế")
# plt.ylabel("Giá trị dự đoán")
# plt.title("Prediction vs Actual")
# plt.show()


# from sklearn.model_selection import learning_curve

# train_sizes, train_scores, test_scores = learning_curve(stacking_model, X_train, y_train, cv=5)

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
