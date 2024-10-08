import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



#Đọc dữ liệu từ file csv
data = pd.read_csv('Data.csv')
#Xử lý dữ liệu
data['Extracurricular_Activities'] = data['Extracurricular_Activities'].apply(lambda x: 1 if x == 'Yes' else 0)
X = data.drop(columns=['Performance_Index'])
y = data['Performance_Index']



# #Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# #Huấn luyện mô hình hồi quy tuyến tính
model = LinearRegression()
model.fit(X_train, y_train)

# #Kiểm tra mô hình bằng cạc sử dụng tập dữ liệu kiểm tra
y_pred = model.predict(X_test)

# #Đánh giá mô hình
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

data = {
    'Metric': ['MAE', 'MSE', 'R²'],
    'Value': [mae, mse, r2]
}
model_evaluation_table = pd.DataFrame(data)

# Print model evaluation table
print("\n\nModel Evaluation Table:")
print(model_evaluation_table.to_string(index=False))
print("\n\n")
# # #Lưu mô hình
# model_filename = 'trained_LR.pkl'
# joblib.dump(model, model_filename)
# print(f"Model save to {model_filename}")



# Vẽ biểu đồ dự đoán so với thực tế
# plt.figure(figsize=(8,6))
# plt.scatter(y_test, y_pred, color='green')
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
# plt.xlabel("Giá trị thực tế")
# plt.ylabel("Giá trị dự đoán")
# plt.title("Prediction vs Actual")
# plt.show()



# # Chia dữ liệu thành tập huấn luyện (70%), kiểm định (20%) và thử nghiệm (10%)
# X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
# X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

# # Huấn luyện mô hình trên tập huấn luyện
# model.fit(X_train, y_train)

# # Đánh giá trên tập kiểm định
# y_val_pred = model.predict(X_val)
# mae_val = mean_absolute_error(y_val, y_val_pred)
# mse_val = mean_squared_error(y_val, y_val_pred)
# r2_val = r2_score(y_val, y_val_pred)

# # Đánh giá trên tập thử nghiệm
# y_test_pred = model.predict(X_test)
# mae_test = mean_absolute_error(y_test, y_test_pred)
# mse_test = mean_squared_error(y_test, y_test_pred)
# r2_test = r2_score(y_test, y_test_pred)

# print("Validation Metrics:")
# print("MAE:", mae_val)
# print("MSE:", mse_val)
# print("R2:", r2_val)

# print("Test Metrics:")
# print("MAE:", mae_test)
# print("MSE:", mse_test)
# print("R2:", r2_test)


# from sklearn.model_selection import learning_curve
# import matplotlib.pyplot as plt

# # ... (các bước huấn luyện mô hình như đã thực hiện trước đó)

# train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, cv=5)

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

