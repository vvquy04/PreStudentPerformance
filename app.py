from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import logging

# Thiết lập logging
logging.basicConfig(level=logging.DEBUG)

# Khởi tạo ứng dụng Flask
app = Flask(__name__)

# Load the saved Lasso regression model and scaler
try:
    lasso_model = joblib.load('trained_LS.pkl')
    scaler = joblib.load('scaler.pkl')
    logging.info("Mô hình và scaler đã được tải thành công")
except Exception as e:
    logging.error(f"Lỗi khi tải mô hình hoặc scaler: {str(e)}")

# Trang chủ với form nhập dữ liệu
@app.route('/')
def home():
    return render_template('index.html')

# Xử lý dự đoán khi nhận dữ liệu mới từ form
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Lấy dữ liệu từ form
            input_data = {
                'Hours_Studied': float(request.form['Hours_Studied']),
                'Previous_Scores': float(request.form['Previous_Scores']),
                'Extracurricular_Activities': int(request.form['Extracurricular_Activities']),
                'Sleep_Hours': float(request.form['Sleep_Hours']),
                'Sample_Question_Papers_Practiced': float(request.form['Sample_Question_Papers_Practiced'])
            }
            logging.info(f"Dữ liệu đầu vào: {input_data}")

            # Chuyển đổi dữ liệu thành DataFrame để có tên cột
            input_df = pd.DataFrame([input_data])

            # Chuyển đổi và chuẩn hóa dữ liệu
            input_scaled = scaler.transform(input_df)
            logging.info(f"Dữ liệu sau khi chuẩn hóa: {input_scaled}")

            # Dự đoán kết quả với mô hình đã huấn luyện
            lasso_pred = lasso_model.predict(input_scaled)
            logging.info(f"Kết quả dự đoán: {lasso_pred}")

            # Tạo biến kết quả dự đoán
            results = {
                "du_doan": lasso_pred[0]
            }

            return render_template('index.html', predictions=results)

        except Exception as e:
            logging.error(f"Lỗi trong quá trình dự đoán: {str(e)}")
            return render_template('index.html', error=f"Đã xảy ra lỗi: {str(e)}")

# Chạy ứng dụng Flask
if __name__ == '__main__':
    app.run(debug=True)