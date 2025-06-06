## **📈 Stock Price Prediction using LSTM**
This project uses a Long Short-Term Memory (LSTM) neural network to predict stock prices based on historical stock data.

## 📌 Project Overview

Stock market prediction is a challenging task due to its dynamic and volatile nature. Traditional models struggle with sequential
data dependencies, which is where **LSTM** networks come into play. This project:
- Uses historical **Open**, **High**, **Low**, and **Close** prices
- Preprocesses and normalizes the data for training
- Builds an LSTM model to learn patterns in sequential data
- Predicts future stock prices based on past trends
- Visualizes the actual vs. predicted prices using an interactive dashboard

## 🧠 Technologies Used
- **Python 3.10+**
- **TensorFlow / Keras** – Deep learning framework
- **Pandas / NumPy** – Data manipulation and analysis
- **Matplotlib** – For charting
- **Streamlit** – To build a user-friendly web app interface

## 🧰 Files and Folders

Stock-price-prediction-using-LSTM/
│
├── app.py # Streamlit app to run model and show results

├── Stock price prediction using LSTM.ipynb # Jupyter Notebook for model building and evaluation

├── lstm_stock_model.h5 # Trained LSTM model file

├── stock_data.csv # Cleaned and preprocessed data

├── NSE-Tata-Global-Beverages-Limited.csv # Original stock price dataset

├── .gitignore

├── README.md # Project documentation

└── screenshots/ # Folder for UI/output screenshots

📸 Screenshots
🔹 Web Application Interface

![Screenshot 2025-06-06 102746](https://github.com/user-attachments/assets/a847f0da-88ce-4854-8dd7-43528a3f098c)

![Screenshot 2025-06-06 102633](https://github.com/user-attachments/assets/935e60c6-025b-4ca8-a82f-a679d9daaff8)


🔹 Prediction vs. Actual Graph

![Screenshot 2025-06-06 102937](https://github.com/user-attachments/assets/16e636c9-cd60-40df-ae90-21504dddcc7b)

![Screenshot 2025-06-06 102946](https://github.com/user-attachments/assets/e1e79e27-6f20-401e-b9fd-25bd842ed0df)



