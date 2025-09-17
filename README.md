# 📱 SMS Spam Classifier

A simple **Machine Learning project** that classifies SMS messages as **SPAM** or **NOT SPAM** using **Python, scikit-learn, and Streamlit**.  

This project demonstrates an end-to-end ML workflow: dataset handling, model training, evaluation, saving, and creating an interactive demo for recruiters or users.

---

## 📂 Project Structure

```
spam-classifier/
├─ data/                     # Dataset: SMSSpamCollection
├─ models/                   # Trained model saved here
├─ train.py                  # Script to train the model
├─ app.py                    # Streamlit interactive demo
├─ requirements.txt          # Project dependencies
└─ README.md
```

---

## 🛠 Tools & Libraries
- **Python 3.x**  
- **pandas** – Data handling  
- **scikit-learn** – Machine Learning (TF-IDF, Naive Bayes)  
- **joblib** – Save/load model  
- **Streamlit** – Interactive web demo  

---

## 💾 Dataset
- [SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)  
- Place the file `SMSSpamCollection` in the `data/` folder before running the scripts.

---

## ⚡ How to Run

### 1️⃣ Clone the repository
```bash
git clone https://github.com/EmranZZ/SPAM-Detection
cd spam-classifier
```

### 2️⃣ Create virtual environment & install dependencies
```bash
python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate   # On Linux/Mac

pip install -r requirements.txt
```

### 3️⃣ Train the Model
```bash
python train.py
```
- Trains the **Naive Bayes classifier**  
- Saves model in `models/spam_classifier.joblib`  

### 4️⃣ Run the Interactive Demo
```bash
streamlit run app.py
```
- Opens a browser  
- Enter any SMS message  
- Get **SPAM / NOT SPAM** prediction with probability  

---

## 📊 Results
- **Accuracy:** ~95%  
- Confusion matrix and classification report printed during training  

---

## 📸 Screenshot
*Add a screenshot of your Streamlit demo here for recruiters*  

Example:

![Demo Screenshot](screenshot.png)

---

## 💼 Author
**MOHAMMAD EMRAN AHMED**  
- GitHub: [github.com/EmranZZ](https://github.com/EmranZZ)  
- Email: ahmed777emran@gmail.com 
