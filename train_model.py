import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Sign MNIST format: first column = label, rest = pixels
data = pd.read_csv("sign_mnist_train.csv")

X = data.iloc[:, 1:].values # pixel values
y = data.iloc[:, 0]    # label numbers (0=A, 1=B, etc.)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=50)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

pickle.dump(model, open("model.pkl", "wb"))
print("Model saved!")