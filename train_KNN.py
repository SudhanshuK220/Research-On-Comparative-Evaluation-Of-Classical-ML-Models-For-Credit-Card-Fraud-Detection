from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from credit_card_preprocessing import preprocess_data, evaluate_model

def train_knn(X_train, y_train, X_test, y_test):
    print("\nTraining K-Nearest Neighbors (KNN)...")
    knn_model = KNeighborsClassifier(n_neighbors=39, n_jobs=-1)
    knn_model.fit(X_train, y_train)
    
    print("Evaluating KNN on the testing set... ")
    y_pred = knn_model.predict(X_test)
    
    print("Calculating ROC-AUC Score...")
    y_pred_proba = knn_model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Pass roc_auc to our evaluation function so it saves to the JSON
    evaluate_model("K-Nearest Neighbors", y_test, y_pred, roc_auc)

if __name__ == "__main__":
    print(" Starting K-Nearest Neighbors (KNN) Pipeline ")
    X_train, X_test, y_train, y_test = preprocess_data()
    if X_train is not None:
        train_knn(X_train, y_train, X_test, y_test)