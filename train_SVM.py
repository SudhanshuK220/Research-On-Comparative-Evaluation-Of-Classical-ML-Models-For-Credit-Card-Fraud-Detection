from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score
from credit_card_preprocessing import preprocess_data, evaluate_model

def train_svm(X_train, y_train, X_test, y_test):
    print("\nTraining SVM (LinearSVC)... This might take a minute or two.")
    svm_model = LinearSVC(random_state=42, dual=False, max_iter=500)
    svm_model.fit(X_train, y_train)
    
    print("Evaluating SVM on the testing set...")
    y_pred = svm_model.predict(X_test)
    
    
    print("Calculating ROC-AUC Score...")
    y_scores = svm_model.decision_function(X_test)
    roc_auc = roc_auc_score(y_test, y_scores)
    
    # Pass roc_auc to our evaluation function so it saves to the JSON
    evaluate_model("Support Vector Machine (LinearSVC)", y_test, y_pred, roc_auc)

if __name__ == "__main__":
    print(" Starting Support Vector Machine (SVM) Pipeline ")
    X_train, X_test, y_train, y_test = preprocess_data()
    if X_train is not None:
        train_svm(X_train, y_train, X_test, y_test)