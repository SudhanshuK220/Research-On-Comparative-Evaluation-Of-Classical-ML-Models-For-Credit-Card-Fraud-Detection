import json
import pandas as pd
import os

def generate_comparative_table():
    file_path = "model_performance_results.json"
    
    # Check if the results file exists
    if not os.path.exists(file_path):
        print(f"Error: '{file_path}' not found.")
        print("Please ensure you have run your training scripts and the JSON file exists.")
        return

    # Load the saved data
    with open(file_path, 'r') as f:
        data = json.load(f)

    table_data = []

    # Loop through each model saved in the JSON file
    for model_name, metrics in data.items():
        # We focus on "1" because that represents the Fraud class
        class_1_metrics = metrics['classification_report']['1']
        
        # Extract the key metrics
        accuracy = metrics['classification_report']['accuracy']
        precision = class_1_metrics['precision']
        recall = class_1_metrics['recall']
        f1_score = class_1_metrics['f1-score']
        
        # Get ROC-AUC (if it's missing for some reason, return None)
        roc_auc = metrics.get('roc_auc_score', None)

        # Append to our table data
        table_data.append({
            "Models": model_name,
            "Accuracy": accuracy,
            "Precision (Fraud)": precision,
            "Recall (Fraud)": recall,
            "F1-Score (Fraud)": f1_score,
            "ROC-AUC": roc_auc
        })

    # Convert the dictionary into a Pandas DataFrame for neat tabular formatting
    df = pd.DataFrame(table_data)
    
    # Round all numbers to 4 decimal places 
    df = df.round(4)

    # Print the table neatly to the terminal
    print("                                               MODEL COMPARATIVE EVALUATION TABLE                          ")
    print(df.to_string(index=False))
    

    # Export the table to a CSV file so it can be easily copied to Word/Excel
    csv_filename = "comparative_evaluation_table.csv"
    df.to_csv(csv_filename, index=False)
    
    print(f"\n Table successfully saved as '{csv_filename}'!")
    

if __name__ == "__main__":
    generate_comparative_table()