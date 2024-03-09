from sklearn.model_selection import train_test_split
import pandas as pd

# Load the CSV file into a pandas DataFrame
file_path = 'data/chatgpt_paraphrases.csv'
df = pd.read_csv(file_path)

# Adjustable Size of Dataset
subset_size = 0.01

X = df['text']
y = df['paraphrases'].apply(eval)  # Convert the string representation of lists to actual lists

# Extract only the first paraphrase from the list
y_first_paraphrase = y.apply(lambda x: x[0] if len(x) > 0 else "")

X_subset, _, y_subset, _ = train_test_split(X, y_first_paraphrase, test_size=1-subset_size, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_subset, y_subset, test_size=0.2, random_state=42)

train_data = pd.DataFrame({'text': X_train, 'paraphrase': y_train})
test_data = pd.DataFrame({'text': X_test, 'paraphrase': y_test})

train_data.to_csv('data/train_data.csv', index=False)
test_data.to_csv('data/test_data.csv', index=False)

'''
print('Train data:')
print(len(train_data))
print('Test data:')
print(len(test_data))
'''