import pickle
from sklearn.neighbors import KNeighborsClassifier

# Load the saved model
with open('model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Check the type of the loaded object
if isinstance(loaded_model, KNeighborsClassifier):
    print("The loaded object is a KNeighborsClassifier.")
else:
    print("The loaded object is not a KNeighborsClassifier. It may be a different type of object.")
