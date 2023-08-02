# Step 1: Euclidean Distance Function
def euclidean_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

# Step 2: Finding k Nearest Neighbors
def get_k_nearest_neighbors(k, train_data, test_point):
    distances = [(euclidean_distance(point, test_point), label) for point, label in train_data]
    distances.sort()  # Sort the distances from closest to farthest
    return distances[:k]

# Step 3: Majority Voting and Classification
def classify(k_nearest_neighbors):
    votes = {}  # Dictionary to hold the votes for each label
    for _, label in k_nearest_neighbors:
        if label in votes:
            votes[label] += 1
        else:
            votes[label] = 1
    return max(votes, key=votes.get)

# Step 4: K-NN Prediction
def knn(train_data, test_data, k):
    predictions = []
    for test_point in test_data:
        k_nearest_neighbors = get_k_nearest_neighbors(k, train_data, test_point)
        predicted_label = classify(k_nearest_neighbors)
        predictions.append(predicted_label)
    return predictions

# Toy Dataset
train_data = [((1, 2), "A"), ((2, 3), "A"), ((5, 1), "B"), ((4, 3), "B")]
test_data = [(3, 2), (6, 2)]

# Set k (number of neighbors to consider)
k = 3

# Perform k-NN prediction
predictions = knn(train_data, test_data, k)

# Print the test points and their corresponding predictions
print("Test Point:     Predicted Label:")
print("-----------------------------")
for i in range(len(test_data)):
    test_point = test_data[i]
    predicted_label = predictions[i]
    print(f"{test_point}        {predicted_label}")
