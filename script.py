# 1.  Import necessary libraries and data set

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt
import numpy as np

#Load breast cancer data into a new variable
breast_cancer_data = load_breast_cancer()

# 2.  Taking a look at the data.

print(breast_cancer_data.data[0])
#What do these numbers mean? Print names to see.
print(breast_cancer_data.feature_names)


# 3.  See what we have to classify - print target and target names

print(breast_cancer_data.target)
print(breast_cancer_data.target_names)


# 4, 5, and 6.  Import train_test_split (done above), call function - data, target, size and state

# Give option to change random_state.  This is a preview to part 18 - leave variable do_random_state False for now, then experiment to see how different values affect outcomes.

do_random_state=False 
if do_random_state:
    randState=random.randint()
else:
    randState=100
    
training_data, validation_data, training_labels, validation_labels = train_test_split(breast_cancer_data.data, breast_cancer_data.target, train_size = 0.8, random_state = randState)

# 7.  Print lengths of training data and labels, should be equal if everything is working properly

print(len(training_data))
print(len(training_labels))

# 8 and 9.  Import KNeighborsClassifier (above) and create a KNeighborsClassifier

classifier = KNeighborsClassifier(3)

# 10.  Train with fit function

classifier.fit(training_data, training_labels)

# 11.  Call score function to determine accuracy of model on validation set

classifier.score(validation_data, validation_labels)

# 12.  Not bad.  Put the past lines into a for loop to find the best value for k
accuracies=[]
k_list=[]
for k in range(1, 101):    
    classifier = KNeighborsClassifier(k)
    classifier.fit(training_data, training_labels)
    score = classifier.score(validation_data, validation_labels)
    print(k, score)
    k_list.append(k)
    accuracies.append(score)

#print best k value
print("Best k value and matching score:")
print(k_list[accuracies.index(max(accuracies))], max(accuracies))

# 13-17.  Import matplotlib (at the top) and store the k values into k_list (done above) and the scores in accuracies (also above), plot these and set appropriate x and y labels and plot title.
plt.plot(k_list, accuracies)

plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title("Breast Cancer Classifier Accuracy")
plt.show()

# 18. Change do_random_state to "True" to try different states
