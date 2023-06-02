# Take-Home Technical Test: ECG Heartbeat Categorization

To reproduce my results, follow the steps:

1. Clone the project. Create an data directory with `mkdir data` and download ECG data from [Kaggle](https://www.kaggle.com/datasets/shayanfazeli/heartbeat) to it.

2. Run src/data/create\_dataset.py. This creates numpy arrays holding the training, validation and test data, and trims the sequences to a length of 160.

3. Run src/data/smote.py. This generates synthetic examples for the minority classes. It could take **hours** to run this script, due to DTW distance needing to run dynamic
programming to calculate and DTW needs to run for every sequence pairs.

4. Run src/data/balance.py. This creates a balanced training set, where each class has 3000 examples.

5. Run src/model/train.py with a single parameter `--lr lr`. This trains a model using learning rate `lr`. Early stopping is implemented.

6. Run src/model/eval.py. This calculates ROC AUC, accuracy and confusion matrix for the best model selected in the previous step.
