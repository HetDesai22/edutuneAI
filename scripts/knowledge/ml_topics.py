"""Machine Learning topics."""

ML_TOPICS = [
    {
        "key": "supervised_learning",
        "name": "Supervised Learning",
        "short": "supervised learning",
        "domain": "Machine Learning",
        "question": "What is supervised learning?",
        "definition": (
            "Supervised learning is a type of machine learning in which a model learns a mapping "
            "from inputs X to outputs y using a labelled dataset, so that it can predict the "
            "label for new, unseen inputs."
        ),
        "simple": (
            "In supervised learning we teach the computer with examples that already have the "
            "right answers. For instance, we show it thousands of house descriptions together "
            "with their prices. It learns the relationship, and later it can guess the price of a "
            "house it has never seen. If the answer is a number, it is regression; if it is a "
            "category, it is classification."
        ),
        "eli10": (
            "It is like learning with flashcards. One side shows a picture of an animal, the "
            "other side says 'cat' or 'dog'. After practising with many cards, you can name "
            "animals on new cards you have never seen. The computer learns the same way, from "
            "examples that come with the right answer."
        ),
        "analogy": (
            "Supervised learning is like a student practising with a solved question bank. Every "
            "question comes with the correct answer, so after each attempt the student checks, "
            "notices mistakes and adjusts. The labels play the role of the teacher's answer key."
        ),
        "how_it_works": [
            "Collect a labelled dataset of (input, correct output) pairs.",
            "Split it into training, validation and test sets.",
            "Choose a model (linear regression, decision tree, neural network...) and a loss function.",
            "Train: adjust model parameters to minimise the loss between predictions and labels.",
            "Tune hyper-parameters on the validation set and report final performance on the test set.",
        ],
        "example": (
            "Email spam detection: inputs are emails, labels are 'spam' or 'not spam'. A logistic "
            "regression or naive Bayes model learns which word patterns indicate spam. On a test "
            "set of 1,000 unseen emails it might correctly classify 970, giving 97% accuracy."
        ),
        "applications": [
            "Spam filtering and sentiment analysis (classification)",
            "House price and sales forecasting (regression)",
            "Medical diagnosis from images or test results",
            "Credit scoring and fraud detection",
        ],
        "advantages": [
            "Clear objective and easy-to-measure performance",
            "Usually more accurate than unsupervised methods when labels exist",
            "Many mature algorithms and tools",
        ],
        "limitations": [
            "Needs labelled data, which can be expensive to create",
            "Label noise or bias is learned by the model",
            "May not generalise to data that differs from the training distribution",
        ],
        "key_points": [
            "Supervised learning learns from labelled (input, output) pairs.",
            "Two main tasks: regression (continuous output) and classification (discrete output).",
            "Training minimises a loss function such as MSE or cross-entropy.",
            "Data is split into train, validation and test sets to measure generalisation.",
            "Common algorithms: linear/logistic regression, decision trees, SVM, KNN, neural networks.",
        ],
        "viva": [
            ("What is labelled data?", "Data where every input comes with the correct output, for example an image tagged 'cat'."),
            ("What are the two main types of supervised learning?", "Regression, which predicts continuous values, and classification, which predicts discrete classes."),
            ("Why do we need a separate test set?", "To estimate how well the model performs on unseen data. Evaluating on training data would overestimate performance."),
            ("What is a loss function?", "A function that measures how far the model's predictions are from the true labels; training tries to minimise it."),
            ("Give two supervised learning algorithms.", "Linear regression and decision trees. Others include SVM, KNN, logistic regression and neural networks."),
        ],
        "interview": [
            ("How do you decide between a regression and a classification formulation?", "By the target variable. If it is a continuous quantity like price or temperature, it is regression. If it is a category like fraud/not fraud, it is classification. Sometimes a problem can be framed either way, for example predicting a rating as a number or as a class, and the choice depends on how the prediction will be used and evaluated."),
            ("What would you do if you have very few labels?", "Use transfer learning from a pre-trained model, data augmentation, semi-supervised or self-supervised learning, active learning to label the most informative examples, and simpler models with regularisation to avoid overfitting."),
        ],
        "mcqs": [
            {"q": "Predicting a house price from its size is an example of:", "options": ["Clustering", "Regression", "Dimensionality reduction", "Reinforcement learning"], "answer": "B", "explanation": "The output is a continuous number, which makes it a regression task."},
            {"q": "What does supervised learning require?", "options": ["Only unlabelled data", "Labelled training data", "A reward signal", "No data"], "answer": "B", "explanation": "Supervised learning needs inputs paired with correct outputs."},
        ],
    },
    {
        "key": "unsupervised_learning",
        "name": "Unsupervised Learning",
        "short": "unsupervised learning",
        "domain": "Machine Learning",
        "question": "What is unsupervised learning?",
        "definition": (
            "Unsupervised learning is a type of machine learning that finds structure in data "
            "without labels, for example grouping similar items (clustering), reducing the number "
            "of features (dimensionality reduction) or detecting unusual points (anomaly detection)."
        ),
        "simple": (
            "In unsupervised learning, we give the computer data but no correct answers. Its job "
            "is to discover patterns on its own. For example, a shop can give it customer "
            "purchase data, and it may discover groups such as 'budget shoppers' and 'luxury "
            "shoppers' even though nobody told it those groups exist."
        ),
        "eli10": (
            "Imagine you get a big box of mixed LEGO pieces and nobody tells you anything. You "
            "might sort them into piles by colour or by shape all by yourself. Unsupervised "
            "learning is the computer sorting things into groups without being told the answers."
        ),
        "analogy": (
            "Unsupervised learning is like a new librarian receiving thousands of unlabelled "
            "books. Without a catalogue, they start grouping books that share similar words and "
            "themes. Later they notice the groups correspond to 'history', 'science' and "
            "'cooking', but the grouping came first, without labels."
        ),
        "how_it_works": [
            "Collect unlabelled data and scale the features so distances are meaningful.",
            "Choose a goal: clustering, dimensionality reduction, density estimation or anomaly detection.",
            "Apply an algorithm such as K-Means, DBSCAN, hierarchical clustering, PCA or autoencoders.",
            "Evaluate with internal measures (e.g. silhouette score) and domain inspection.",
            "Interpret and use the discovered structure, e.g. for segmentation or as features.",
        ],
        "example": (
            "Customer segmentation: an online store has spending and visit-frequency data for "
            "10,000 customers. K-Means with k = 4 finds groups such as frequent low spenders, "
            "rare high spenders, loyal high spenders and inactive customers. Marketing can then "
            "design a campaign for each group."
        ),
        "applications": [
            "Customer segmentation",
            "Anomaly and fraud detection",
            "Dimensionality reduction for visualisation (PCA, t-SNE)",
            "Topic modelling of documents",
        ],
        "advantages": [
            "No need for expensive labels",
            "Can reveal hidden structure that humans did not expect",
            "Useful as a preprocessing step for supervised learning",
        ],
        "limitations": [
            "Hard to evaluate because there is no ground truth",
            "Results depend strongly on feature scaling and hyper-parameters like k",
            "Discovered groups may not be meaningful for the business problem",
        ],
        "key_points": [
            "Unsupervised learning works with unlabelled data.",
            "Main tasks: clustering, dimensionality reduction, anomaly detection, density estimation.",
            "Algorithms: K-Means, hierarchical clustering, DBSCAN, PCA, autoencoders.",
            "Evaluation uses internal metrics like silhouette score or inspection by experts.",
            "Self-supervised learning (used to pre-train LLMs) creates labels from the data itself.",
        ],
        "viva": [
            ("How does unsupervised learning differ from supervised learning?", "Supervised learning uses labelled examples to learn an input-output mapping; unsupervised learning has no labels and discovers structure in the data."),
            ("Name two unsupervised learning tasks.", "Clustering and dimensionality reduction. Anomaly detection is another."),
            ("What is PCA used for?", "Principal Component Analysis reduces the number of features by projecting data onto the directions of maximum variance."),
            ("How can you evaluate a clustering result?", "With internal metrics such as the silhouette score or Davies-Bouldin index, and by checking whether the clusters make sense to domain experts."),
        ],
        "interview": [
            ("Give a real business use case of unsupervised learning.", "Fraud detection in banking: most transactions are unlabelled, so we model normal behaviour with clustering or an isolation forest and flag transactions that are far from normal patterns. Analysts then review the flagged cases, which can also produce labels for a later supervised model."),
            ("What is the difference between unsupervised and self-supervised learning?", "Unsupervised learning finds structure without any targets. Self-supervised learning creates targets automatically from the data, such as predicting the next word or a masked word, and then trains like supervised learning. LLM pre-training is self-supervised."),
        ],
        "mcqs": [
            {"q": "Which of these is an unsupervised learning algorithm?", "options": ["Linear regression", "K-Means", "Logistic regression", "Decision tree classifier"], "answer": "B", "explanation": "K-Means groups unlabelled data into clusters."},
            {"q": "PCA is mainly used for:", "options": ["Classification", "Dimensionality reduction", "Reinforcement learning", "Text generation"], "answer": "B", "explanation": "PCA projects data onto fewer dimensions that keep most of the variance."},
        ],
    },
    {
        "key": "regression",
        "name": "Regression (Linear Regression)",
        "short": "regression",
        "domain": "Machine Learning",
        "question": "What is regression in machine learning?",
        "definition": (
            "Regression is a supervised learning task that predicts a continuous numerical value. "
            "Linear regression, the simplest form, models the target as a weighted sum of the "
            "input features plus a bias: y = w1*x1 + ... + wn*xn + b."
        ),
        "simple": (
            "Regression is used when the answer is a number, like a price, a temperature or a "
            "salary. Linear regression draws the best straight line through the data points. "
            "'Best' means the line with the smallest total squared distance between the "
            "predicted values and the real values."
        ),
        "eli10": (
            "Imagine you notice that the more hours you practise football, the more goals you "
            "score. If you draw your results on graph paper and put a ruler through the middle "
            "of the dots, you can guess how many goals you would score after practising for 10 "
            "hours. That ruler line is linear regression."
        ),
        "analogy": (
            "Linear regression is like adjusting a rubber band stretched across pins on a board. "
            "Each pin is a data point pulling on the band. The band settles where the total pull "
            "(the squared error) is smallest, which is the best-fit line."
        ),
        "how_it_works": [
            "Model: y_hat = w*x + b (or a weighted sum of many features).",
            "Loss: Mean Squared Error, MSE = (1/n) * sum((y - y_hat)^2).",
            "Find w and b that minimise MSE, either with the closed-form normal equation or with gradient descent.",
            "Evaluate using MSE, RMSE, MAE or R-squared on test data.",
            "Extensions: polynomial regression, Ridge (L2) and Lasso (L1) regularisation.",
        ],
        "example": (
            "Predicting house price from size: after training, the model is price = 3,000 * size "
            "+ 50,000. A 1,200 sq ft house is predicted at 3,000 * 1,200 + 50,000 = 3,650,000. "
            "The weight 3,000 means each extra square foot adds about 3,000 to the price."
        ),
        "applications": [
            "House price prediction",
            "Sales and demand forecasting",
            "Estimating crop yield from rainfall and fertiliser",
            "Predicting a student's marks from study hours",
        ],
        "advantages": [
            "Simple, fast and easy to interpret",
            "Coefficients show how each feature affects the prediction",
            "Works well when the relationship is roughly linear",
        ],
        "limitations": [
            "Cannot capture non-linear relationships without feature engineering",
            "Sensitive to outliers because errors are squared",
            "Assumes features are not strongly correlated (multicollinearity causes unstable weights)",
        ],
        "key_points": [
            "Regression predicts continuous values.",
            "Linear regression: y = wx + b, trained by minimising Mean Squared Error.",
            "Solved by the normal equation or gradient descent.",
            "Metrics: MSE, RMSE, MAE and R-squared.",
            "Ridge (L2) and Lasso (L1) add regularisation to reduce overfitting.",
        ],
        "viva": [
            ("What loss function does linear regression use?", "Mean Squared Error, the average of the squared differences between predicted and actual values."),
            ("What does R-squared measure?", "The proportion of variance in the target explained by the model; 1 means a perfect fit, 0 means no better than predicting the mean."),
            ("What is the difference between Lasso and Ridge regression?", "Ridge adds an L2 penalty that shrinks weights; Lasso adds an L1 penalty that can shrink some weights exactly to zero, performing feature selection."),
            ("Why is linear regression sensitive to outliers?", "Because errors are squared, a single point far from the line has a very large influence on the loss."),
        ],
        "interview": [
            ("What are the assumptions of linear regression?", "Linearity between features and target, independence of errors, homoscedasticity (constant error variance), normally distributed residuals for inference, and no strong multicollinearity among features."),
            ("How would you handle a non-linear relationship in a regression problem?", "Add polynomial or interaction features, transform variables (for example a log transform), or switch to a non-linear model such as a decision tree ensemble or a neural network, while checking validation error to avoid overfitting."),
        ],
        "mcqs": [
            {"q": "Which metric is commonly minimised when training linear regression?", "options": ["Cross-entropy", "Mean Squared Error", "Accuracy", "F1 score"], "answer": "B", "explanation": "Linear regression minimises the mean of squared residuals."},
            {"q": "Which regularisation can set some weights exactly to zero?", "options": ["L2 (Ridge)", "L1 (Lasso)", "Dropout", "Batch normalisation"], "answer": "B", "explanation": "The L1 penalty produces sparse solutions."},
        ],
        "numerical": {
            "problem": "A linear regression model is y = 2x + 1. For the data points (1, 4), (2, 5) and (3, 6), calculate the Mean Squared Error.",
            "solution": "Step 1: Predictions: x=1 -> 3, x=2 -> 5, x=3 -> 7.\nStep 2: Errors (y - y_hat): 4-3 = 1, 5-5 = 0, 6-7 = -1.\nStep 3: Squared errors: 1, 0, 1.\nStep 4: MSE = (1 + 0 + 1) / 3 = 0.667.\nAnswer: MSE is approximately 0.67.",
        },
    },
    {
        "key": "classification",
        "name": "Classification",
        "short": "classification",
        "domain": "Machine Learning",
        "question": "What is classification in machine learning?",
        "definition": (
            "Classification is a supervised learning task that assigns an input to one of a set "
            "of discrete classes, such as spam/not spam or cat/dog/bird. Models usually output a "
            "probability for each class and pick the most likely one."
        ),
        "simple": (
            "Classification is when the answer is a category rather than a number. Is this email "
            "spam or not? Is this tumour benign or malignant? The model learns from labelled "
            "examples and draws a boundary that separates the classes. Logistic regression, "
            "decision trees, SVMs and neural networks are common classifiers."
        ),
        "eli10": (
            "Think of sorting fruit into baskets: apples in one, bananas in another. You look at "
            "the colour, shape and size to decide. A classification model does the same: it "
            "looks at the features of something and decides which basket it belongs in."
        ),
        "analogy": (
            "A classifier is like a post office sorting machine. Each letter (input) is examined "
            "for clues such as the postcode, and dropped into the right bin (class). The machine "
            "learned the rules by seeing many letters that were already sorted correctly."
        ),
        "how_it_works": [
            "Collect labelled examples for each class.",
            "Train a model that outputs class scores or probabilities (e.g. sigmoid for binary, softmax for multi-class).",
            "Minimise cross-entropy loss between predicted probabilities and true labels.",
            "Predict the class with the highest probability, or apply a decision threshold.",
            "Evaluate with accuracy, precision, recall, F1 score and the confusion matrix.",
        ],
        "example": (
            "A disease test classifier is evaluated on 100 patients: TP = 40, FN = 10, FP = 5, "
            "TN = 45. Accuracy = (40+45)/100 = 85%. Precision = 40/(40+5) = 0.89. Recall = "
            "40/(40+10) = 0.80. For a disease screening test, recall matters most because "
            "missing a sick patient is costly."
        ),
        "applications": [
            "Spam and phishing detection",
            "Medical diagnosis",
            "Image recognition",
            "Sentiment analysis of reviews",
        ],
        "advantages": [
            "Directly answers yes/no or category questions",
            "Probabilistic outputs allow adjustable thresholds",
            "Wide choice of algorithms for different data types",
        ],
        "limitations": [
            "Accuracy is misleading on imbalanced datasets",
            "Needs enough labelled examples for every class",
            "Decision thresholds must be tuned to the cost of different errors",
        ],
        "key_points": [
            "Classification predicts discrete classes.",
            "Binary uses sigmoid; multi-class uses softmax; loss is cross-entropy.",
            "Confusion matrix: TP, FP, TN, FN.",
            "Precision = TP/(TP+FP); Recall = TP/(TP+FN); F1 is their harmonic mean.",
            "For imbalanced data prefer precision, recall, F1 or ROC-AUC over accuracy.",
        ],
        "viva": [
            ("What is a confusion matrix?", "A table comparing predicted and actual classes, showing true positives, false positives, true negatives and false negatives."),
            ("Define precision and recall.", "Precision = TP/(TP+FP), the fraction of predicted positives that are correct. Recall = TP/(TP+FN), the fraction of actual positives that are found."),
            ("Why is accuracy misleading for imbalanced data?", "If 99% of cases are negative, a model that always predicts negative has 99% accuracy but finds no positives."),
            ("Is logistic regression used for regression or classification?", "Classification. It outputs a probability using the sigmoid function and applies a threshold, usually 0.5."),
        ],
        "interview": [
            ("In fraud detection, would you optimise precision or recall?", "It depends on costs, but usually recall is prioritised because missing a fraud is expensive, while keeping precision acceptable so analysts are not flooded with false alarms. I would look at the precision-recall curve, choose a threshold based on business cost, and report F1 or PR-AUC."),
            ("How do you handle class imbalance?", "Use suitable metrics (precision, recall, F1, PR-AUC), resample the data (oversampling such as SMOTE, or undersampling), use class weights in the loss, tune the decision threshold, and collect more minority-class data if possible."),
        ],
        "mcqs": [
            {"q": "Which function converts scores into probabilities for multi-class classification?", "options": ["ReLU", "Softmax", "Tanh", "MSE"], "answer": "B", "explanation": "Softmax turns a vector of scores into probabilities that sum to 1."},
            {"q": "Recall is defined as:", "options": ["TP/(TP+FP)", "TP/(TP+FN)", "TN/(TN+FP)", "(TP+TN)/Total"], "answer": "B", "explanation": "Recall measures the fraction of actual positives that were correctly identified."},
        ],
        "numerical": {
            "problem": "A classifier gives TP = 30, FP = 10, FN = 20, TN = 40. Compute accuracy, precision, recall and F1 score.",
            "solution": "Step 1: Total = 30 + 10 + 20 + 40 = 100.\nStep 2: Accuracy = (TP + TN) / Total = (30 + 40) / 100 = 0.70.\nStep 3: Precision = TP / (TP + FP) = 30 / 40 = 0.75.\nStep 4: Recall = TP / (TP + FN) = 30 / 50 = 0.60.\nStep 5: F1 = 2 * P * R / (P + R) = 2 * 0.75 * 0.60 / 1.35 = 0.667.\nAnswer: Accuracy 70%, Precision 0.75, Recall 0.60, F1 about 0.67.",
        },
    },
    {
        "key": "clustering",
        "name": "Clustering (K-Means)",
        "short": "clustering",
        "domain": "Machine Learning",
        "question": "What is clustering?",
        "definition": (
            "Clustering is an unsupervised learning task that groups data points so that points "
            "in the same cluster are more similar to each other than to points in other "
            "clusters. K-Means is the most widely used clustering algorithm."
        ),
        "simple": (
            "Clustering puts similar things together without being told the groups in advance. "
            "K-Means does it by choosing k centre points, assigning each data point to its "
            "nearest centre, moving each centre to the average of its points, and repeating until "
            "nothing changes."
        ),
        "eli10": (
            "Imagine a playground where kids naturally gather in groups: some near the swings, "
            "some near the slide, some near the sandbox. If you had to find these groups from a "
            "photo, you would look at who is standing close together. Clustering is the computer "
            "finding those groups."
        ),
        "analogy": (
            "K-Means is like placing k ice-cream stalls on a beach. Each visitor walks to the "
            "nearest stall. Then each stall owner moves to the middle of their customers. People "
            "switch to whichever stall is now nearest, and the owners move again, until the "
            "stalls stop moving. Each stall's customers form a cluster."
        ),
        "how_it_works": [
            "Choose the number of clusters k and initialise k centroids (k-means++ is a good initialisation).",
            "Assignment step: assign each point to the nearest centroid, usually by Euclidean distance.",
            "Update step: move each centroid to the mean of the points assigned to it.",
            "Repeat assignment and update until assignments stop changing.",
            "Choose k using the elbow method or silhouette score.",
        ],
        "example": (
            "Points: (1,1), (1,2), (8,8), (9,8) with k = 2 and initial centroids (1,1) and (8,8). "
            "The first two points join cluster 1 and the last two join cluster 2. New centroids "
            "are (1, 1.5) and (8.5, 8). Assignments no longer change, so the algorithm stops."
        ),
        "applications": [
            "Customer segmentation",
            "Image compression by colour quantisation",
            "Grouping similar documents or news articles",
            "Anomaly detection (points far from every cluster)",
        ],
        "advantages": [
            "K-Means is simple, fast and scales to large datasets",
            "Easy to interpret using centroids",
            "Needs no labels",
        ],
        "limitations": [
            "k must be chosen in advance",
            "Sensitive to initialisation, outliers and feature scaling",
            "K-Means assumes roughly spherical, similar-sized clusters; DBSCAN handles arbitrary shapes",
        ],
        "key_points": [
            "Clustering groups similar unlabelled points.",
            "K-Means alternates assignment and centroid-update steps.",
            "It minimises within-cluster sum of squares (inertia).",
            "Choose k with the elbow method or silhouette score.",
            "Other methods: hierarchical clustering, DBSCAN, Gaussian mixture models.",
        ],
        "viva": [
            ("What does K represent in K-Means?", "The number of clusters the algorithm should form."),
            ("What is the elbow method?", "A way to choose k: plot inertia against k and pick the point where adding more clusters stops giving a big improvement (the 'elbow')."),
            ("Why should features be scaled before K-Means?", "Because it uses distances; a feature with a large range would dominate the distance calculation."),
            ("How is DBSCAN different from K-Means?", "DBSCAN groups dense regions, does not need k, can find arbitrarily shaped clusters and marks outliers as noise."),
        ],
        "interview": [
            ("K-Means gives different results each time you run it. Why, and how do you fix it?", "Because centroids are initialised randomly and K-Means converges to a local minimum. Use k-means++ initialisation, run several initialisations (n_init) and keep the solution with the lowest inertia, and fix the random seed for reproducibility."),
            ("How is the silhouette score computed?", "For each point, a is the mean distance to points in its own cluster and b is the mean distance to points in the nearest other cluster. The silhouette is (b - a) / max(a, b), ranging from -1 to 1; higher means better-separated clusters. The overall score is the average over points."),
        ],
        "mcqs": [
            {"q": "K-Means updates each centroid to:", "options": ["The median of all data", "The mean of the points assigned to it", "A random point", "The farthest point"], "answer": "B", "explanation": "The update step moves the centroid to the average of its cluster members."},
            {"q": "Which clustering algorithm does NOT need the number of clusters in advance?", "options": ["K-Means", "DBSCAN", "K-Medoids", "None"], "answer": "B", "explanation": "DBSCAN finds clusters based on density using eps and min_samples."},
        ],
        "numerical": {
            "problem": "In K-Means, a cluster contains the points (2, 3), (4, 5) and (6, 7). What is the new centroid after the update step?",
            "solution": "Step 1: Mean of x-coordinates = (2 + 4 + 6) / 3 = 4.\nStep 2: Mean of y-coordinates = (3 + 5 + 7) / 3 = 5.\nAnswer: The new centroid is (4, 5).",
        },
    },
    {
        "key": "decision_trees",
        "name": "Decision Trees",
        "short": "decision trees",
        "domain": "Machine Learning",
        "question": "What is a decision tree?",
        "definition": (
            "A decision tree is a supervised learning model that predicts a target by learning a "
            "sequence of if-then questions about the features. Internal nodes test a feature, "
            "branches are outcomes and leaves give the prediction."
        ),
        "simple": (
            "A decision tree works like the game '20 Questions'. It asks a question about the "
            "data, such as 'Is income greater than 50,000?', then follows the yes or no branch "
            "and asks the next question, until it reaches an answer. It chooses the questions "
            "that best separate the classes, using measures like Gini impurity or entropy."
        ),
        "eli10": (
            "Think of a flowchart to decide whether to go outside and play. Is it raining? If "
            "yes, stay in. If no, is it too hot? If yes, go to the pool; if no, play football. "
            "A decision tree is a flowchart like this that the computer builds by itself from "
            "examples."
        ),
        "analogy": (
            "A decision tree is like a doctor's diagnostic checklist. The doctor asks the most "
            "informative question first ('Do you have a fever?'), then narrows down with more "
            "specific questions until reaching a diagnosis. The tree learns which questions to "
            "ask, and in which order, from past patient records."
        ),
        "how_it_works": [
            "Start with all training data at the root node.",
            "For each feature and threshold, measure how well a split separates the targets (Gini impurity, entropy/information gain, or variance for regression).",
            "Choose the best split and create child nodes.",
            "Repeat recursively until a stopping rule applies (max depth, minimum samples, pure node).",
            "Prune the tree or limit depth to avoid overfitting.",
        ],
        "example": (
            "Loan approval: the tree first splits on 'credit score > 700'. If yes, it checks "
            "'income > 40,000' and approves if true. If the credit score is 700 or less, it "
            "checks 'existing debt < 10,000'. A new applicant with score 720 and income 55,000 "
            "follows the left branches and is approved."
        ),
        "applications": [
            "Loan and credit approval",
            "Medical decision support",
            "Customer churn prediction",
            "As building blocks for random forests and gradient boosting",
        ],
        "advantages": [
            "Easy to understand and visualise",
            "Handles numerical and categorical data with little preprocessing (no scaling needed)",
            "Captures non-linear relationships and feature interactions",
        ],
        "limitations": [
            "Deep trees overfit easily",
            "Unstable: small data changes can produce a very different tree",
            "Greedy splitting does not guarantee the globally best tree",
        ],
        "key_points": [
            "A decision tree is a flowchart of feature tests leading to predictions.",
            "Splits are chosen by Gini impurity or information gain (entropy reduction).",
            "Gini = 1 - sum(p_i^2); Entropy = -sum(p_i * log2 p_i).",
            "Control overfitting with max_depth, min_samples_leaf and pruning.",
            "Ensembles (random forest, gradient boosting) fix much of a single tree's instability.",
        ],
        "viva": [
            ("What is Gini impurity?", "A measure of how mixed the classes in a node are: Gini = 1 - sum of squared class probabilities. It is 0 for a pure node."),
            ("What is information gain?", "The reduction in entropy after splitting a node on a feature. The split with the highest information gain is chosen."),
            ("Why do decision trees overfit?", "Because they can keep splitting until every leaf is pure, memorising noise in the training data."),
            ("What is pruning?", "Removing branches that add little predictive power, to reduce overfitting and simplify the tree."),
        ],
        "interview": [
            ("Why don't decision trees require feature scaling?", "Each split compares one feature with a threshold, so only the ordering of values matters, not their scale. Multiplying a feature by 1000 just moves the threshold."),
            ("How would you prevent a decision tree from overfitting?", "Limit max_depth, set min_samples_split or min_samples_leaf, use cost-complexity pruning, and validate with cross-validation. Often the best fix is to use an ensemble like a random forest or gradient boosting."),
        ],
        "mcqs": [
            {"q": "The Gini impurity of a node containing only one class is:", "options": ["1", "0.5", "0", "Undefined"], "answer": "C", "explanation": "A pure node has Gini = 1 - 1^2 = 0."},
            {"q": "Which is a common way to reduce overfitting in decision trees?", "options": ["Increase depth", "Pruning", "Remove the root", "Use more features only"], "answer": "B", "explanation": "Pruning removes branches that do not improve generalisation."},
        ],
        "numerical": {
            "problem": "A node contains 6 positive and 4 negative examples. Calculate its Gini impurity and entropy.",
            "solution": "Step 1: p(+) = 6/10 = 0.6, p(-) = 4/10 = 0.4.\nStep 2: Gini = 1 - (0.6^2 + 0.4^2) = 1 - (0.36 + 0.16) = 0.48.\nStep 3: Entropy = -(0.6 log2 0.6 + 0.4 log2 0.4) = -(0.6 * -0.737 + 0.4 * -1.322) = 0.442 + 0.529 = 0.971.\nAnswer: Gini = 0.48, Entropy is about 0.971 bits.",
        },
    },
    {
        "key": "random_forest",
        "name": "Random Forest",
        "short": "random forests",
        "domain": "Machine Learning",
        "question": "What is a random forest?",
        "definition": (
            "A random forest is an ensemble of decision trees. Each tree is trained on a bootstrap "
            "sample of the data and considers a random subset of features at each split; the "
            "forest predicts by majority vote (classification) or averaging (regression)."
        ),
        "simple": (
            "One decision tree can make silly mistakes, but many different trees voting together "
            "are much more reliable. A random forest builds hundreds of trees, each trained on a "
            "slightly different random sample of the data and features, and combines their "
            "votes. This makes it more accurate and much less likely to overfit."
        ),
        "eli10": (
            "If you ask one friend to guess how many sweets are in a jar, they might be way off. "
            "If you ask 100 friends and take the average, the answer is usually very close. A "
            "random forest asks lots of decision trees and combines their answers."
        ),
        "analogy": (
            "A random forest is like a panel of doctors who each studied slightly different "
            "patient records and focus on different symptoms. Each gives a diagnosis, and the "
            "panel goes with the majority. Individual doctors make errors, but their errors "
            "differ, so the group decision is more reliable."
        ),
        "how_it_works": [
            "Draw a bootstrap sample (sampling with replacement) of the training data for each tree - this is bagging.",
            "Grow a decision tree on it; at every split consider only a random subset of features (e.g. sqrt of total features).",
            "Repeat to build many trees (n_estimators).",
            "Predict by majority vote for classification or the average for regression.",
            "Use out-of-bag samples (not drawn for a tree) to estimate accuracy without a separate validation set.",
        ],
        "example": (
            "Predicting customer churn with 20 features: a random forest of 300 trees, each using "
            "about 4 random features per split. For one customer, 210 trees vote 'will churn' and "
            "90 vote 'will stay', so the forest predicts churn with estimated probability 0.7. "
            "Feature importance shows 'monthly charges' and 'contract type' matter most."
        ),
        "applications": [
            "Credit risk and fraud detection",
            "Customer churn prediction",
            "Medical outcome prediction",
            "Feature importance analysis on tabular data",
        ],
        "advantages": [
            "High accuracy on tabular data with little tuning",
            "Reduces the overfitting of single trees (lower variance)",
            "Gives feature importance scores and out-of-bag error estimates",
        ],
        "limitations": [
            "Less interpretable than a single tree",
            "Larger memory and slower prediction with many trees",
            "Cannot extrapolate beyond the range of training targets in regression",
        ],
        "key_points": [
            "Random forest = bagging + random feature selection over many decision trees.",
            "Combining de-correlated trees reduces variance.",
            "Prediction: majority vote (classification) or mean (regression).",
            "Out-of-bag error gives a built-in validation estimate.",
            "Key hyper-parameters: n_estimators, max_depth, max_features.",
        ],
        "viva": [
            ("What is bagging?", "Bootstrap aggregating: training several models on random samples drawn with replacement and combining their predictions."),
            ("Why does a random forest select random features at each split?", "To make the trees less correlated; averaging de-correlated trees reduces variance more effectively."),
            ("What is out-of-bag error?", "The error measured on training samples that were not included in a tree's bootstrap sample, used as a built-in validation estimate."),
            ("Does random forest reduce bias or variance?", "Mainly variance. Each tree has low bias but high variance; averaging many trees lowers variance."),
        ],
        "interview": [
            ("What is the difference between random forest and gradient boosting?", "Random forest builds trees independently in parallel on bootstrap samples and averages them, mainly reducing variance. Gradient boosting builds shallow trees sequentially, where each tree corrects the errors of the previous ones, mainly reducing bias. Boosting is often more accurate but needs more careful tuning and is more prone to overfitting."),
            ("How is feature importance computed in a random forest?", "Impurity-based importance sums the decrease in Gini or entropy from splits on each feature across all trees. Permutation importance shuffles a feature and measures the drop in performance, which is more reliable when features have many unique values."),
        ],
        "mcqs": [
            {"q": "A random forest combines trees using:", "options": ["Gradient steps", "Majority vote or averaging", "Backpropagation", "Only the best tree"], "answer": "B", "explanation": "Classification uses a vote and regression uses the mean prediction."},
            {"q": "Random forest mainly reduces:", "options": ["Bias", "Variance", "Training data", "Number of features"], "answer": "B", "explanation": "Averaging many high-variance trees lowers the ensemble's variance."},
        ],
    },
    {
        "key": "svm",
        "name": "Support Vector Machine (SVM)",
        "short": "SVM",
        "domain": "Machine Learning",
        "question": "What is a Support Vector Machine (SVM)?",
        "definition": (
            "A Support Vector Machine is a supervised learning algorithm that finds the "
            "hyperplane separating classes with the maximum margin. The closest training points "
            "to the hyperplane are the support vectors. With kernels, SVMs can learn non-linear "
            "boundaries."
        ),
        "simple": (
            "An SVM tries to draw the widest possible 'street' between two classes. The street's "
            "middle line is the decision boundary, and the points touching the edges of the "
            "street are called support vectors. A wide street means the classifier is confident "
            "and generalises better. If the data cannot be separated by a straight line, the "
            "kernel trick lets SVM draw curved boundaries."
        ),
        "eli10": (
            "Imagine red and blue marbles on a table and you need to put a stick between them. "
            "You want the stick placed so it is as far as possible from the nearest red and "
            "nearest blue marble. That way, if a new marble rolls in, you are less likely to "
            "put it on the wrong side. That is what an SVM does."
        ),
        "analogy": (
            "SVM is like building a road between two villages that must stay as far from both as "
            "possible. Only the houses closest to the road (support vectors) determine where it "
            "goes; houses far away do not matter. The kernel trick is like lifting the map into "
            "3D so that villages that were tangled together can be separated by a flat sheet."
        ),
        "how_it_works": [
            "Represent each example as a point in feature space with label +1 or -1.",
            "Find the hyperplane w.x + b = 0 that maximises the margin 2/||w||.",
            "Soft margin: parameter C trades off margin width against misclassified training points.",
            "Kernel trick: replace dot products with a kernel K(x, x') (linear, polynomial, RBF) to get non-linear boundaries.",
            "Classify new points by the sign of w.x + b.",
        ],
        "example": (
            "Classifying emails as spam using TF-IDF features: text data is high-dimensional and "
            "often almost linearly separable, so a linear SVM works very well. For a dataset "
            "shaped like two concentric circles, a linear SVM fails but an RBF-kernel SVM finds "
            "a circular boundary."
        ),
        "applications": [
            "Text classification and spam detection",
            "Image classification with hand-crafted features",
            "Bioinformatics, e.g. protein classification",
            "Handwriting recognition",
        ],
        "advantages": [
            "Effective in high-dimensional spaces",
            "Maximum margin gives good generalisation",
            "Kernels allow flexible non-linear boundaries",
        ],
        "limitations": [
            "Training is slow on very large datasets (roughly quadratic or worse in samples)",
            "Sensitive to the choice of kernel, C and gamma, and to feature scaling",
            "Does not directly output probabilities",
        ],
        "key_points": [
            "SVM finds the maximum-margin separating hyperplane.",
            "Support vectors are the training points closest to the boundary.",
            "Margin width = 2/||w||.",
            "C controls the soft-margin trade-off; gamma controls RBF kernel width.",
            "The kernel trick computes similarities in a high-dimensional space without explicitly mapping to it.",
        ],
        "viva": [
            ("What are support vectors?", "The training points closest to the decision boundary. They alone determine the position of the hyperplane."),
            ("What is the kernel trick?", "Computing dot products in a high-dimensional feature space through a kernel function, without explicitly transforming the data, so linear methods can learn non-linear boundaries."),
            ("What does the C parameter do?", "It controls the penalty for misclassification. Large C gives a narrow margin with few training errors; small C allows a wider margin with more errors."),
            ("Name three common kernels.", "Linear, polynomial and radial basis function (RBF/Gaussian)."),
        ],
        "interview": [
            ("When would you choose an SVM over a neural network?", "When the dataset is small to medium sized and high-dimensional, like text features, where SVMs train reliably with strong generalisation and few hyper-parameters. For very large datasets or raw images, audio and text, neural networks usually win."),
            ("What happens if gamma is too high in an RBF SVM?", "Each training point's influence becomes very local, so the boundary wraps tightly around individual points and the model overfits. Too low gamma makes the boundary overly smooth and underfits. Tune gamma and C together with cross-validation."),
        ],
        "mcqs": [
            {"q": "SVM chooses the hyperplane that:", "options": ["Passes through the most points", "Maximises the margin", "Minimises the number of features", "Is always vertical"], "answer": "B", "explanation": "SVM is a maximum-margin classifier."},
            {"q": "Which technique allows SVMs to learn non-linear boundaries?", "options": ["Dropout", "Kernel trick", "Pooling", "Bagging"], "answer": "B", "explanation": "Kernels implicitly map data into a higher-dimensional space."},
        ],
        "numerical": {
            "problem": "A linear SVM has weight vector w = (3, 4). What is the width of its margin?",
            "solution": "Step 1: ||w|| = sqrt(3^2 + 4^2) = sqrt(25) = 5.\nStep 2: Margin width = 2 / ||w|| = 2 / 5 = 0.4.\nAnswer: The margin width is 0.4 units.",
        },
    },
    {
        "key": "knn",
        "name": "K-Nearest Neighbours (KNN)",
        "short": "KNN",
        "domain": "Machine Learning",
        "question": "What is the K-Nearest Neighbours (KNN) algorithm?",
        "definition": (
            "K-Nearest Neighbours is a simple, instance-based supervised learning algorithm. To "
            "predict a new point, it finds the k closest training points by a distance measure "
            "and takes a majority vote (classification) or average (regression) of their labels."
        ),
        "simple": (
            "KNN says 'you are like your neighbours'. To classify a new point, it looks at the k "
            "training points closest to it. If most of them are 'cat', it predicts 'cat'. There "
            "is no real training phase; the algorithm just stores the data and does the work at "
            "prediction time."
        ),
        "eli10": (
            "If you move to a new school and want to guess which sports club is most popular, "
            "you could ask the five kids sitting closest to you. If three say football, you guess "
            "football. KNN guesses the answer by asking the closest examples."
        ),
        "analogy": (
            "KNN is like judging a restaurant by looking at the ratings of the k most similar "
            "restaurants you already know: same cuisine, price and area. You do not build a "
            "general theory of restaurants; you just compare with your nearest known examples."
        ),
        "how_it_works": [
            "Store all training examples (lazy learning - no model is fitted).",
            "Scale the features so every feature contributes fairly to distance.",
            "For a new point, compute its distance (Euclidean, Manhattan...) to every training point.",
            "Select the k nearest points.",
            "Predict by majority vote (classification) or average (regression), optionally weighted by distance.",
        ],
        "example": (
            "Classify a fruit with weight 150 g and sweetness 7 when k = 3. The 3 nearest known "
            "fruits are 2 apples and 1 orange, so KNN predicts apple. With k = 1, only the single "
            "closest fruit decides, which is more sensitive to noise."
        ),
        "applications": [
            "Recommendation systems (similar users or items)",
            "Pattern recognition such as handwritten digits",
            "Missing-value imputation",
            "Simple baselines for classification tasks",
        ],
        "advantages": [
            "Very simple to understand and implement",
            "No training time",
            "Naturally handles multi-class problems and non-linear boundaries",
        ],
        "limitations": [
            "Slow prediction on large datasets because it compares with every stored point",
            "Requires feature scaling and suffers from the curse of dimensionality",
            "Sensitive to the choice of k and to noisy or irrelevant features",
        ],
        "key_points": [
            "KNN is a lazy, instance-based learner.",
            "Prediction uses the k closest training examples.",
            "Small k: low bias, high variance (overfits). Large k: smoother, may underfit.",
            "Always scale features; choose an odd k for binary classification to avoid ties.",
            "KD-trees or approximate nearest-neighbour search speed up prediction.",
        ],
        "viva": [
            ("Why is KNN called a lazy learner?", "Because it does not build a model during training; it stores the data and does all computation at prediction time."),
            ("What happens when k is very small?", "The prediction depends on very few points, so the model is sensitive to noise and tends to overfit."),
            ("Why is feature scaling important for KNN?", "KNN uses distances, so features with larger ranges would dominate if not scaled."),
            ("What is the curse of dimensionality?", "In high dimensions, points become far apart and distances become less meaningful, which hurts distance-based methods like KNN."),
        ],
        "interview": [
            ("How do you choose k in KNN?", "Try a range of values using cross-validation and pick the k with the best validation score. A common starting point is around the square root of the number of samples, and an odd k avoids ties in binary classification."),
            ("How would you make KNN fast for a million data points?", "Use spatial indexes like KD-trees or ball trees for low dimensions, and approximate nearest-neighbour libraries such as FAISS or HNSW for high dimensions. Reducing dimensionality with PCA also helps."),
        ],
        "mcqs": [
            {"q": "KNN is best described as:", "options": ["Eager learner", "Lazy, instance-based learner", "Generative model", "Reinforcement learning method"], "answer": "B", "explanation": "KNN stores training data and defers computation until prediction."},
            {"q": "Increasing k in KNN generally:", "options": ["Makes the boundary smoother", "Makes the model overfit more", "Removes the need for scaling", "Speeds up training"], "answer": "A", "explanation": "More neighbours average out noise, giving a smoother decision boundary."},
        ],
        "numerical": {
            "problem": "Using KNN with k = 3, classify the point P = (2, 3). Training points: A = (1, 3) class Red, B = (2, 5) class Blue, C = (3, 3) class Red, D = (6, 6) class Blue. Use Euclidean distance.",
            "solution": "Step 1: d(P, A) = sqrt((2-1)^2 + (3-3)^2) = 1.\nStep 2: d(P, B) = sqrt(0^2 + 2^2) = 2.\nStep 3: d(P, C) = sqrt(1^2 + 0^2) = 1.\nStep 4: d(P, D) = sqrt(4^2 + 3^2) = 5.\nStep 5: The 3 nearest are A (Red), C (Red) and B (Blue).\nAnswer: Majority vote is Red, so P is classified as Red.",
        },
    },
    {
        "key": "gradient_descent",
        "name": "Gradient Descent",
        "short": "gradient descent",
        "domain": "Machine Learning",
        "question": "What is gradient descent?",
        "definition": (
            "Gradient descent is an iterative optimisation algorithm that minimises a loss "
            "function by repeatedly moving the parameters in the direction of the negative "
            "gradient: theta = theta - learning_rate * gradient of the loss."
        ),
        "simple": (
            "Gradient descent is how most machine learning models learn. The loss tells us how "
            "wrong the model is. The gradient tells us which direction makes the loss increase "
            "fastest, so we take a small step in the opposite direction. We repeat this many "
            "times until the loss stops decreasing. The step size is called the learning rate."
        ),
        "eli10": (
            "Imagine you are standing on a foggy hill and want to reach the bottom. You cannot "
            "see far, but you can feel which way the ground slopes under your feet. So you take "
            "a small step downhill, feel again, and take another step. Keep doing this and you "
            "reach the valley. That is gradient descent."
        ),
        "analogy": (
            "Gradient descent is like adjusting the volume knob of a radio to remove static when "
            "you cannot see the dial. You turn it slightly, listen whether it got better or worse, "
            "and keep turning in the direction that improves the sound. Turning too fast (large "
            "learning rate) makes you overshoot; too slowly takes forever."
        ),
        "how_it_works": [
            "Initialise the parameters (weights) randomly or to zero.",
            "Compute the loss on the data (or a mini-batch).",
            "Compute the gradient of the loss with respect to each parameter.",
            "Update: w = w - learning_rate * gradient.",
            "Repeat for many iterations/epochs until convergence.",
        ],
        "example": (
            "Minimise f(w) = (w - 3)^2 starting at w = 0 with learning rate 0.1. The gradient is "
            "2(w - 3) = -6, so w becomes 0 - 0.1 * (-6) = 0.6. Next the gradient is -4.8 and w "
            "becomes 1.08. Each step moves w closer to the minimum at w = 3."
        ),
        "applications": [
            "Training linear and logistic regression",
            "Training neural networks (together with backpropagation)",
            "Fine-tuning large language models",
            "Matrix factorisation for recommender systems",
        ],
        "advantages": [
            "Simple and general: works for any differentiable loss",
            "Scales to millions or billions of parameters with mini-batches",
            "Many improved variants (Momentum, RMSProp, Adam)",
        ],
        "limitations": [
            "Choosing the learning rate is critical",
            "Can get stuck in local minima or saddle points for non-convex losses",
            "Slow convergence on badly scaled or ill-conditioned problems",
        ],
        "key_points": [
            "Update rule: w = w - learning_rate * dLoss/dw.",
            "Batch GD uses all data; stochastic GD uses one example; mini-batch GD uses small batches.",
            "Too large a learning rate diverges; too small converges slowly.",
            "Adam combines momentum and adaptive per-parameter learning rates.",
            "Learning-rate schedules (warm-up, cosine decay) improve training.",
        ],
        "viva": [
            ("What is the learning rate?", "A hyper-parameter that controls the size of each parameter update in gradient descent."),
            ("What is the difference between batch, stochastic and mini-batch gradient descent?", "Batch GD computes the gradient on the whole dataset, SGD on a single example and mini-batch GD on a small subset. Mini-batch is the standard compromise between stability and speed."),
            ("What happens if the learning rate is too high?", "Updates overshoot the minimum; the loss oscillates or diverges."),
            ("What is momentum?", "An extension that accumulates a moving average of past gradients, which speeds up movement in consistent directions and damps oscillations."),
        ],
        "interview": [
            ("Why is Adam so widely used?", "Adam keeps a moving average of gradients (momentum) and of squared gradients, and uses them to give each parameter its own adaptive step size. It works well with little tuning, handles sparse and noisy gradients, and is the default for training transformers. SGD with momentum can generalise slightly better in some vision tasks."),
            ("Your training loss is going up instead of down. What would you check?", "First the learning rate, which is the most common cause, so I would lower it or add warm-up. Then check for bugs: the sign of the update, labels matching inputs, loss computation, data normalisation, and exploding gradients, for which gradient clipping helps."),
        ],
        "mcqs": [
            {"q": "In gradient descent, parameters are updated in the direction of:", "options": ["The gradient", "The negative gradient", "A random direction", "The largest weight"], "answer": "B", "explanation": "Moving against the gradient decreases the loss."},
            {"q": "If the learning rate is too large, the loss will most likely:", "options": ["Decrease smoothly", "Oscillate or diverge", "Become exactly zero", "Stay constant"], "answer": "B", "explanation": "Large steps overshoot the minimum."},
            {"q": "Which variant uses a small subset of data per update?", "options": ["Batch GD", "Mini-batch GD", "Newton's method", "Grid search"], "answer": "B", "explanation": "Mini-batch gradient descent uses small batches such as 32 or 64 examples."},
        ],
        "numerical": {
            "problem": "Minimise the loss L(w) = w^2 using gradient descent with learning rate 0.1, starting at w = 5. Compute w after two updates.",
            "solution": "Step 1: The gradient is dL/dw = 2w.\nStep 2: Update 1: gradient = 2 * 5 = 10, so w = 5 - 0.1 * 10 = 4.\nStep 3: Update 2: gradient = 2 * 4 = 8, so w = 4 - 0.1 * 8 = 3.2.\nAnswer: After two updates w = 3.2, moving towards the minimum at w = 0.",
        },
        "troubleshooting": {
            "problem": "My model's training loss becomes NaN after a few iterations. What is wrong?",
            "solution": "A NaN loss usually means the optimisation is numerically unstable. Check these in order:\n1. Learning rate too high - reduce it by 10x or add a warm-up schedule.\n2. Exploding gradients - add gradient clipping (e.g. max norm 1.0).\n3. Invalid operations - log(0) or division by zero in a custom loss; add a small epsilon.\n4. Bad input data - check for NaN or infinite values and normalise features.\n5. Mixed precision overflow - fp16 can overflow; use bf16 or loss scaling.",
        },
    },
    {
        "key": "overfitting",
        "name": "Overfitting",
        "short": "overfitting",
        "domain": "Machine Learning",
        "question": "What is overfitting?",
        "definition": (
            "Overfitting occurs when a model learns the training data too closely, including its "
            "noise, so it performs very well on training data but poorly on new, unseen data. It "
            "is a sign of high variance."
        ),
        "simple": (
            "Overfitting is when a model memorises instead of learning. It gets almost perfect "
            "scores on the data it was trained on, but it fails on new data because it learned "
            "details and noise that do not generalise. You can spot it when training accuracy "
            "keeps improving while validation accuracy gets worse."
        ),
        "eli10": (
            "Imagine a student who memorises the exact answers to last year's exam instead of "
            "understanding the lessons. If this year's questions are slightly different, the "
            "student is stuck. A model that overfits is like that student: great at the old "
            "questions, bad at new ones."
        ),
        "analogy": (
            "Overfitting is like a tailor who makes a suit that fits one customer so perfectly it "
            "even follows the wrinkles in their shirt on that particular day. It will not fit "
            "anyone else, and it will not even fit the same customer tomorrow."
        ),
        "how_it_works": [
            "The model has too much capacity (too many parameters or too deep) relative to the data.",
            "During training, it reduces training loss by fitting noise and quirks of individual examples.",
            "Validation loss stops improving and starts rising while training loss keeps falling.",
            "Detect it by comparing training and validation performance (learning curves).",
            "Fix it with regularisation, more data, simpler models, early stopping, dropout or augmentation.",
        ],
        "example": (
            "A degree-15 polynomial fitted to 10 noisy points passes through every point "
            "(training error close to 0) but wiggles wildly between them, so its test error is "
            "huge. A straight line or degree-2 curve has a small training error but predicts new "
            "points much better."
        ),
        "applications": [
            "Diagnosing model performance through learning curves",
            "Choosing model complexity and regularisation strength",
            "Deciding when to stop training (early stopping)",
            "Fine-tuning LLMs on small datasets, where overfitting is a real risk",
        ],
        "advantages": [
            "Understanding overfitting guides model selection",
            "Clear diagnostic: training vs validation gap",
            "Well-known, effective remedies exist",
        ],
        "limitations": [
            "An overfit model gives falsely optimistic training results",
            "It generalises poorly and can fail in production",
            "Removing it completely may cause underfitting; a balance is needed",
        ],
        "key_points": [
            "Overfitting = low training error, high validation/test error (high variance).",
            "Causes: model too complex, too little data, training too long, noisy features.",
            "Remedies: more data, augmentation, L1/L2 regularisation, dropout, early stopping, simpler model.",
            "Cross-validation gives a reliable estimate of generalisation.",
            "It is one side of the bias-variance trade-off; underfitting is the other.",
        ],
        "viva": [
            ("How can you detect overfitting?", "By comparing training and validation performance: a large gap, with high training accuracy and much lower validation accuracy, indicates overfitting."),
            ("Give two methods to reduce overfitting.", "Regularisation such as L2 weight decay or dropout, and getting more training data or using data augmentation. Early stopping is another."),
            ("What is early stopping?", "Stopping training when validation loss stops improving, before the model starts to memorise training data."),
            ("How does dropout reduce overfitting?", "It randomly switches off neurons during training, so the network cannot rely on any single neuron and learns more robust features."),
        ],
        "interview": [
            ("Your model has 99% training accuracy and 70% validation accuracy. What do you do?", "This is overfitting. I would first check for data leakage or a distribution mismatch between the splits. Then I would reduce model complexity, add regularisation such as weight decay or dropout, use data augmentation or collect more data, apply early stopping, and use cross-validation to confirm the improvement."),
            ("Explain the bias-variance trade-off.", "Bias is error from overly simple assumptions, which causes underfitting. Variance is error from sensitivity to the particular training set, which causes overfitting. Increasing model complexity lowers bias but raises variance, so we pick the complexity that minimises total validation error."),
        ],
        "mcqs": [
            {"q": "A model with very low training error but high test error is:", "options": ["Underfitting", "Overfitting", "Perfectly fitted", "Not trained"], "answer": "B", "explanation": "The gap between training and test error is the hallmark of overfitting."},
            {"q": "Which technique does NOT help reduce overfitting?", "options": ["Dropout", "Early stopping", "Adding many more parameters", "Data augmentation"], "answer": "C", "explanation": "More parameters increase capacity and usually make overfitting worse."},
        ],
        "troubleshooting": {
            "problem": "My training accuracy is 99% but my validation accuracy is only 70%. How do I fix it?",
            "solution": "This large gap means the model is overfitting. Try these steps:\n1. Check for data problems first - leakage, duplicates across splits, or different distributions.\n2. Add regularisation - weight decay (L2) and dropout.\n3. Use data augmentation or collect more training data.\n4. Reduce model complexity - fewer layers, units or tree depth.\n5. Use early stopping based on validation loss.\n6. Confirm the improvement with cross-validation rather than a single split.",
        },
    },
    {
        "key": "underfitting",
        "name": "Underfitting",
        "short": "underfitting",
        "domain": "Machine Learning",
        "question": "What is underfitting?",
        "definition": (
            "Underfitting occurs when a model is too simple to capture the underlying pattern in "
            "the data, so it performs poorly on both the training data and new data. It is a "
            "sign of high bias."
        ),
        "simple": (
            "Underfitting is when a model has not learned enough. It makes big errors even on "
            "the training data, because it is too simple or was not trained long enough. For "
            "example, trying to fit a straight line to data that clearly follows a curve."
        ),
        "eli10": (
            "Imagine trying to draw a picture of a cat using only one straight line. No matter "
            "how you place the line, it will not look like a cat. The drawing tool is too simple. "
            "That is underfitting: the model is too simple to show the real pattern."
        ),
        "analogy": (
            "Underfitting is like summarising an entire novel as 'some things happened'. It is "
            "never exactly wrong, but it misses all the important details, so it is useless for "
            "answering questions about the story."
        ),
        "how_it_works": [
            "The model's capacity is too low or its features are not informative enough.",
            "Both training and validation errors stay high and close to each other.",
            "It may also happen if training stops too early or regularisation is too strong.",
            "Detect it with learning curves: both curves plateau at a high error.",
            "Fix it with a more complex model, better features, longer training or weaker regularisation.",
        ],
        "example": (
            "Predicting salary from years of experience when salary actually grows quickly at "
            "first and then levels off. A straight line misses the curve and gives training R^2 "
            "= 0.55 and test R^2 = 0.53. A polynomial or tree-based model captures the curve and "
            "raises both scores."
        ),
        "applications": [
            "Diagnosing learning curves",
            "Deciding to increase model capacity or improve features",
            "Tuning regularisation strength",
            "Checking whether training ran long enough",
        ],
        "advantages": [
            "Easy to detect: training error itself is high",
            "Usually easy to fix by adding capacity or features",
            "Underfit models are simple and stable, which can be useful as baselines",
        ],
        "limitations": [
            "Poor accuracy everywhere",
            "Misses important relationships in the data",
            "Adding too much capacity to fix it can lead to overfitting",
        ],
        "key_points": [
            "Underfitting = high training error and high test error (high bias).",
            "Causes: model too simple, poor features, too much regularisation, too little training.",
            "Remedies: more complex model, feature engineering, less regularisation, longer training.",
            "Opposite of overfitting in the bias-variance trade-off.",
            "Learning curves show both errors high and close together.",
        ],
        "viva": [
            ("How is underfitting different from overfitting?", "Underfitting means the model performs poorly even on training data (high bias); overfitting means it performs well on training data but poorly on new data (high variance)."),
            ("Give two causes of underfitting.", "A model that is too simple for the data, and excessive regularisation. Training for too few epochs is another cause."),
            ("How can you fix underfitting?", "Use a more complex model, add better features, reduce regularisation or train for longer."),
        ],
        "interview": [
            ("Both your training and validation accuracy are around 60%. What is happening and what do you do?", "Both are low and close, so the model is underfitting. I would increase capacity with a deeper network or a non-linear model, engineer more informative features, reduce regularisation, train longer with a well-tuned learning rate, and verify the labels are correct, since noisy labels also cap performance."),
        ],
        "mcqs": [
            {"q": "Underfitting is associated with:", "options": ["High variance", "High bias", "Too much data", "Too many parameters"], "answer": "B", "explanation": "An underfit model makes strong, overly simple assumptions."},
            {"q": "Which change can reduce underfitting?", "options": ["Increase regularisation", "Use a more complex model", "Train for fewer epochs", "Remove informative features"], "answer": "B", "explanation": "More capacity lets the model capture the pattern."},
        ],
        "troubleshooting": {
            "problem": "My model has high error on both the training and test sets. What should I do?",
            "solution": "High error on both sets means the model is underfitting (high bias). Try:\n1. A more expressive model - more layers or units, or a non-linear algorithm.\n2. Better features - add interactions, polynomial terms or domain features.\n3. Less regularisation - lower weight decay or dropout.\n4. Longer training with a properly tuned learning rate.\n5. Check the data - noisy or wrong labels limit what any model can learn.",
        },
    },
]
