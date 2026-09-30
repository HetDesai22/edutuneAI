"""Side-by-side comparisons between related topics (category: comparison)."""

COMPARISONS = [
    {
        "a": "RNN", "b": "LSTM",
        "rows": [
            ("Memory", "Single hidden state", "Hidden state + separate cell state"),
            ("Gates", "None", "Forget, input and output gates"),
            ("Long-term dependencies", "Poor (vanishing gradients)", "Good (additive cell-state updates)"),
            ("Parameters", "Fewer", "About 4x more for the same hidden size"),
            ("Training speed", "Faster per step", "Slower per step"),
        ],
        "summary": "Use an LSTM when the sequence has long-range dependencies; a vanilla RNN is only suitable for short sequences or as a teaching baseline.",
    },
    {
        "a": "LSTM", "b": "GRU",
        "rows": [
            ("Gates", "3 (forget, input, output)", "2 (update, reset)"),
            ("States", "Cell state + hidden state", "Hidden state only"),
            ("Parameters", "4 gated components", "3 gated components (about 25% fewer)"),
            ("Speed", "Slower", "Faster"),
            ("Accuracy", "Slightly better on some very long sequences", "Comparable on many tasks"),
        ],
        "summary": "GRU is a lighter, faster alternative that often performs as well as an LSTM; try both and compare validation results.",
    },
    {
        "a": "CNN", "b": "RNN",
        "rows": [
            ("Best for", "Grid data such as images", "Sequential data such as text, speech and time series"),
            ("Core operation", "Convolution with shared filters over space", "Recurrence with shared weights over time"),
            ("Context", "Local receptive fields that grow with depth", "Hidden state carries past context"),
            ("Parallelism", "Highly parallel", "Sequential over time steps"),
            ("Typical example", "Image classification", "Next-word prediction"),
        ],
        "summary": "CNNs exploit spatial structure; RNNs exploit temporal order. Both share weights, but along different dimensions.",
    },
    {
        "a": "RNN", "b": "Transformer",
        "rows": [
            ("Processing", "One token at a time", "All tokens in parallel"),
            ("Long-range dependencies", "Hard (vanishing gradients)", "Direct via self-attention"),
            ("Training speed on GPUs", "Slow", "Fast"),
            ("Cost for long sequences", "Linear in length", "Quadratic in length (attention)"),
            ("Used in", "Older NLP systems, small time-series models", "Modern LLMs such as GPT, Llama and Qwen"),
        ],
        "summary": "Transformers replaced RNNs for most language tasks because attention captures long-range context and trains in parallel.",
    },
    {
        "a": "GAN", "b": "VAE",
        "rows": [
            ("Training", "Adversarial game (generator vs discriminator)", "Maximise ELBO (reconstruction + KL)"),
            ("Stability", "Unstable, can mode-collapse", "Stable"),
            ("Sample quality", "Sharp", "Often blurry"),
            ("Encoder", "No", "Yes, maps data to a latent distribution"),
            ("Latent space", "Less structured", "Smooth and continuous"),
        ],
        "summary": "Choose a GAN for sharp samples and a VAE for stable training and a meaningful latent space.",
    },
    {
        "a": "supervised learning", "b": "unsupervised learning",
        "rows": [
            ("Data", "Labelled (input, output) pairs", "Unlabelled inputs only"),
            ("Goal", "Predict outputs for new inputs", "Discover structure in data"),
            ("Tasks", "Regression, classification", "Clustering, dimensionality reduction"),
            ("Evaluation", "Clear metrics against labels", "Harder; internal metrics or expert review"),
            ("Example", "Spam detection", "Customer segmentation"),
        ],
        "summary": "Use supervised learning when you have labels and a clear target; unsupervised learning when you want to explore unlabelled data.",
    },
    {
        "a": "regression", "b": "classification",
        "rows": [
            ("Output", "Continuous number", "Discrete class"),
            ("Example", "Predicting house price", "Predicting spam / not spam"),
            ("Typical loss", "Mean Squared Error", "Cross-entropy"),
            ("Metrics", "MSE, RMSE, MAE, R-squared", "Accuracy, precision, recall, F1"),
            ("Example algorithm", "Linear regression", "Logistic regression"),
        ],
        "summary": "The type of the target variable decides: numbers mean regression, categories mean classification.",
    },
    {
        "a": "overfitting", "b": "underfitting",
        "rows": [
            ("Training error", "Very low", "High"),
            ("Validation/test error", "High", "High"),
            ("Bias / variance", "Low bias, high variance", "High bias, low variance"),
            ("Cause", "Model too complex or too little data", "Model too simple or too little training"),
            ("Fix", "Regularisation, more data, early stopping", "More capacity, better features, longer training"),
        ],
        "summary": "Aim for the sweet spot between the two, where validation error is lowest.",
    },
    {
        "a": "decision trees", "b": "random forests",
        "rows": [
            ("Model", "A single tree", "An ensemble of many trees"),
            ("Overfitting", "Prone to overfitting", "Much more robust"),
            ("Interpretability", "Very easy to interpret", "Harder; use feature importance"),
            ("Accuracy", "Moderate", "Usually higher"),
            ("Training cost", "Low", "Higher (but parallelisable)"),
        ],
        "summary": "Use a single tree when explanation matters most; use a random forest when accuracy and robustness matter.",
    },
    {
        "a": "KNN", "b": "K-Means",
        "rows": [
            ("Learning type", "Supervised", "Unsupervised"),
            ("Purpose", "Classify or regress a new point", "Group data into k clusters"),
            ("Meaning of k", "Number of neighbours", "Number of clusters"),
            ("Training", "None (lazy learner)", "Iterative centroid updates"),
            ("Needs labels", "Yes", "No"),
        ],
        "summary": "They are often confused because both use k and distances, but KNN predicts labels while K-Means discovers groups.",
    },
    {
        "a": "LoRA", "b": "QLoRA",
        "rows": [
            ("Base model precision", "16-bit or 32-bit (frozen)", "4-bit NF4 (frozen)"),
            ("Trainable part", "Low-rank adapters", "Low-rank adapters (in bf16/fp16)"),
            ("GPU memory", "Lower than full fine-tuning", "Much lower (about 4x smaller base weights)"),
            ("Speed", "Faster per step", "Slightly slower (dequantisation)"),
            ("Extra techniques", "-", "NF4, double quantisation, paged optimisers"),
        ],
        "summary": "QLoRA is LoRA on a 4-bit quantised base model; use it when GPU memory is the bottleneck.",
    },
    {
        "a": "full fine-tuning", "b": "PEFT",
        "rows": [
            ("Parameters updated", "All", "A small fraction (often under 2%)"),
            ("GPU memory", "Very high (weights + gradients + optimiser states)", "Low"),
            ("Checkpoint size", "Full model (GBs)", "Adapter only (MBs)"),
            ("Catastrophic forgetting", "Higher risk", "Lower risk (base frozen)"),
            ("Performance", "Best possible adaptation", "Close to full fine-tuning on most tasks"),
        ],
        "summary": "PEFT gives most of the benefit of full fine-tuning at a fraction of the cost, which is why it is the default for adapting LLMs.",
    },
    {
        "a": "RLHF", "b": "DPO",
        "rows": [
            ("Data", "Human preference comparisons", "Preference pairs (chosen, rejected)"),
            ("Reward model", "Trained explicitly", "Not needed (implicit)"),
            ("Optimisation", "Reinforcement learning (PPO)", "Supervised-style loss"),
            ("Complexity", "High, can be unstable", "Simple and stable"),
            ("Compute cost", "High", "Lower"),
        ],
        "summary": "Both align models with human preferences; DPO achieves it with a much simpler training pipeline.",
    },
    {
        "a": "prompt engineering", "b": "fine-tuning",
        "rows": [
            ("Changes model weights", "No", "Yes (all or a few via PEFT)"),
            ("Needs training data", "No (a few examples in the prompt)", "Yes"),
            ("Cost", "Very low", "Moderate (low with QLoRA)"),
            ("Consistency", "Can be brittle", "More consistent behaviour"),
            ("Best for", "Quick prototyping", "Specialised style, format or domain at scale"),
        ],
        "summary": "Start with prompt engineering; fine-tune when you need reliable, specialised behaviour. This project demonstrates the fine-tuning route.",
    },
    {
        "a": "instruction tuning", "b": "pre-training",
        "rows": [
            ("Data", "Curated instruction-response pairs", "Huge amounts of raw text"),
            ("Objective", "Generate the desired response", "Predict the next token"),
            ("Dataset size", "Hundreds to millions of examples", "Trillions of tokens"),
            ("What it teaches", "How to follow instructions and respond", "Language, facts and general knowledge"),
            ("Compute", "Small (especially with QLoRA)", "Enormous"),
        ],
        "summary": "Pre-training builds general knowledge; instruction tuning teaches the model how to use it helpfully.",
    },
    {
        "a": "sigmoid", "b": "ReLU",
        "rows": [
            ("Formula", "1 / (1 + e^-x)", "max(0, x)"),
            ("Output range", "(0, 1)", "[0, infinity)"),
            ("Vanishing gradients", "Yes (saturates, max derivative 0.25)", "Much less for positive inputs"),
            ("Computation", "Needs an exponential", "Very cheap"),
            ("Typical use", "Binary output layer, gates", "Hidden layers"),
        ],
        "summary": "Use ReLU (or GELU) in hidden layers and sigmoid for binary probability outputs.",
    },
    {
        "a": "batch gradient descent", "b": "stochastic gradient descent",
        "rows": [
            ("Data per update", "Entire dataset", "One example (or a mini-batch)"),
            ("Update noise", "Smooth", "Noisy"),
            ("Speed per update", "Slow", "Fast"),
            ("Memory", "High", "Low"),
            ("Escaping local minima", "Harder", "Noise can help"),
        ],
        "summary": "Mini-batch gradient descent combines the two and is the standard in deep learning.",
    },
    {
        "a": "BFS", "b": "DFS",
        "rows": [
            ("Frontier", "Queue (FIFO)", "Stack (LIFO)"),
            ("Exploration", "Level by level", "Deepest branch first"),
            ("Optimal (equal costs)", "Yes", "No"),
            ("Complete", "Yes (finite branching)", "No in infinite-depth spaces"),
            ("Memory", "O(b^d), high", "O(b*m), low"),
        ],
        "summary": "BFS finds the shallowest solution but uses a lot of memory; DFS is memory-efficient but not optimal. Iterative deepening combines their strengths.",
    },
]
