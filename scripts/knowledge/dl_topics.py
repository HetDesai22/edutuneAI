"""Deep Learning topics."""

DL_TOPICS = [
    {
        "key": "ann",
        "name": "Artificial Neural Network (ANN)",
        "short": "neural networks",
        "domain": "Deep Learning",
        "question": "What is an artificial neural network?",
        "definition": (
            "An artificial neural network (ANN) is a model made of layers of connected neurons. "
            "Each neuron computes a weighted sum of its inputs plus a bias and applies a "
            "non-linear activation function. Stacking layers lets the network learn complex "
            "functions from data."
        ),
        "simple": (
            "A neural network is a stack of simple calculators called neurons. Each neuron takes "
            "some numbers, multiplies them by weights, adds them up and passes the result through "
            "an activation function. The input layer receives the data, hidden layers find "
            "patterns, and the output layer gives the prediction. Training adjusts the weights so "
            "the predictions become correct."
        ),
        "eli10": (
            "Imagine a team of friends passing notes. The first friends look at a picture and "
            "each notices something small, like a line or a colour. They pass notes to the next "
            "friends, who combine them into bigger ideas like 'ear' or 'whisker'. The last friend "
            "says 'cat!'. A neural network is a team like that, and it learns which notes matter "
            "most."
        ),
        "analogy": (
            "A neural network is like a company's decision process. Front-line staff (input "
            "layer) collect raw information, middle managers (hidden layers) combine it into "
            "summaries, and the CEO (output layer) makes the final decision. Training is like "
            "performance reviews: whoever contributed to a wrong decision has their influence "
            "(weights) adjusted."
        ),
        "how_it_works": [
            "Input layer receives the features, e.g. pixel values.",
            "Each neuron computes z = w1*x1 + w2*x2 + ... + b, then a = activation(z).",
            "Hidden layers transform the representation step by step (forward propagation).",
            "The output layer produces predictions; a loss function measures the error.",
            "Backpropagation computes gradients and gradient descent updates all weights.",
        ],
        "example": (
            "Handwritten digit recognition (MNIST): each 28x28 image becomes 784 inputs. A "
            "network with one hidden layer of 128 ReLU neurons and an output layer of 10 softmax "
            "neurons (one per digit) can reach about 97-98% test accuracy after a few epochs."
        ),
        "applications": [
            "Image and speech recognition",
            "Tabular prediction such as credit scoring",
            "Recommendation systems",
            "As the building block of CNNs, RNNs and transformers",
        ],
        "advantages": [
            "Can approximate very complex, non-linear functions",
            "Learns features automatically from raw data",
            "Scales well with more data and compute",
        ],
        "limitations": [
            "Needs lots of data and compute",
            "Hard to interpret (black box)",
            "Many hyper-parameters; prone to overfitting without regularisation",
        ],
        "key_points": [
            "Neuron: weighted sum + bias, then a non-linear activation.",
            "Layers: input, one or more hidden layers, output.",
            "Forward propagation computes predictions; backpropagation computes gradients.",
            "Without non-linear activations, stacked layers collapse into one linear model.",
            "Universal approximation: a wide enough network can approximate any continuous function.",
        ],
        "viva": [
            ("What is a perceptron?", "The simplest neural network unit: it computes a weighted sum of inputs plus bias and applies a step function to output a class. A single perceptron can only learn linearly separable problems."),
            ("Why do we need activation functions?", "They introduce non-linearity. Without them, any number of layers is equivalent to a single linear transformation."),
            ("What is a hidden layer?", "A layer between input and output whose neurons learn intermediate representations of the data."),
            ("Why can't a single perceptron solve XOR?", "XOR is not linearly separable, and a single perceptron can only draw one straight line. A network with a hidden layer can solve it."),
            ("What is an epoch?", "One complete pass through the entire training dataset."),
        ],
        "interview": [
            ("How would you decide the number of hidden layers and neurons?", "Start simple - one or two hidden layers - and increase capacity until validation performance stops improving, while watching for overfitting. Use known architectures for the data type, tune with a validation set or random search, and add regularisation such as dropout and weight decay when capacity grows."),
            ("Why are weights initialised randomly rather than all zero?", "If all weights are equal, every neuron in a layer receives the same gradient and stays identical, so the layer behaves like one neuron. Random initialisation breaks this symmetry. Schemes like Xavier or He initialisation scale the randomness to keep activations and gradients stable."),
        ],
        "mcqs": [
            {"q": "What does a single artificial neuron compute before the activation?", "options": ["A weighted sum of inputs plus a bias", "The maximum input", "A random number", "The derivative of the loss"], "answer": "A", "explanation": "z = sum(w_i * x_i) + b, then the activation is applied."},
            {"q": "Which problem cannot be solved by a single-layer perceptron?", "options": ["AND", "OR", "XOR", "NOT"], "answer": "C", "explanation": "XOR is not linearly separable."},
            {"q": "One complete pass over the training data is called:", "options": ["A batch", "An epoch", "A step", "A layer"], "answer": "B", "explanation": "An epoch processes every training example once."},
        ],
        "numerical": {
            "problem": "A neuron has inputs x = (2, 3), weights w = (0.5, -1) and bias b = 1. Compute its output using the ReLU activation.",
            "solution": "Step 1: Weighted sum z = 0.5 * 2 + (-1) * 3 + 1 = 1 - 3 + 1 = -1.\nStep 2: ReLU(z) = max(0, z) = max(0, -1) = 0.\nAnswer: The neuron outputs 0 (it is inactive for this input).",
        },
    },
    {
        "key": "backpropagation",
        "name": "Backpropagation",
        "short": "backpropagation",
        "domain": "Deep Learning",
        "question": "What is backpropagation?",
        "definition": (
            "Backpropagation is the algorithm that computes the gradient of the loss with respect "
            "to every weight in a neural network by applying the chain rule of calculus layer by "
            "layer, from the output back to the input. These gradients are then used by gradient "
            "descent to update the weights."
        ),
        "simple": (
            "After the network makes a prediction, we measure how wrong it was. Backpropagation "
            "works backwards through the network and figures out how much each weight "
            "contributed to that error. Then each weight is nudged in the direction that reduces "
            "the error. Forward pass: make a prediction. Backward pass: assign blame and fix."
        ),
        "eli10": (
            "Imagine a relay team that lost a race. The coach watches the video backwards, from "
            "the finish line to the start, and works out how much each runner slowed the team "
            "down. Then each runner practises the part they got wrong. Backpropagation is the "
            "coach working backwards to find who needs to improve and by how much."
        ),
        "analogy": (
            "Backpropagation is like tracing a mistake in a factory assembly line. The final "
            "product is faulty, so inspectors start at the end of the line and move backwards, "
            "checking how much each station contributed to the defect. Each station then adjusts "
            "its process proportionally to its share of the blame."
        ),
        "how_it_works": [
            "Forward pass: compute each layer's outputs and the final loss, storing intermediate values.",
            "Compute the gradient of the loss with respect to the output.",
            "Apply the chain rule backwards: dL/dw = dL/da * da/dz * dz/dw for each layer.",
            "Reuse gradients from later layers to compute earlier ones efficiently.",
            "Update all weights with gradient descent: w = w - learning_rate * dL/dw.",
        ],
        "example": (
            "One neuron with y_hat = w*x, loss L = (y_hat - y)^2, x = 2, y = 10, w = 3. Forward: "
            "y_hat = 6, L = 16. Backward: dL/dy_hat = 2(6 - 10) = -8 and dy_hat/dw = x = 2, so "
            "dL/dw = -16. With learning rate 0.05, w becomes 3 - 0.05 * (-16) = 3.8, moving the "
            "prediction towards 10."
        ),
        "applications": [
            "Training every modern neural network: MLPs, CNNs, RNNs and transformers",
            "Fine-tuning LLMs and LoRA adapters",
            "Backpropagation through time for recurrent networks",
            "Computing input gradients for saliency maps and adversarial examples",
        ],
        "advantages": [
            "Efficient: computes all gradients in roughly the cost of one extra forward pass",
            "General: works for any differentiable architecture",
            "Automated by frameworks like PyTorch (autograd)",
        ],
        "limitations": [
            "Gradients can vanish or explode in deep or recurrent networks",
            "Requires storing activations, which uses a lot of memory",
            "Requires every operation to be differentiable",
        ],
        "key_points": [
            "Backpropagation = chain rule applied from output to input.",
            "Forward pass computes the loss; backward pass computes gradients.",
            "Gradients feed an optimiser such as SGD or Adam.",
            "Memory cost comes from storing activations (gradient checkpointing trades compute for memory).",
            "In RNNs it becomes backpropagation through time (BPTT).",
        ],
        "viva": [
            ("Which mathematical rule is backpropagation based on?", "The chain rule of calculus, which gives the derivative of composed functions."),
            ("What is the difference between the forward pass and the backward pass?", "The forward pass computes predictions and the loss; the backward pass computes the gradient of the loss with respect to every parameter."),
            ("Is backpropagation an optimiser?", "No. Backpropagation only computes gradients; an optimiser such as SGD or Adam uses them to update weights."),
            ("What is backpropagation through time?", "Backpropagation applied to an RNN unrolled over time steps, summing gradients across all steps for the shared weights."),
        ],
        "interview": [
            ("Why does backpropagation need to store activations, and how can you reduce that memory?", "The derivative for each layer depends on its input activations from the forward pass, so they must be kept until the backward pass. Gradient checkpointing stores only some activations and recomputes the rest during the backward pass, trading about 30% extra compute for much lower memory. Mixed precision and smaller batches also help."),
            ("What is the relationship between backpropagation and automatic differentiation?", "Backpropagation is reverse-mode automatic differentiation applied to neural networks. Frameworks like PyTorch record operations in a computation graph during the forward pass and traverse it in reverse to compute gradients automatically."),
        ],
        "mcqs": [
            {"q": "Backpropagation computes gradients using:", "options": ["The chain rule", "Bayes' theorem", "Random search", "The kernel trick"], "answer": "A", "explanation": "It applies the chain rule layer by layer from output to input."},
            {"q": "Which statement is true?", "options": ["Backpropagation updates the weights by itself", "Backpropagation computes gradients; the optimiser updates weights", "Backpropagation is only for CNNs", "Backpropagation needs no loss function"], "answer": "B", "explanation": "Gradient computation and the parameter update are separate steps."},
        ],
        "numerical": {
            "problem": "For y_hat = w * x with loss L = (y_hat - y)^2, let x = 1, y = 4 and w = 2. Compute dL/dw and the new weight after one update with learning rate 0.1.",
            "solution": "Step 1: Forward pass: y_hat = 2 * 1 = 2, L = (2 - 4)^2 = 4.\nStep 2: dL/dy_hat = 2(y_hat - y) = 2(2 - 4) = -4.\nStep 3: dy_hat/dw = x = 1.\nStep 4: Chain rule: dL/dw = -4 * 1 = -4.\nStep 5: Update: w = 2 - 0.1 * (-4) = 2.4.\nAnswer: dL/dw = -4 and the new weight is 2.4.",
        },
    },
    {
        "key": "activation_functions",
        "name": "Activation Functions",
        "short": "activation functions",
        "domain": "Deep Learning",
        "question": "What are activation functions in neural networks?",
        "definition": (
            "An activation function is the non-linear function a neuron applies to its weighted "
            "sum. It lets neural networks learn non-linear relationships. Common choices are "
            "sigmoid, tanh, ReLU, Leaky ReLU, GELU and softmax."
        ),
        "simple": (
            "An activation function decides how strongly a neuron 'fires' based on its input. "
            "Without it, a neural network would just be one big linear equation, no matter how "
            "many layers it had. ReLU, which outputs max(0, x), is the most common in hidden "
            "layers; sigmoid and softmax are used at the output to produce probabilities."
        ),
        "eli10": (
            "Think of a light dimmer switch. If you barely touch it, the light stays off; push "
            "harder and the light gets brighter. An activation function is like that switch for "
            "each tiny brain cell in the network: it decides how much the cell lights up."
        ),
        "analogy": (
            "Activation functions are like the bends in a garden hose. A straight hose (linear "
            "layers only) can only point in one direction. Adding bends at each joint (non-linear "
            "activations) lets you route water around any obstacle, just as activations let a "
            "network model complicated patterns."
        ),
        "how_it_works": [
            "The neuron computes z = w.x + b.",
            "The activation turns z into an output a = f(z).",
            "Sigmoid: 1/(1+e^-z), range (0, 1). Tanh: range (-1, 1), zero-centred.",
            "ReLU: max(0, z) - cheap and avoids saturation for positive inputs. Leaky ReLU and GELU are smooth or non-zero variants.",
            "Softmax: turns a vector of scores into probabilities that sum to 1 (multi-class output).",
        ],
        "example": (
            "For z = -2, 0 and 3: sigmoid gives 0.12, 0.5 and 0.95; tanh gives -0.96, 0 and "
            "0.995; ReLU gives 0, 0 and 3. Notice that sigmoid and tanh squash large values into "
            "a small range (their gradient becomes tiny), while ReLU passes positive values "
            "unchanged."
        ),
        "applications": [
            "ReLU / GELU in hidden layers of CNNs and transformers",
            "Sigmoid for binary classification outputs and LSTM gates",
            "Softmax for multi-class outputs and attention weights",
            "Tanh in RNN hidden states",
        ],
        "advantages": [
            "Enable neural networks to model non-linear functions",
            "ReLU is cheap to compute and speeds up training",
            "Output activations map scores to probabilities",
        ],
        "limitations": [
            "Sigmoid and tanh saturate, causing vanishing gradients",
            "ReLU can produce 'dead neurons' that always output 0",
            "A poor choice of activation can slow or prevent learning",
        ],
        "key_points": [
            "Activation functions add non-linearity.",
            "Sigmoid (0 to 1) and tanh (-1 to 1) saturate for large |z|.",
            "ReLU = max(0, z): default for hidden layers; Leaky ReLU fixes dead neurons.",
            "Softmax converts scores to a probability distribution.",
            "Sigmoid's maximum derivative is only 0.25, one cause of vanishing gradients.",
        ],
        "viva": [
            ("What is the range of the sigmoid function?", "Between 0 and 1 (exclusive)."),
            ("Why is ReLU preferred over sigmoid in hidden layers?", "It does not saturate for positive inputs, so gradients do not vanish as easily, and it is cheap to compute, which speeds up training."),
            ("What is the dying ReLU problem?", "Neurons whose input is always negative output 0 and receive zero gradient, so they stop learning. Leaky ReLU fixes this with a small slope for negative inputs."),
            ("Where is softmax used?", "In the output layer for multi-class classification, and to turn attention scores into weights in transformers."),
        ],
        "interview": [
            ("Which activation would you use in the output layer for binary, multi-class and regression tasks?", "Binary classification: sigmoid with binary cross-entropy. Multi-class, single label: softmax with categorical cross-entropy. Multi-label: independent sigmoids. Regression: a linear (identity) output with MSE or MAE loss."),
            ("Why do transformers use GELU instead of ReLU?", "GELU is a smooth version of ReLU that weights inputs by their probability under a Gaussian, so small negative values are not cut off abruptly. It gives slightly better optimisation and accuracy in large language models, although ReLU also works."),
        ],
        "mcqs": [
            {"q": "ReLU(x) is defined as:", "options": ["1/(1+e^-x)", "max(0, x)", "tanh(x)", "x^2"], "answer": "B", "explanation": "ReLU passes positive values and outputs 0 for negative ones."},
            {"q": "Which activation outputs values between -1 and 1?", "options": ["Sigmoid", "ReLU", "Tanh", "Softmax"], "answer": "C", "explanation": "Tanh is zero-centred with range (-1, 1)."},
            {"q": "Which activation is used to produce a probability distribution over many classes?", "options": ["Softmax", "ReLU", "Leaky ReLU", "Step"], "answer": "A", "explanation": "Softmax outputs non-negative values that sum to 1."},
        ],
        "numerical": {
            "problem": "Compute the sigmoid output and its derivative at z = 0.",
            "solution": "Step 1: sigmoid(z) = 1 / (1 + e^-z) = 1 / (1 + e^0) = 1 / 2 = 0.5.\nStep 2: The derivative is sigmoid(z) * (1 - sigmoid(z)) = 0.5 * 0.5 = 0.25.\nAnswer: Output 0.5 and derivative 0.25 - which is the maximum possible derivative of sigmoid.",
        },
    },
    {
        "key": "vanishing_gradient",
        "name": "Vanishing and Exploding Gradients",
        "short": "vanishing gradients",
        "domain": "Deep Learning",
        "question": "What is the vanishing gradient problem?",
        "definition": (
            "The vanishing gradient problem occurs when gradients become extremely small as they "
            "are backpropagated through many layers or time steps, so early layers learn very "
            "slowly or not at all. The exploding gradient problem is the opposite: gradients grow "
            "very large and make training unstable."
        ),
        "simple": (
            "During backpropagation, the gradient is multiplied by a factor at every layer. If "
            "those factors are smaller than 1 (like sigmoid's derivative, at most 0.25), the "
            "gradient shrinks towards zero and the early layers barely learn: that is vanishing "
            "gradients. If the factors are larger than 1, the gradient blows up: that is "
            "exploding gradients. It is a big problem in deep networks and in RNNs over long "
            "sequences."
        ),
        "eli10": (
            "Remember the whispering game, where a message is passed along a long line of "
            "friends? By the end, the message is so quiet and mixed up that the last person "
            "hears almost nothing. In a deep network, the 'learning message' fades the same way "
            "as it travels backwards through many layers."
        ),
        "analogy": (
            "Vanishing gradients are like repeatedly photocopying a photocopy. Each copy is a "
            "bit fainter, and after 50 copies the page is almost blank. Exploding gradients are "
            "like a microphone next to a speaker: each loop amplifies the sound until it becomes "
            "a deafening screech."
        ),
        "how_it_works": [
            "Backpropagation multiplies local derivatives layer by layer (the chain rule).",
            "With sigmoid or tanh, derivatives are below 1 (sigmoid at most 0.25), so the product shrinks exponentially with depth.",
            "In RNNs the same weight matrix is multiplied at every time step, so gradients vanish or explode over long sequences.",
            "Result: early layers or early time steps receive almost no learning signal, or updates become huge and cause NaN.",
            "Fixes: ReLU, careful initialisation, batch/layer normalisation, residual connections, LSTM/GRU gates and gradient clipping.",
        ],
        "example": (
            "In an RNN reading the sentence 'The cat, which was sitting on the mat near the door "
            "all afternoon, was hungry', the verb 'was' must agree with 'cat' many words earlier. "
            "With vanishing gradients, the error at 'was' barely reaches the weights involved at "
            "'cat', so a vanilla RNN fails to learn this long-range dependency. An LSTM keeps the "
            "information in its cell state."
        ),
        "applications": [
            "Explains why LSTMs and GRUs replaced vanilla RNNs",
            "Motivated ReLU and residual connections (ResNet) for very deep networks",
            "Motivated normalisation layers and careful initialisation (Xavier, He)",
            "Gradient clipping is standard in RNN and transformer training",
        ],
        "advantages": [
            "Understanding it guides architecture design",
            "Clear symptoms make it diagnosable: early-layer gradient norms near zero, or NaN losses",
            "Well-established fixes exist",
        ],
        "limitations": [
            "Prevents learning long-range dependencies in vanilla RNNs",
            "Makes very deep plain networks hard to train",
            "Exploding gradients cause unstable training and NaN losses",
        ],
        "key_points": [
            "Vanishing: gradients shrink exponentially through layers or time steps.",
            "Exploding: gradients grow exponentially and destabilise training.",
            "Cause: repeated multiplication by derivatives or weights smaller or larger than 1.",
            "Fixes for vanishing: ReLU, residual connections, normalisation, LSTM/GRU, good initialisation.",
            "Fix for exploding: gradient clipping, lower learning rate, good initialisation.",
        ],
        "viva": [
            ("What is the main cause of the vanishing gradient problem?", "Repeated multiplication of small derivatives during backpropagation, for example sigmoid derivatives of at most 0.25, which makes gradients shrink exponentially with depth."),
            ("How does LSTM address vanishing gradients?", "Its cell state is updated additively and controlled by gates, so gradients can flow across many time steps without being repeatedly squashed."),
            ("What is gradient clipping?", "Rescaling gradients when their norm exceeds a threshold, which prevents exploding gradients."),
            ("How do residual connections help?", "They add a shortcut path y = F(x) + x, so gradients can flow directly to earlier layers through the identity connection."),
            ("What is the main problem with vanilla RNNs?", "They suffer from vanishing and exploding gradients, so they struggle to learn long-term dependencies."),
        ],
        "interview": [
            ("How would you detect vanishing gradients during training?", "Monitor per-layer gradient norms, for example with TensorBoard. If early-layer gradients are orders of magnitude smaller than later layers and their weights barely change, while the loss plateaus early, gradients are vanishing. Exploding gradients show as sudden loss spikes, very large gradient norms or NaN values."),
            ("Why did ResNet make it possible to train networks with over 100 layers?", "Residual connections add the input of a block to its output, so the gradient has an identity path back to earlier layers. That prevents it from being repeatedly multiplied by small factors. Blocks only need to learn a residual correction, which also makes optimisation easier."),
        ],
        "mcqs": [
            {"q": "Which technique is mainly used to handle exploding gradients?", "options": ["Dropout", "Gradient clipping", "Max pooling", "Data augmentation"], "answer": "B", "explanation": "Clipping caps the gradient norm so updates stay stable."},
            {"q": "Which activation function is most associated with vanishing gradients?", "options": ["ReLU", "Sigmoid", "Leaky ReLU", "Linear"], "answer": "B", "explanation": "Sigmoid saturates and its derivative is at most 0.25."},
            {"q": "Which architecture was designed to reduce vanishing gradients in sequence models?", "options": ["CNN", "LSTM", "KNN", "SVM"], "answer": "B", "explanation": "LSTM gates and its additive cell state preserve gradient flow."},
        ],
        "numerical": {
            "problem": "A 10-layer network uses sigmoid activations. If each layer contributes at most the maximum sigmoid derivative (0.25) to the gradient, by what factor can the gradient shrink by the time it reaches the first layer?",
            "solution": "Step 1: The maximum derivative of sigmoid is 0.25.\nStep 2: Through 10 layers the factor is 0.25^10.\nStep 3: 0.25^10 = (1/4)^10 = 1 / 1,048,576, which is about 9.5 x 10^-7.\nAnswer: The gradient can shrink by a factor of about one million, which is why early layers barely learn.",
        },
        "troubleshooting": {
            "problem": "My RNN learns short sentences well but completely fails on long sequences. Why?",
            "solution": "This is the classic vanishing gradient problem. Gradients shrink as they are backpropagated through many time steps, so the network cannot learn long-range dependencies.\nFixes:\n1. Replace the vanilla RNN with an LSTM or GRU.\n2. Use gradient clipping to prevent the related exploding-gradient issue.\n3. Consider attention or a transformer for very long sequences.\n4. Use truncated backpropagation through time and good initialisation (e.g. orthogonal recurrent weights).",
        },
    },
    {
        "key": "cnn",
        "name": "Convolutional Neural Network (CNN)",
        "short": "CNN",
        "domain": "Deep Learning",
        "question": "What is a Convolutional Neural Network (CNN)?",
        "definition": (
            "A Convolutional Neural Network is a deep learning architecture designed for "
            "grid-like data such as images. It uses convolutional layers with small learnable "
            "filters that slide over the input to detect local patterns, pooling layers to reduce "
            "spatial size, and fully connected layers for the final prediction."
        ),
        "simple": (
            "A CNN looks at an image through small windows, called filters, that slide across it. "
            "Early filters detect simple things like edges and colours; deeper layers combine them "
            "into shapes like eyes or wheels, and finally into whole objects. Because the same "
            "filter is reused across the image, CNNs need far fewer parameters than a normal "
            "neural network."
        ),
        "eli10": (
            "Imagine looking for Waldo with a small magnifying glass. You slide it across the "
            "page bit by bit, looking for his red-and-white stripes. A CNN does the same with "
            "lots of tiny magnifying glasses, each looking for something different: lines, "
            "circles, stripes. Put together, they recognise the whole picture."
        ),
        "analogy": (
            "A CNN is like a team of inspectors checking a large wall mural. Each inspector has a "
            "small stencil (filter) for one pattern, such as a vertical line, and slides it over "
            "the whole wall, marking where it matches. Senior inspectors then look at these marks "
            "to find bigger patterns, like faces, built from the smaller ones."
        ),
        "how_it_works": [
            "Convolution layer: small filters (e.g. 3x3) slide over the input computing dot products, producing feature maps.",
            "Activation (ReLU) adds non-linearity.",
            "Pooling layer (e.g. 2x2 max pooling) downsamples feature maps, adding some translation invariance.",
            "Stacking layers builds a hierarchy: edges, then textures, then parts, then objects.",
            "Flatten and fully connected layers with softmax produce the class probabilities.",
        ],
        "example": (
            "Classifying cats vs dogs: a 224x224x3 image passes through conv layers with 32, 64 "
            "and 128 filters, each followed by ReLU and max pooling. The first layer's filters "
            "respond to edges, middle layers to fur texture and ears, and the last layers to "
            "whole faces. A final dense layer with softmax outputs, for example, cat 0.92 and "
            "dog 0.08."
        ),
        "applications": [
            "Image classification and object detection (e.g. YOLO)",
            "Medical imaging such as tumour detection in MRI scans",
            "Face recognition and self-driving car perception",
            "Also used on audio spectrograms and some text tasks",
        ],
        "advantages": [
            "Parameter sharing: far fewer weights than a fully connected network",
            "Learns spatial feature hierarchies automatically",
            "Translation invariance from convolution and pooling",
        ],
        "limitations": [
            "Needs a lot of labelled data (often solved with transfer learning)",
            "Computationally heavy for high-resolution inputs",
            "Limited global context; vision transformers capture long-range relations better",
        ],
        "key_points": [
            "CNNs use convolution, activation, pooling and fully connected layers.",
            "Filters (kernels) are learned and shared across positions.",
            "Output size = (W - F + 2P) / S + 1.",
            "Pooling reduces size and adds translation invariance.",
            "Famous architectures: LeNet, AlexNet, VGG, ResNet, EfficientNet.",
        ],
        "viva": [
            ("What is a convolution operation?", "Sliding a small filter over the input and computing the dot product at each position to produce a feature map that shows where the pattern appears."),
            ("What is the purpose of pooling?", "To reduce the spatial size of feature maps, lowering computation and parameters and giving some invariance to small shifts."),
            ("What are stride and padding?", "Stride is how many pixels the filter moves each step; padding adds border pixels (usually zeros) so the filter can cover edges and control the output size."),
            ("Why do CNNs have fewer parameters than fully connected networks?", "Because each filter is small and its weights are shared across all positions of the image."),
            ("What is a feature map?", "The output of applying one filter across the input, showing where that filter's pattern is detected."),
        ],
        "interview": [
            ("What is transfer learning and why is it used with CNNs?", "Transfer learning reuses a CNN pre-trained on a large dataset like ImageNet. Its early layers already detect general features like edges and textures, so we replace the final layer and fine-tune on our smaller dataset. This needs much less data and training time and usually gives higher accuracy than training from scratch."),
            ("What is the receptive field in a CNN?", "The region of the input image that affects a particular neuron's output. It grows with depth, pooling and stride, which is why deeper layers can recognise larger structures such as whole objects."),
        ],
        "mcqs": [
            {"q": "Which layer in a CNN reduces the spatial dimensions of feature maps?", "options": ["Convolution", "Pooling", "Softmax", "Embedding"], "answer": "B", "explanation": "Pooling (e.g. max pooling) downsamples feature maps."},
            {"q": "CNNs are most commonly used for:", "options": ["Tabular data only", "Image data", "Reinforcement rewards", "Database queries"], "answer": "B", "explanation": "CNNs exploit the grid structure of images."},
            {"q": "Sharing the same filter weights across the image is called:", "options": ["Dropout", "Parameter sharing", "Bagging", "Attention"], "answer": "B", "explanation": "Parameter sharing is a key reason CNNs are efficient."},
        ],
        "numerical": {
            "problem": "An input image is 32x32. A convolution layer uses a 5x5 filter, stride 1 and no padding. What is the output size? What if padding 2 is used?",
            "solution": "Formula: Output = (W - F + 2P) / S + 1.\nStep 1: No padding: (32 - 5 + 0) / 1 + 1 = 28, so the output is 28x28.\nStep 2: With padding 2: (32 - 5 + 4) / 1 + 1 = 32, so the output is 32x32.\nAnswer: 28x28 without padding and 32x32 with padding 2 ('same' padding).",
        },
        "troubleshooting": {
            "problem": "My CNN overfits badly because I only have 2,000 training images. What can I do?",
            "solution": "With a small image dataset:\n1. Use transfer learning - start from a pre-trained model such as ResNet and fine-tune only the last layers first.\n2. Apply data augmentation - random flips, crops, rotations and colour jitter.\n3. Add regularisation - dropout, weight decay and early stopping.\n4. Reduce model size if training from scratch.\n5. Use cross-validation to get a reliable estimate with so little data.",
        },
    },
    {
        "key": "rnn",
        "name": "Recurrent Neural Network (RNN)",
        "short": "RNN",
        "domain": "Deep Learning",
        "question": "What is a Recurrent Neural Network (RNN)?",
        "definition": (
            "A Recurrent Neural Network is a neural network designed for sequential data. It "
            "processes the sequence one step at a time and keeps a hidden state that is updated "
            "at every step, h_t = tanh(W_x x_t + W_h h_(t-1) + b), so information from earlier "
            "steps can influence later ones."
        ),
        "simple": (
            "An RNN is a neural network with memory. When it reads a sentence, it processes one "
            "word at a time and carries a 'summary so far' (the hidden state) to the next step. "
            "That is why it is used for text, speech and time series, where order matters. Its "
            "weakness is that the memory fades over long sequences because of vanishing gradients."
        ),
        "eli10": (
            "Imagine reading a story one word at a time, and after each word you update a little "
            "note in your head about what is happening. By the end, your note helps you understand "
            "the ending. An RNN reads the same way, keeping a small memory note as it goes. But "
            "if the story is very long, it starts forgetting the beginning."
        ),
        "analogy": (
            "An RNN is like a person listening to a friend's story over the phone. They cannot "
            "see the whole story at once; they hear one sentence at a time and keep a running "
            "mental summary. Each new sentence is interpreted in light of that summary. If the "
            "story goes on for an hour, details from the start get fuzzy - just like vanishing "
            "gradients in an RNN."
        ),
        "how_it_works": [
            "The input sequence x_1, x_2, ..., x_T is processed one step at a time.",
            "At each step: h_t = tanh(W_x x_t + W_h h_(t-1) + b) - the hidden state mixes the new input with the previous memory.",
            "The same weights are shared across all time steps.",
            "An output y_t = W_y h_t can be produced at each step or only at the end.",
            "Training uses backpropagation through time (BPTT), which is where vanishing and exploding gradients arise.",
        ],
        "example": (
            "Next-word prediction: reading 'I grew up in France, so I speak fluent ...', the RNN "
            "updates its hidden state word by word and predicts 'French'. For sentiment analysis, "
            "an RNN reads a whole review and uses its final hidden state to classify it as "
            "positive or negative."
        ),
        "applications": [
            "Language modelling and text generation",
            "Speech recognition",
            "Time-series forecasting, e.g. stock prices or weather",
            "Sentiment analysis and machine translation (encoder-decoder RNNs)",
        ],
        "advantages": [
            "Handles variable-length sequences",
            "Shares weights across time steps, so parameters do not grow with sequence length",
            "Captures the order and context of sequential data",
        ],
        "limitations": [
            "Vanishing and exploding gradients make long-term dependencies hard to learn",
            "Sequential processing cannot be parallelised across time, so training is slow",
            "Largely replaced by LSTM, GRU and especially transformers",
        ],
        "key_points": [
            "RNNs process sequences step by step with a hidden state (memory).",
            "h_t = tanh(W_x x_t + W_h h_(t-1) + b); weights are shared over time.",
            "Trained with backpropagation through time (BPTT).",
            "Main problem: vanishing/exploding gradients over long sequences.",
            "LSTM and GRU add gates to fix this; transformers replace recurrence with attention.",
        ],
        "viva": [
            ("What is the hidden state in an RNN?", "A vector that summarises the information seen so far in the sequence; it is updated at every time step and passed to the next."),
            ("What is the main problem with vanilla RNNs?", "Vanishing and exploding gradients during backpropagation through time, which make it hard to learn long-term dependencies."),
            ("Why are RNNs suited to sequential data?", "Because they process inputs in order and carry information forward through the hidden state, so earlier inputs influence later outputs."),
            ("What is backpropagation through time?", "Training an RNN by unrolling it across time steps and applying backpropagation to the unrolled network, accumulating gradients for the shared weights."),
            ("Name two variants of RNN that handle long-term dependencies better.", "LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit)."),
        ],
        "interview": [
            ("What is the difference between RNN and LSTM?", "Both process sequences step by step, but a vanilla RNN has a single hidden state updated through a tanh layer, so gradients vanish over long sequences. An LSTM adds a separate cell state and three gates - forget, input and output - that control what to keep, add and expose. The additive cell-state update lets gradients flow over many steps, so LSTMs learn long-term dependencies much better, at the cost of about four times more parameters."),
            ("Why have transformers largely replaced RNNs?", "RNNs must process tokens one after another, which prevents parallel training on GPUs and makes long-range dependencies hard to learn. Transformers use self-attention, so every token can directly attend to every other token, and whole sequences are processed in parallel. This scales much better with data and compute."),
        ],
        "mcqs": [
            {"q": "Which neural network is commonly used for sequential data?", "options": ["CNN", "RNN", "GAN", "Autoencoder"], "answer": "B", "explanation": "RNNs are designed to process sequences with a hidden state that carries context."},
            {"q": "RNNs are trained using:", "options": ["Backpropagation through time", "K-Means", "The kernel trick", "Bagging"], "answer": "A", "explanation": "The RNN is unrolled across time steps and backpropagation is applied."},
            {"q": "The main limitation of vanilla RNNs is:", "options": ["They cannot handle images", "Vanishing gradients over long sequences", "They have no parameters", "They need labelled clusters"], "answer": "B", "explanation": "Gradients shrink through many time steps, limiting long-term memory."},
        ],
        "numerical": {
            "problem": "A vanilla RNN has input size 10 and hidden size 20. How many trainable parameters are in the recurrent layer (W_x, W_h and bias)?",
            "solution": "Step 1: W_x maps input to hidden: 20 x 10 = 200 parameters.\nStep 2: W_h maps hidden to hidden: 20 x 20 = 400 parameters.\nStep 3: Bias: 20 parameters.\nStep 4: Total = 200 + 400 + 20 = 620.\nAnswer: 620 trainable parameters (some libraries use two bias vectors, giving 640).",
        },
    },
    {
        "key": "lstm",
        "name": "Long Short-Term Memory (LSTM)",
        "short": "LSTM",
        "domain": "Deep Learning",
        "question": "What is an LSTM?",
        "definition": (
            "Long Short-Term Memory (LSTM) is a type of recurrent neural network that adds a cell "
            "state and three gates - forget, input and output - to control what information is "
            "kept, added and output at each time step. This lets it learn long-term dependencies "
            "and greatly reduces the vanishing gradient problem."
        ),
        "simple": (
            "An LSTM is an RNN with a better memory system. Besides the normal hidden state, it "
            "has a 'cell state' that works like a conveyor belt carrying important information "
            "along the sequence. Three gates decide what to forget, what new information to "
            "store and what to output. This lets LSTMs remember things from much earlier in a "
            "sequence."
        ),
        "eli10": (
            "Imagine you have a backpack for a long school trip. At each stop you decide what to "
            "throw away (forget gate), what new souvenirs to put in (input gate), and what to "
            "take out and show your friends (output gate). Because you choose carefully, you "
            "still have your important things at the end of the trip. An LSTM's memory works "
            "like that backpack."
        ),
        "analogy": (
            "Imagine you are reading a long story. You remember the main character's name "
            "throughout (kept in the cell state), forget minor details like what someone had for "
            "breakfast (forget gate), note important new events as they happen (input gate), and "
            "recall just the relevant facts when answering a question about the story (output "
            "gate). An LSTM manages its memory in the same selective way."
        ),
        "how_it_works": [
            "Forget gate f_t = sigmoid(W_f [h_(t-1), x_t] + b_f) decides which parts of the old cell state to erase.",
            "Input gate i_t and candidate values c~_t = tanh(...) decide what new information to add.",
            "Cell state update: c_t = f_t * c_(t-1) + i_t * c~_t (an additive update).",
            "Output gate o_t decides what to expose: h_t = o_t * tanh(c_t).",
            "Because the cell state is updated additively, gradients can flow across many steps without vanishing.",
        ],
        "example": (
            "Sentence: 'Maria grew up in Spain ... [50 words later] ... she speaks fluent "
            "Spanish.' When the LSTM reads 'Spain', the input gate stores it in the cell state. "
            "The forget gate keeps it through the unrelated words in between. At 'fluent', the "
            "output gate exposes it and the model predicts 'Spanish'. A vanilla RNN would "
            "usually have lost this information."
        ),
        "applications": [
            "Speech recognition and handwriting recognition",
            "Machine translation (before transformers)",
            "Time-series forecasting such as energy demand",
            "Text generation and sentiment analysis",
        ],
        "advantages": [
            "Learns long-term dependencies much better than a vanilla RNN",
            "Gates give fine-grained control over memory",
            "Works well on moderate-length sequences and time series",
        ],
        "limitations": [
            "About four times more parameters than a vanilla RNN, so slower",
            "Still sequential, so training cannot be parallelised over time",
            "Transformers usually outperform LSTMs on large-scale language tasks",
        ],
        "key_points": [
            "LSTM adds a cell state and three gates: forget, input, output.",
            "Gates use sigmoid (0 to 1) to decide how much information passes.",
            "c_t = f_t * c_(t-1) + i_t * c~_t: the additive update eases gradient flow.",
            "Solves most of the vanishing gradient problem of vanilla RNNs.",
            "Parameters = 4 * (h * (h + x) + h) for hidden size h and input size x.",
        ],
        "viva": [
            ("What are the three gates in an LSTM?", "The forget gate, the input gate and the output gate."),
            ("What is the role of the forget gate?", "It decides which information in the previous cell state should be discarded, outputting values between 0 (forget) and 1 (keep)."),
            ("How does LSTM solve the vanishing gradient problem?", "The cell state is updated additively and regulated by gates, which creates a path where gradients can flow across many time steps without being repeatedly squashed."),
            ("What is the difference between the cell state and the hidden state?", "The cell state is the long-term memory carried along the sequence; the hidden state is the filtered output at each step, controlled by the output gate."),
            ("Why is sigmoid used in the gates?", "It outputs values between 0 and 1, which act as a soft switch deciding what fraction of information passes through."),
        ],
        "interview": [
            ("Walk me through what happens inside an LSTM cell at one time step.", "The cell takes the current input x_t and the previous hidden state h_(t-1). The forget gate decides what to erase from the previous cell state. The input gate and a tanh candidate layer decide what new information to write. The cell state is updated as c_t = f_t * c_(t-1) + i_t * c~_t. Finally, the output gate filters tanh(c_t) to produce the new hidden state h_t, which is the output and is passed to the next step."),
            ("When would you still choose an LSTM over a transformer today?", "For small datasets, streaming or on-device applications where memory and latency are tight, and some time-series problems where the sequential inductive bias helps. LSTMs process each new step in constant memory, while a transformer's attention cost grows with sequence length."),
        ],
        "mcqs": [
            {"q": "Which LSTM gate decides what information to discard from the cell state?", "options": ["Input gate", "Output gate", "Forget gate", "Reset gate"], "answer": "C", "explanation": "The forget gate scales the previous cell state by values between 0 and 1."},
            {"q": "LSTMs were designed mainly to address:", "options": ["Overfitting in CNNs", "The vanishing gradient problem in RNNs", "Slow K-Means convergence", "Class imbalance"], "answer": "B", "explanation": "Gated, additive memory lets gradients flow over long sequences."},
            {"q": "How many gates does a standard LSTM cell have?", "options": ["One", "Two", "Three", "Four"], "answer": "C", "explanation": "Forget, input and output gates."},
        ],
        "numerical": {
            "problem": "An LSTM layer has input size 8 and hidden size 16. How many trainable parameters does it have (single bias per gate)?",
            "solution": "Step 1: Each of the 4 components (forget, input, candidate, output) has weights for [h_(t-1), x_t] plus a bias.\nStep 2: Per component: h * (h + x) + h = 16 * (16 + 8) + 16 = 16 * 24 + 16 = 400.\nStep 3: Total = 4 * 400 = 1,600.\nAnswer: 1,600 trainable parameters (PyTorch uses two bias vectors, giving 1,664).",
        },
    },
    {
        "key": "gru",
        "name": "Gated Recurrent Unit (GRU)",
        "short": "GRU",
        "domain": "Deep Learning",
        "question": "What is a GRU?",
        "definition": (
            "A Gated Recurrent Unit (GRU) is a simplified gated RNN that merges the LSTM's cell "
            "and hidden state into one hidden state and uses two gates: an update gate that "
            "controls how much past information to keep, and a reset gate that controls how much "
            "past information to use when computing the new candidate state."
        ),
        "simple": (
            "A GRU is like a lighter LSTM. It keeps only one memory (the hidden state) and two "
            "gates. The update gate decides how much of the old memory to keep versus replace, "
            "and the reset gate decides how much of the old memory to consider when forming new "
            "information. It is faster than an LSTM and often performs just as well."
        ),
        "eli10": (
            "Imagine a notebook where you write what is happening in a movie. At each scene you "
            "decide: should I keep my old notes or rewrite them (update gate)? And when writing "
            "new notes, should I look at my old notes or start fresh (reset gate)? A GRU manages "
            "its memory with just these two decisions."
        ),
        "analogy": (
            "If an LSTM is a filing cabinet with separate drawers for long-term files and today's "
            "notes, a GRU is a single well-organised notebook with a pencil and an eraser. It "
            "does almost the same job with fewer moving parts."
        ),
        "how_it_works": [
            "Update gate z_t = sigmoid(W_z [h_(t-1), x_t]) decides how much of the old state to keep.",
            "Reset gate r_t = sigmoid(W_r [h_(t-1), x_t]) decides how much of the old state to use for the candidate.",
            "Candidate state: h~_t = tanh(W [r_t * h_(t-1), x_t]).",
            "New state: h_t = (1 - z_t) * h_(t-1) + z_t * h~_t, an interpolation between old and new.",
            "No separate cell state and no output gate, so about 3/4 of the parameters of an LSTM.",
        ],
        "example": (
            "For real-time keyword spotting on a phone, a GRU with 64 hidden units processes "
            "audio frames and detects phrases like 'Hey assistant'. It achieves accuracy similar "
            "to an LSTM but runs faster and uses less memory, which matters on mobile hardware."
        ),
        "applications": [
            "Speech and keyword recognition on devices",
            "Time-series forecasting",
            "Text classification and generation on smaller datasets",
            "Music and sequence modelling",
        ],
        "advantages": [
            "Fewer parameters than LSTM, so faster training and inference",
            "Handles vanishing gradients much better than vanilla RNNs",
            "Often matches LSTM accuracy, especially on smaller datasets",
        ],
        "limitations": [
            "Slightly less expressive than LSTM on some very long-sequence tasks",
            "Still sequential, so no parallelism over time",
            "Generally outperformed by transformers on large-scale tasks",
        ],
        "key_points": [
            "GRU has two gates: update and reset.",
            "It merges the cell state and hidden state into one hidden state.",
            "h_t = (1 - z_t) * h_(t-1) + z_t * h~_t.",
            "Parameters = 3 * (h * (h + x) + h), about 25% fewer than LSTM.",
            "A good default when compute is limited.",
        ],
        "viva": [
            ("What are the two gates in a GRU?", "The update gate and the reset gate."),
            ("How does a GRU differ from an LSTM?", "A GRU has two gates instead of three and no separate cell state, so it is simpler and has fewer parameters."),
            ("What does the reset gate do?", "It controls how much of the previous hidden state is used when computing the new candidate state."),
            ("Does GRU suffer from vanishing gradients like a vanilla RNN?", "Much less, because the update gate allows information and gradients to pass through many steps almost unchanged."),
        ],
        "interview": [
            ("How do you choose between GRU and LSTM for a project?", "Treat it as an empirical choice. GRU is faster and has fewer parameters, so it is a good first choice with limited data or compute. LSTM can be slightly better on very long sequences or complex tasks. I would try both with the same budget and compare validation performance, and consider a transformer if the data is large."),
        ],
        "mcqs": [
            {"q": "How many gates does a GRU have?", "options": ["One", "Two", "Three", "Four"], "answer": "B", "explanation": "A GRU has an update gate and a reset gate."},
            {"q": "Compared with an LSTM, a GRU:", "options": ["Has more parameters", "Has a separate cell state", "Has fewer parameters", "Cannot handle sequences"], "answer": "C", "explanation": "It has three gated components instead of four, so about 25% fewer parameters."},
        ],
        "numerical": {
            "problem": "A GRU layer has input size 8 and hidden size 16. How many trainable parameters does it have (single bias per gate)?",
            "solution": "Step 1: A GRU has 3 components: update gate, reset gate and candidate state.\nStep 2: Per component: h * (h + x) + h = 16 * 24 + 16 = 400.\nStep 3: Total = 3 * 400 = 1,200.\nAnswer: 1,200 parameters, compared with 1,600 for an LSTM of the same size (25% fewer).",
        },
    },
]
