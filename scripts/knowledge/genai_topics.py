"""Generative AI topics."""

GENAI_TOPICS = [
    {
        "key": "gan",
        "name": "Generative Adversarial Network (GAN)",
        "short": "GAN",
        "domain": "Generative AI",
        "question": "What is a Generative Adversarial Network (GAN)?",
        "definition": (
            "A Generative Adversarial Network is a generative model made of two neural networks "
            "trained against each other: a generator that creates fake samples from random noise, "
            "and a discriminator that tries to tell real samples from fake ones. Competition "
            "pushes the generator to produce realistic data."
        ),
        "simple": (
            "A GAN is two networks playing a game. The generator tries to create fake images that "
            "look real, and the discriminator tries to catch the fakes. Every time the "
            "discriminator catches a fake, the generator learns to make better ones. After a lot "
            "of training, the generator can create new, realistic images that never existed."
        ),
        "eli10": (
            "Imagine a kid who makes fake banknotes for a board game and a friend who checks "
            "whether each note is real. At first the fakes are easy to spot. But the kid keeps "
            "learning from being caught, and the friend keeps getting sharper. Eventually the "
            "fakes look just like the real thing. That game is a GAN."
        ),
        "analogy": (
            "A GAN is like an art forger and an art detective. The forger (generator) paints "
            "fake masterpieces; the detective (discriminator) examines paintings and says 'real' "
            "or 'fake'. Each learns from the other: the detective spots subtler flaws, so the "
            "forger fixes them. In the end the forgeries are indistinguishable from real art."
        ),
        "how_it_works": [
            "The generator G takes random noise z and produces a fake sample G(z).",
            "The discriminator D receives real and fake samples and outputs the probability that each is real.",
            "Train D to maximise its accuracy at telling real from fake.",
            "Train G to fool D, i.e. to make D(G(z)) close to 1.",
            "Alternate the two updates; ideally they reach an equilibrium where D outputs 0.5 for everything.",
        ],
        "example": (
            "StyleGAN, trained on thousands of face photos, generates high-resolution portraits of "
            "people who do not exist. During training, the discriminator learns to notice "
            "unnatural skin texture or asymmetric eyes, which forces the generator to produce "
            "increasingly realistic faces."
        ),
        "applications": [
            "Realistic image and face generation",
            "Image-to-image translation (sketch to photo, day to night)",
            "Super-resolution and image inpainting",
            "Synthetic data generation for training other models",
        ],
        "advantages": [
            "Produces very sharp, realistic samples",
            "No need for an explicit likelihood function",
            "Fast sampling: one forward pass of the generator",
        ],
        "limitations": [
            "Training is unstable and hard to balance",
            "Mode collapse: the generator produces only a few kinds of outputs",
            "Hard to evaluate; can be misused for deepfakes",
        ],
        "key_points": [
            "GAN = generator vs discriminator in a minimax game.",
            "Objective: min_G max_D E[log D(x)] + E[log(1 - D(G(z)))].",
            "Common problems: mode collapse, non-convergence, vanishing generator gradients.",
            "Variants: DCGAN, WGAN, CycleGAN, StyleGAN.",
            "Diffusion models have largely overtaken GANs for image generation.",
        ],
        "viva": [
            ("What are the two components of a GAN?", "The generator, which creates fake samples from noise, and the discriminator, which classifies samples as real or fake."),
            ("What is mode collapse?", "When the generator produces only a limited variety of outputs, for example the same few faces, because they reliably fool the discriminator."),
            ("What is the input to the generator?", "A random noise vector sampled from a simple distribution, such as a Gaussian."),
            ("What does the discriminator output?", "The probability that its input is a real sample rather than a generated one."),
        ],
        "interview": [
            ("Why is GAN training unstable, and how do you stabilise it?", "The two networks are optimised against each other, so it is a moving-target game rather than simple minimisation. If the discriminator gets too strong, the generator's gradients vanish; if it is too weak, the generator gets no useful signal. Stabilisation techniques include the Wasserstein loss with gradient penalty (WGAN-GP), spectral normalisation, label smoothing, balanced learning rates and architectural guidelines from DCGAN."),
            ("Compare GANs with VAEs.", "GANs learn through an adversarial game and produce sharp samples, but training is unstable and there is no encoder or likelihood. VAEs learn an explicit encoder-decoder with a probabilistic latent space and train stably by maximising the ELBO, but their samples tend to be blurrier. VAEs also give a structured latent space useful for interpolation and representation learning."),
        ],
        "mcqs": [
            {"q": "In a GAN, the discriminator's job is to:", "options": ["Generate images", "Classify samples as real or fake", "Compress data", "Tokenise text"], "answer": "B", "explanation": "The discriminator is a binary classifier of real versus generated samples."},
            {"q": "When a GAN generator produces very little variety, it is called:", "options": ["Overfitting", "Mode collapse", "Vanishing gradient", "Dropout"], "answer": "B", "explanation": "Mode collapse means the generator covers only a few modes of the data distribution."},
        ],
        "troubleshooting": {
            "problem": "My GAN generator keeps producing almost the same image for every noise input. How do I fix this?",
            "solution": "This is mode collapse. Try:\n1. Switch to the Wasserstein loss with gradient penalty (WGAN-GP), which gives smoother gradients.\n2. Use minibatch discrimination or minibatch standard deviation so the discriminator can detect low variety.\n3. Balance training - adjust learning rates, or update the discriminator more often.\n4. Add spectral normalisation to the discriminator.\n5. Check that the noise vector actually reaches the generator and is not ignored.",
        },
    },
    {
        "key": "vae",
        "name": "Variational Autoencoder (VAE)",
        "short": "VAE",
        "domain": "Generative AI",
        "question": "What is a Variational Autoencoder (VAE)?",
        "definition": (
            "A Variational Autoencoder is a generative model with an encoder that maps an input "
            "to a probability distribution (mean and variance) in a latent space, and a decoder "
            "that reconstructs data from samples of that distribution. It is trained to "
            "reconstruct inputs well while keeping the latent space close to a standard normal "
            "distribution."
        ),
        "simple": (
            "A normal autoencoder squeezes data into a small code and rebuilds it. A VAE makes "
            "that code fuzzy: instead of one point, each input becomes a small cloud (a "
            "distribution) in the latent space. Because the whole latent space is organised and "
            "smooth, you can pick a random point and decode it into a brand-new, realistic "
            "sample. That is what makes VAEs generative."
        ),
        "eli10": (
            "Imagine describing any animal using just two sliders: 'how big' and 'how fluffy'. "
            "A VAE learns sliders like these by itself. Once it has learned them, you can set the "
            "sliders to new positions and it will draw a new animal that matches. The sliders "
            "are the latent space."
        ),
        "analogy": (
            "A VAE is like a music producer's mixing desk. The encoder listens to a song and "
            "sets the knobs (latent variables) roughly; the decoder plays music from the knob "
            "settings. Because the knobs are organised smoothly, turning them slightly produces "
            "a slightly different but still musical song, so you can create new songs by "
            "choosing new knob positions."
        ),
        "how_it_works": [
            "Encoder: maps input x to a mean mu and variance sigma^2 of a latent Gaussian q(z|x).",
            "Reparameterisation trick: z = mu + sigma * epsilon with epsilon ~ N(0, 1), so sampling stays differentiable.",
            "Decoder: reconstructs x_hat from z.",
            "Loss (negative ELBO) = reconstruction loss + KL divergence between q(z|x) and N(0, I).",
            "To generate: sample z from N(0, I) and decode it.",
        ],
        "example": (
            "A VAE trained on handwritten digits with a 2-D latent space arranges digits smoothly: "
            "moving across the latent plane gradually morphs a '1' into a '7' and then a '9'. "
            "Sampling a random point and decoding it produces a new, plausible digit."
        ),
        "applications": [
            "Generating images, molecules and music",
            "Anomaly detection using reconstruction error",
            "Learning smooth latent representations for interpolation",
            "The latent encoder in Stable Diffusion is a VAE",
        ],
        "advantages": [
            "Stable training with a clear objective (the ELBO)",
            "Smooth, continuous latent space that supports interpolation",
            "Has an encoder, so it can represent existing data too",
        ],
        "limitations": [
            "Samples are often blurrier than GAN or diffusion outputs",
            "Balancing reconstruction and KL terms is tricky (posterior collapse)",
            "The Gaussian assumption can limit expressiveness",
        ],
        "key_points": [
            "VAE = probabilistic encoder + decoder with a regularised latent space.",
            "Loss = reconstruction loss + KL divergence (negative ELBO).",
            "The reparameterisation trick enables backpropagation through sampling.",
            "Generate by sampling z ~ N(0, I) and decoding.",
            "Unlike a plain autoencoder, a VAE's latent space is continuous and suited to sampling.",
        ],
        "viva": [
            ("What is the reparameterisation trick?", "Writing the latent sample as z = mu + sigma * epsilon with epsilon drawn from N(0, 1), so the randomness is separated from the parameters and gradients can flow through mu and sigma."),
            ("What are the two terms in the VAE loss?", "A reconstruction loss that measures how well the output matches the input, and a KL divergence that keeps the latent distribution close to a standard normal."),
            ("How is a VAE different from a normal autoencoder?", "A normal autoencoder maps each input to a single point; a VAE maps it to a distribution and regularises the latent space, which makes it possible to generate new samples."),
            ("What does KL divergence measure?", "How different one probability distribution is from another."),
        ],
        "interview": [
            ("Why are VAE outputs often blurry?", "The reconstruction loss, often pixel-wise MSE, corresponds to a Gaussian likelihood that rewards predicting the average of plausible outputs. Averaging many sharp possibilities gives a blurry image. The KL term also limits how much information the latent code carries. Better decoders, perceptual losses, or combining a VAE with a GAN or diffusion model reduce blur."),
        ],
        "mcqs": [
            {"q": "Which term in the VAE loss keeps the latent distribution close to N(0, I)?", "options": ["Reconstruction loss", "KL divergence", "Hinge loss", "Cross-entropy with labels"], "answer": "B", "explanation": "The KL term regularises the latent space."},
            {"q": "The reparameterisation trick is used to:", "options": ["Speed up inference", "Allow backpropagation through random sampling", "Remove the decoder", "Increase the batch size"], "answer": "B", "explanation": "It rewrites sampling as a deterministic function plus independent noise."},
        ],
    },
    {
        "key": "transformers",
        "name": "Transformer Architecture",
        "short": "transformers",
        "domain": "Generative AI",
        "question": "What is a Transformer in deep learning?",
        "definition": (
            "The Transformer is a neural network architecture introduced in 'Attention Is All You "
            "Need' (2017) that processes sequences using self-attention instead of recurrence. "
            "Stacks of attention and feed-forward layers, with residual connections and layer "
            "normalisation, let every token relate directly to every other token, and the whole "
            "sequence is processed in parallel."
        ),
        "simple": (
            "A transformer reads all the words of a sentence at once rather than one by one. For "
            "each word, self-attention works out which other words are important for "
            "understanding it. In 'The animal didn't cross the street because it was tired', "
            "attention helps link 'it' to 'animal'. Because everything is processed in parallel, "
            "transformers train fast on GPUs, which is why they power modern LLMs like GPT."
        ),
        "eli10": (
            "Imagine a classroom where, instead of passing notes one by one down a row, every "
            "student can look at everyone else's notes at the same time and decide whose notes "
            "help them most. That is much faster and nobody's message gets lost. A transformer "
            "lets every word 'look at' every other word like that."
        ),
        "analogy": (
            "An RNN is like reading a book one word at a time through a narrow slit. A "
            "transformer is like laying every page on a big table so you can see everything at "
            "once, drawing lines between related words. Multi-head attention is like several "
            "readers each drawing different kinds of lines: one tracks grammar, another tracks "
            "who did what."
        ),
        "how_it_works": [
            "Tokenise the text and convert tokens to embeddings, adding positional information (positional encodings or rotary embeddings).",
            "Self-attention: each token forms query, key and value vectors; Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V.",
            "Multi-head attention runs several attention operations in parallel to capture different relationships.",
            "A position-wise feed-forward network processes each token; residual connections and layer normalisation stabilise training.",
            "Encoder-only (BERT), decoder-only (GPT, with causal masking) and encoder-decoder (T5) variants exist.",
        ],
        "example": (
            "In machine translation of 'The cat sat on the mat' into French, the encoder builds "
            "contextual representations of all six English words in parallel. The decoder "
            "generates 'Le chat ...' one token at a time, using cross-attention to focus on "
            "'cat' when producing 'chat' and on 'mat' when producing 'tapis'."
        ),
        "applications": [
            "Large language models (GPT, Llama, Qwen, Claude)",
            "Machine translation and summarisation",
            "Vision transformers for image classification",
            "Speech recognition (Whisper) and protein structure prediction (AlphaFold)",
        ],
        "advantages": [
            "Parallel processing of sequences, so training is fast on GPUs",
            "Captures long-range dependencies directly through attention",
            "Scales extremely well with data and parameters",
        ],
        "limitations": [
            "Self-attention cost grows quadratically with sequence length",
            "Needs huge datasets and compute to pre-train",
            "Has no built-in sense of order, so positional encoding is required",
        ],
        "key_points": [
            "Transformers replace recurrence with self-attention.",
            "Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V.",
            "Building blocks: multi-head attention, feed-forward layers, residual connections, layer norm, positional encoding.",
            "Encoder-only (BERT), decoder-only (GPT), encoder-decoder (T5).",
            "Attention complexity is O(n^2) in sequence length n.",
        ],
        "viva": [
            ("What paper introduced the transformer?", "'Attention Is All You Need' by Vaswani et al., 2017."),
            ("Why do transformers need positional encoding?", "Self-attention treats the input as a set and has no notion of order, so positional information must be added to the embeddings."),
            ("What is multi-head attention?", "Running several attention operations in parallel with different learned projections, so the model can attend to different kinds of relationships at once, then concatenating the results."),
            ("What is the difference between BERT and GPT?", "BERT is an encoder-only model trained with masked language modelling and sees context in both directions; GPT is a decoder-only model trained to predict the next token using causal (left-to-right) attention."),
            ("Why are transformers faster to train than RNNs?", "They process all tokens of a sequence in parallel, whereas RNNs must process tokens one after another."),
        ],
        "interview": [
            ("Why do we divide by sqrt(d_k) in scaled dot-product attention?", "Dot products of high-dimensional vectors grow in magnitude with the dimension d_k. Large values push softmax into regions with extremely small gradients, where one token takes almost all the weight. Dividing by sqrt(d_k) keeps the variance of the scores around 1, which keeps softmax well-behaved and training stable."),
            ("How would you handle very long documents with a transformer?", "Options include chunking with overlap, sliding-window or sparse attention (Longformer), efficient kernels like FlashAttention to reduce memory, positional schemes that extrapolate (RoPE scaling, ALiBi), and retrieval-augmented generation, which retrieves only the relevant chunks instead of feeding the whole document."),
        ],
        "mcqs": [
            {"q": "The key mechanism that transformers use instead of recurrence is:", "options": ["Convolution", "Self-attention", "Pooling", "Bagging"], "answer": "B", "explanation": "Self-attention lets each token attend directly to all others."},
            {"q": "GPT models are:", "options": ["Encoder-only", "Decoder-only", "Encoder-decoder", "Convolutional"], "answer": "B", "explanation": "GPT uses a stack of decoder blocks with causal masking."},
            {"q": "Why is positional encoding needed in transformers?", "options": ["To reduce parameters", "Because attention alone ignores token order", "To normalise the loss", "To enable dropout"], "answer": "B", "explanation": "Without position information, attention is permutation-invariant."},
        ],
        "troubleshooting": {
            "problem": "My transformer runs out of GPU memory when I increase the input length. Why, and what can I do?",
            "solution": "Self-attention memory grows with the square of the sequence length, so doubling the length roughly quadruples the attention memory.\nFixes:\n1. Use memory-efficient attention such as FlashAttention or SDPA.\n2. Enable gradient checkpointing during training.\n3. Reduce the batch size and use gradient accumulation.\n4. Use mixed precision (bf16/fp16) or 4-bit quantisation for the frozen weights.\n5. Split long inputs into chunks, or use retrieval so only relevant parts are processed.",
        },
    },
    {
        "key": "attention",
        "name": "Attention Mechanism",
        "short": "attention",
        "domain": "Generative AI",
        "question": "What is the attention mechanism?",
        "definition": (
            "Attention is a mechanism that lets a model compute a weighted combination of "
            "information from all positions of an input, where the weights reflect how relevant "
            "each position is to the current one. In scaled dot-product attention, relevance is "
            "the similarity between a query and each key, normalised with softmax."
        ),
        "simple": (
            "Attention lets a model focus on the most relevant parts of the input. Each word asks "
            "a question (query), every word offers a label (key) and some content (value). The "
            "model compares the query with every key to decide how much attention to pay to each "
            "word, then mixes their values using those weights."
        ),
        "eli10": (
            "When your teacher asks 'Who ate the cake?' and you remember a story, you do not "
            "think about every word equally. You focus on the words about cake and eating. "
            "Attention is the computer's way of focusing on the important words and paying less "
            "attention to the rest."
        ),
        "analogy": (
            "Attention works like searching a library. Your query is what you are looking for. "
            "Each book has a key (its title and tags) and a value (its content). You compare "
            "your query with every key, spend most time with the best-matching books, and "
            "combine what you read - weighted by how relevant each book was."
        ),
        "how_it_works": [
            "Project each token embedding into a query (Q), key (K) and value (V) vector with learned matrices.",
            "Compute scores: the dot product of the query with every key.",
            "Scale by 1/sqrt(d_k) and apply softmax to get attention weights that sum to 1.",
            "Output = weighted sum of value vectors: softmax(QK^T / sqrt(d_k)) V.",
            "Self-attention uses Q, K and V from the same sequence; cross-attention takes Q from one sequence and K, V from another.",
        ],
        "example": (
            "In 'The trophy didn't fit in the suitcase because it was too big', when processing "
            "'it', the attention weights might be trophy 0.62, suitcase 0.21 and others 0.17. The "
            "representation of 'it' becomes mostly the trophy's value vector, which resolves the "
            "reference."
        ),
        "applications": [
            "Self-attention in transformers and LLMs",
            "Cross-attention in translation and text-to-image models",
            "Attention in encoder-decoder RNNs (Bahdanau attention)",
            "Attention maps for interpreting model focus",
        ],
        "advantages": [
            "Direct connections between any two positions, regardless of distance",
            "Fully parallelisable computation",
            "Weights can be inspected for some interpretability",
        ],
        "limitations": [
            "Quadratic time and memory in sequence length",
            "Attention weights are not a complete explanation of model behaviour",
            "Needs positional information to use word order",
        ],
        "key_points": [
            "Attention = weighted sum of values, with weights from query-key similarity.",
            "Formula: softmax(QK^T / sqrt(d_k)) V.",
            "Self-attention: Q, K, V from the same sequence; cross-attention: Q from another sequence.",
            "Causal masking prevents a decoder from attending to future tokens.",
            "Multi-head attention learns several types of relationships in parallel.",
        ],
        "viva": [
            ("What are Q, K and V in attention?", "Query, key and value vectors. The query is compared with keys to compute attention weights, which are then used to take a weighted sum of the values."),
            ("What is causal masking?", "Setting attention scores for future positions to minus infinity before softmax, so a token can only attend to itself and earlier tokens. It is used in text generation."),
            ("What is the difference between self-attention and cross-attention?", "In self-attention, queries, keys and values come from the same sequence. In cross-attention, queries come from one sequence (e.g. the decoder) and keys and values from another (e.g. the encoder)."),
            ("Why is softmax used in attention?", "To turn raw similarity scores into positive weights that sum to 1."),
        ],
        "interview": [
            ("What is the computational complexity of self-attention and why does it matter?", "For sequence length n and dimension d, computing QK^T costs O(n^2 * d) time and needs O(n^2) memory for the attention matrix. This limits context length and makes long documents expensive, which motivated FlashAttention, sparse attention and linear-attention variants."),
            ("What is the KV cache in LLM inference?", "During generation, keys and values of previous tokens do not change, so they are stored and reused rather than recomputed at every step. This makes each new token cost O(n) instead of recomputing the whole sequence, at the price of memory that grows with context length."),
        ],
        "mcqs": [
            {"q": "In scaled dot-product attention, the scores are divided by:", "options": ["d_k", "sqrt(d_k)", "n^2", "The batch size"], "answer": "B", "explanation": "Scaling by sqrt(d_k) keeps softmax from saturating."},
            {"q": "Which prevents a GPT-style decoder from looking at future tokens?", "options": ["Dropout", "Causal mask", "Layer normalisation", "Pooling"], "answer": "B", "explanation": "The causal mask blocks attention to later positions."},
        ],
        "numerical": {
            "problem": "A query q = (1, 0) attends to two keys k1 = (1, 0) and k2 = (0, 1) with values v1 = (10, 0) and v2 = (0, 10). With d_k = 2, compute the attention output.",
            "solution": "Step 1: Scores: q.k1 = 1, q.k2 = 0.\nStep 2: Scale by sqrt(2) = 1.414: 0.707 and 0.\nStep 3: Softmax: e^0.707 = 2.028, e^0 = 1; weights = 2.028/3.028 = 0.67 and 1/3.028 = 0.33.\nStep 4: Output = 0.67 * (10, 0) + 0.33 * (0, 10) = (6.7, 3.3).\nAnswer: The output is about (6.7, 3.3); the query attends more to k1, which it matches.",
        },
    },
    {
        "key": "llm",
        "name": "Large Language Model (LLM)",
        "short": "LLM",
        "domain": "Generative AI",
        "question": "What is a Large Language Model (LLM)?",
        "definition": (
            "A Large Language Model is a transformer-based neural network with billions of "
            "parameters, pre-trained on massive text corpora to predict the next token. This "
            "simple objective at scale gives it broad abilities in understanding and generating "
            "language, and it is usually further aligned with instruction tuning and preference "
            "optimisation to follow user instructions."
        ),
        "simple": (
            "An LLM is a very large neural network that has read an enormous amount of text and "
            "learned to predict the next word (token). By doing this billions of times, it picks "
            "up grammar, facts and reasoning patterns. After pre-training, it is fine-tuned to "
            "follow instructions and be helpful, which turns it into an assistant like ChatGPT "
            "or Claude."
        ),
        "eli10": (
            "Imagine playing a game where someone says half a sentence and you guess the next "
            "word. If you had read millions of books, you would become amazingly good at it. An "
            "LLM is a computer program that played this guessing game on a huge amount of text, "
            "so now it can write stories, answer questions and help with homework."
        ),
        "analogy": (
            "An LLM is like someone who has read an entire library and developed a powerful "
            "sense of what text should come next. Pre-training is the reading. Instruction "
            "tuning is like then training them as a helpful librarian who answers visitors' "
            "questions in a useful way, rather than just continuing whatever sentence they hear."
        ),
        "how_it_works": [
            "Tokenise text into sub-word tokens.",
            "Pre-train a decoder-only transformer on trillions of tokens to predict the next token (self-supervised learning).",
            "Supervised fine-tuning (instruction tuning) on instruction-response pairs.",
            "Align with human preferences using RLHF or DPO.",
            "Inference: generate text one token at a time by sampling from the predicted distribution (temperature, top-p).",
        ],
        "example": (
            "Given the prompt 'Explain photosynthesis to a 10-year-old', an instruction-tuned LLM "
            "predicts tokens one after another: 'Plants' -> 'are' -> 'like' -> 'tiny' -> "
            "'chefs' ... producing a full answer. Each token is chosen from a probability "
            "distribution conditioned on the prompt and everything generated so far."
        ),
        "applications": [
            "Chatbots and virtual assistants",
            "Code generation and explanation",
            "Summarisation, translation and writing assistance",
            "Question answering over documents (retrieval-augmented generation)",
        ],
        "advantages": [
            "One model handles many tasks through prompting",
            "Strong few-shot and zero-shot abilities",
            "Can be specialised cheaply with fine-tuning methods like LoRA",
        ],
        "limitations": [
            "Hallucination: can produce confident but false statements",
            "Knowledge cutoff and possible bias from training data",
            "Expensive to train and serve; limited context window",
        ],
        "key_points": [
            "LLMs are large transformer models trained to predict the next token.",
            "Pipeline: pre-training -> supervised fine-tuning -> preference alignment (RLHF/DPO).",
            "Generation is autoregressive, controlled by temperature and top-p sampling.",
            "Main risk: hallucination, so important facts need verification.",
            "Parameter-efficient methods (LoRA, QLoRA) make adapting LLMs affordable.",
        ],
        "viva": [
            ("What is the pre-training objective of GPT-style LLMs?", "Next-token prediction: given the previous tokens, predict the next token, trained with cross-entropy loss on a huge text corpus."),
            ("What is hallucination in LLMs?", "When the model generates fluent, confident text that is factually incorrect or not supported by its input."),
            ("What is a token?", "A unit of text the model processes, usually a word or part of a word produced by a tokenizer such as BPE."),
            ("What does the temperature parameter control?", "The randomness of sampling. Low temperature makes outputs more deterministic; high temperature makes them more diverse."),
            ("What is a context window?", "The maximum number of tokens the model can take into account at once, including the prompt and generated text."),
        ],
        "interview": [
            ("How would you reduce hallucinations in an LLM application?", "Ground the model with retrieval-augmented generation so answers come from trusted documents, instruct it to cite sources and to say when it does not know, lower the temperature for factual tasks, fine-tune on high-quality domain data, and add verification steps or human review for high-stakes outputs. I would also build an evaluation set to measure factual accuracy."),
            ("What is the difference between a base model and an instruction-tuned model?", "A base model is only pre-trained to continue text, so a question may get another question or a random continuation. An instruction-tuned model has been fine-tuned on instruction-response examples (and often preference data), so it follows instructions, answers in a helpful format and adopts a consistent style."),
        ],
        "mcqs": [
            {"q": "GPT-style LLMs are pre-trained to:", "options": ["Classify images", "Predict the next token", "Cluster documents", "Maximise a reward from a game"], "answer": "B", "explanation": "Next-token prediction on large text corpora is the pre-training objective."},
            {"q": "An LLM confidently stating a false fact is called:", "options": ["Overfitting", "Hallucination", "Tokenisation", "Quantisation"], "answer": "B", "explanation": "Hallucination is fluent but unsupported or incorrect output."},
        ],
        "troubleshooting": {
            "problem": "My LLM chatbot sometimes makes up facts. How can I reduce this?",
            "solution": "Hallucinations happen because the model generates plausible text, not verified facts. To reduce them:\n1. Use retrieval-augmented generation (RAG) to give the model trusted source documents.\n2. Instruct it to answer only from the provided context and to say 'I don't know' otherwise.\n3. Lower the temperature for factual questions.\n4. Fine-tune on high-quality, domain-specific examples.\n5. Add citations and verification, and keep a human in the loop for important answers.",
        },
    },
    {
        "key": "prompt_engineering",
        "name": "Prompt Engineering",
        "short": "prompt engineering",
        "domain": "Generative AI",
        "question": "What is prompt engineering?",
        "definition": (
            "Prompt engineering is the practice of designing the input text (prompt) given to a "
            "language model so that it produces the desired output, without changing the "
            "model's weights. Techniques include clear instructions, role prompts, few-shot "
            "examples, chain-of-thought prompting and output format constraints."
        ),
        "simple": (
            "Prompt engineering means asking the AI in the right way. The same model can give a "
            "vague answer or an excellent one depending on how you phrase the request. Being "
            "specific, giving examples, specifying the format and asking it to think step by "
            "step all help. Importantly, prompt engineering does not train the model; it only "
            "changes the input."
        ),
        "eli10": (
            "If you ask a friend 'Tell me something', you might get anything. If you say 'Tell "
            "me a funny joke about cats in two sentences', you get exactly what you want. Prompt "
            "engineering is learning how to ask AI clear questions so it gives better answers."
        ),
        "analogy": (
            "Prompt engineering is like briefing a very capable freelancer who knows nothing "
            "about your project. A one-line brief gets a generic result; a brief with the goal, "
            "audience, examples of what you like and the required format gets exactly what you "
            "need. The freelancer's skills do not change - only your instructions do."
        ),
        "how_it_works": [
            "State the task clearly and specifically, including the audience and goal.",
            "Give a role or context (e.g. 'You are a patient ML tutor').",
            "Provide few-shot examples of the desired input-output behaviour.",
            "Ask for step-by-step reasoning (chain-of-thought) for complex problems.",
            "Specify the output format (bullets, JSON, word limit) and iterate based on results.",
        ],
        "example": (
            "Weak prompt: 'Explain CNN.' Better prompt: 'You are a tutor for first-year students. "
            "Explain what a CNN is in under 150 words, use one real-world analogy, and end with "
            "two quiz questions.' The second prompt produces a focused, well-structured answer "
            "from the same model."
        ),
        "applications": [
            "Building chatbots and assistants with system prompts",
            "Few-shot classification and information extraction",
            "Generating structured outputs such as JSON",
            "Rapid prototyping before deciding whether fine-tuning is needed",
        ],
        "advantages": [
            "No training, data or GPU required",
            "Fast to iterate and easy to change",
            "Works with any capable model, including closed APIs",
        ],
        "limitations": [
            "Behaviour can be brittle - small wording changes may change results",
            "Long prompts use context and increase cost on every request",
            "Cannot teach the model genuinely new skills or deep domain style as reliably as fine-tuning",
        ],
        "key_points": [
            "Prompt engineering changes the input, not the model's weights.",
            "Techniques: clear instructions, roles, few-shot examples, chain-of-thought, format constraints.",
            "Zero-shot = no examples; few-shot = a few examples in the prompt.",
            "Try prompting first; fine-tune when you need consistent behaviour at scale.",
            "Prompt injection is a security risk in LLM applications.",
        ],
        "viva": [
            ("What is the difference between zero-shot and few-shot prompting?", "Zero-shot prompting gives only the instruction; few-shot prompting also includes a few worked examples of the task in the prompt."),
            ("What is chain-of-thought prompting?", "Asking the model to reason step by step before giving the final answer, which improves performance on multi-step problems."),
            ("Does prompt engineering change the model's weights?", "No. It only changes the input text; the model parameters stay the same."),
            ("What is a system prompt?", "An instruction given to a chat model that sets its role, behaviour and constraints for the whole conversation."),
        ],
        "interview": [
            ("When would you choose fine-tuning over prompt engineering?", "Prompting is the first step because it is cheap and fast. I would fine-tune when prompts get very long or brittle, when I need a consistent style or format at scale, when the task needs specialised domain behaviour the model does not show from prompting, or when I want a smaller, cheaper model to match a larger one on a narrow task. Parameter-efficient fine-tuning like LoRA keeps the cost low."),
            ("What is prompt injection and how do you defend against it?", "Prompt injection is when untrusted input, like a web page or user message, contains instructions that override the developer's intent. Defences include separating trusted instructions from untrusted data, limiting tool permissions, validating outputs, filtering inputs and requiring human confirmation for sensitive actions."),
        ],
        "mcqs": [
            {"q": "Providing a few input-output examples inside the prompt is called:", "options": ["Fine-tuning", "Few-shot prompting", "Quantisation", "Pre-training"], "answer": "B", "explanation": "Few-shot prompting shows examples without updating weights."},
            {"q": "Which statement about prompt engineering is true?", "options": ["It updates model weights", "It changes only the input to the model", "It requires a GPU for training", "It needs labelled datasets"], "answer": "B", "explanation": "Prompting guides a frozen model through its input."},
        ],
    },
    {
        "key": "fine_tuning",
        "name": "Fine-Tuning",
        "short": "fine-tuning",
        "domain": "Generative AI",
        "question": "What is fine-tuning in machine learning?",
        "definition": (
            "Fine-tuning is the process of taking a model that was pre-trained on a large, "
            "general dataset and continuing to train it on a smaller, task-specific or "
            "domain-specific dataset so that it adapts to that task. It is a form of transfer "
            "learning."
        ),
        "simple": (
            "Training a large model from scratch is extremely expensive. Fine-tuning starts from "
            "a model that already knows a lot and teaches it something specific with a much "
            "smaller dataset. For example, a general language model can be fine-tuned on "
            "teaching examples so it answers like a patient tutor."
        ),
        "eli10": (
            "Imagine a kid who already knows how to play football. If they join a basketball "
            "team, they do not need to learn running and teamwork from zero - they just practise "
            "basketball skills. Fine-tuning is like that: the model already knows a lot, and we "
            "only teach it the new, special skill."
        ),
        "analogy": (
            "Fine-tuning is like a qualified doctor completing a specialisation in cardiology. "
            "They do not go back to primary school; they build on years of general medical "
            "training with focused practice in one area."
        ),
        "how_it_works": [
            "Start from pre-trained weights, e.g. an LLM pre-trained on web text.",
            "Prepare a task-specific dataset, e.g. instruction-response pairs.",
            "Train with a small learning rate so general knowledge is not destroyed.",
            "Choose what to update: all parameters (full fine-tuning) or a small subset or adapters (PEFT: LoRA, QLoRA).",
            "Evaluate on held-out data and compare with the original model.",
        ],
        "example": (
            "This project, EduTune AI, fine-tunes Qwen2.5-1.5B-Instruct on a few hundred "
            "educational instruction-response examples. Only small LoRA adapters are trained, "
            "while the base model stays frozen in 4-bit precision. The result answers AI/ML "
            "questions in a consistent teaching style with analogies, structure and viva-style "
            "answers."
        ),
        "applications": [
            "Specialising LLMs for domains like medicine, law or education",
            "Transfer learning for image classification with small datasets",
            "Teaching a model a specific output format or tone",
            "Adapting speech models to new accents or vocabulary",
        ],
        "advantages": [
            "Needs far less data and compute than training from scratch",
            "Often gives better task performance than prompting alone",
            "Produces consistent behaviour without long prompts",
        ],
        "limitations": [
            "Risk of catastrophic forgetting of general abilities",
            "Can overfit a small dataset",
            "Full fine-tuning of large models needs a lot of GPU memory (solved by PEFT)",
        ],
        "key_points": [
            "Fine-tuning adapts a pre-trained model to a specific task or domain.",
            "It is a form of transfer learning.",
            "Full fine-tuning updates all weights; PEFT updates only a small number of parameters.",
            "Use a low learning rate and validation monitoring to avoid forgetting and overfitting.",
            "Instruction tuning is fine-tuning on instruction-response pairs.",
        ],
        "viva": [
            ("What is the difference between pre-training and fine-tuning?", "Pre-training learns general knowledge from a huge dataset; fine-tuning adapts the pre-trained model to a specific task using a much smaller dataset."),
            ("What is catastrophic forgetting?", "When fine-tuning on a new task causes the model to lose abilities it had learned before."),
            ("Why is a small learning rate used in fine-tuning?", "To make gentle updates that adapt the model without destroying the useful knowledge in the pre-trained weights."),
            ("What is the difference between full fine-tuning and PEFT?", "Full fine-tuning updates every parameter; PEFT freezes the base model and trains only a small number of added or selected parameters."),
        ],
        "interview": [
            ("How would you fine-tune an LLM on a single 8 GB GPU?", "Use QLoRA: load the base model in 4-bit NF4, freeze it, and train LoRA adapters on the attention and MLP projections. Add gradient checkpointing, a small micro-batch with gradient accumulation, a moderate sequence length and bf16 compute. This fits models of a few billion parameters on 8 GB, and the result is a small adapter file that is loaded on top of the base model."),
            ("How do you know whether fine-tuning actually helped?", "Compare the base and fine-tuned models on the same held-out set: validation loss and perplexity on in-domain responses, task-specific metrics where they exist, and a qualitative side-by-side comparison of outputs for instruction following and style. I would also check general-ability benchmarks to catch catastrophic forgetting."),
        ],
        "mcqs": [
            {"q": "Fine-tuning is a form of:", "options": ["Unsupervised clustering", "Transfer learning", "Data augmentation", "Feature scaling"], "answer": "B", "explanation": "Fine-tuning transfers knowledge from a pre-trained model to a new task."},
            {"q": "Losing previously learned abilities after fine-tuning is called:", "options": ["Mode collapse", "Catastrophic forgetting", "Vanishing gradient", "Underfitting"], "answer": "B", "explanation": "Catastrophic forgetting happens when new training overwrites old knowledge."},
        ],
        "troubleshooting": {
            "problem": "After fine-tuning, my model is great at my task but worse at general questions. What happened?",
            "solution": "This is catastrophic forgetting. To reduce it:\n1. Use parameter-efficient fine-tuning (LoRA/QLoRA) so the base weights stay frozen.\n2. Lower the learning rate and train for fewer epochs.\n3. Mix some general-purpose instruction data into the training set.\n4. Monitor validation loss and stop early.\n5. With LoRA, you can simply disable the adapter to get the original model back.",
        },
    },
    {
        "key": "lora",
        "name": "LoRA (Low-Rank Adaptation)",
        "short": "LoRA",
        "domain": "Generative AI",
        "question": "What is LoRA?",
        "definition": (
            "LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method that freezes "
            "the pre-trained weights and injects small trainable low-rank matrices into selected "
            "layers. Instead of updating a large weight matrix W, it learns an update Delta W = "
            "B x A, where A and B have a small rank r, so the layer computes W x + (alpha / r) "
            "B A x."
        ),
        "simple": (
            "Fine-tuning every weight in a large model is expensive. LoRA keeps the original "
            "weights frozen and adds two small matrices next to certain layers. Only these small "
            "matrices are trained. Together they represent the change the model needs. This "
            "often means training less than 1% of the parameters, and the result is a small "
            "adapter file that sits on top of the base model."
        ),
        "eli10": (
            "Imagine a huge, finished painting you are not allowed to repaint. Instead, you put "
            "a thin, clear sticker sheet on top and draw small changes on it. The painting now "
            "looks different, but the original is untouched, and you can peel the sticker off "
            "whenever you want. LoRA is that sticker sheet for an AI model."
        ),
        "analogy": (
            "LoRA is like adding a small set of adjustment knobs to a huge, finely tuned "
            "machine. Instead of rebuilding every gear, you add a compact attachment that shifts "
            "its behaviour for your task. You can swap attachments for different tasks while "
            "keeping one copy of the machine."
        ),
        "how_it_works": [
            "Freeze the pre-trained weight matrix W (size d x k).",
            "Add two trainable matrices: A (r x k) and B (d x r), with rank r much smaller than d and k (e.g. r = 16).",
            "Initialise B to zero so training starts exactly from the original model.",
            "Forward pass: h = W x + (alpha / r) * B A x; only A and B receive gradients.",
            "After training, save only A and B (the adapter), or optionally merge them into W for zero extra latency.",
        ],
        "example": (
            "A 4096 x 4096 attention projection has 16.8 million weights. With LoRA rank r = 8, A "
            "is 8 x 4096 and B is 4096 x 8, which is only 65,536 trainable parameters, about "
            "0.39% of the original. In this project, LoRA adapters on the attention and MLP "
            "projections of Qwen2.5-1.5B train roughly 1% of the model's parameters."
        ),
        "applications": [
            "Fine-tuning LLMs for domains and styles (like this teaching assistant)",
            "Customising Stable Diffusion to new styles or characters",
            "Serving many task-specific adapters on one shared base model",
            "Low-cost experimentation on consumer GPUs",
        ],
        "advantages": [
            "Trains a tiny fraction of the parameters, so memory and compute drop",
            "Adapters are small files (megabytes, not gigabytes) and easy to swap",
            "No extra inference latency if merged; base model stays intact",
        ],
        "limitations": [
            "May not match full fine-tuning on tasks that need large changes",
            "Rank, alpha and target modules need tuning",
            "Still needs the full base model in memory (QLoRA reduces this with 4-bit)",
        ],
        "key_points": [
            "LoRA freezes W and learns a low-rank update Delta W = B A.",
            "Rank r controls capacity; alpha / r scales the update.",
            "B is initialised to zero so the model starts unchanged.",
            "Trainable parameters per layer = r * (d + k) instead of d * k.",
            "Typically applied to attention (q, k, v, o) and MLP projections.",
        ],
        "viva": [
            ("What does 'low-rank' mean in LoRA?", "The weight update is represented as the product of two thin matrices B (d x r) and A (r x k) with a small rank r, which needs far fewer parameters than a full d x k update."),
            ("Why is matrix B initialised to zero in LoRA?", "So that B A = 0 at the start and the model begins exactly as the pre-trained model; training then learns the change."),
            ("What is the role of lora_alpha?", "It scales the LoRA update by alpha / r, controlling how strongly the adapter changes the layer's output."),
            ("Which layers is LoRA usually applied to?", "The linear projections in attention (query, key, value, output) and often the MLP (gate, up, down) projections."),
            ("Can a LoRA adapter be removed after training?", "Yes. The base weights are unchanged, so the adapter can be disabled, swapped or merged."),
        ],
        "interview": [
            ("Explain LoRA and why it works.", "LoRA is based on the observation that the weight change needed to adapt a pre-trained model has a low intrinsic rank. So instead of learning a full update to a d x k matrix, LoRA learns B (d x r) and A (r x k) with small r, keeps W frozen, and computes W x + (alpha / r) B A x. This cuts trainable parameters by orders of magnitude, reduces optimiser memory, produces small swappable adapters and, once merged, adds no inference latency."),
            ("How do you choose the LoRA rank?", "Start with r = 8 or 16 and alpha about 2r. Higher ranks add capacity for harder tasks or larger datasets, but increase memory and can overfit small datasets. Applying LoRA to more modules, such as all linear layers, often helps more than raising the rank. Pick based on validation loss."),
        ],
        "mcqs": [
            {"q": "In LoRA, which weights are trained?", "options": ["All model weights", "Only the low-rank matrices A and B", "Only the embedding layer", "Only the output layer bias"], "answer": "B", "explanation": "The pre-trained weights are frozen; only the adapter matrices are updated."},
            {"q": "LoRA represents the weight update as:", "options": ["W^2", "B x A with small rank r", "A random matrix", "The inverse of W"], "answer": "B", "explanation": "Delta W = B A is a low-rank factorisation."},
            {"q": "Why is LoRA memory-efficient?", "options": ["It deletes layers", "It trains far fewer parameters, so gradients and optimiser states are small", "It uses no GPU", "It shortens the input"], "answer": "B", "explanation": "Optimiser states and gradients are only needed for the small adapter matrices."},
        ],
        "numerical": {
            "problem": "A linear layer has a 2048 x 2048 weight matrix. How many trainable parameters does LoRA add with rank r = 16, and what percentage of the original layer is that?",
            "solution": "Step 1: Original parameters = 2048 * 2048 = 4,194,304.\nStep 2: A is 16 x 2048 = 32,768 parameters; B is 2048 x 16 = 32,768 parameters.\nStep 3: LoRA parameters = 32,768 + 32,768 = 65,536.\nStep 4: Percentage = 65,536 / 4,194,304 * 100 = 1.5625%.\nAnswer: 65,536 trainable parameters, about 1.56% of the layer.",
        },
        "troubleshooting": {
            "problem": "I trained a LoRA adapter but the model's answers look exactly the same as before. What could be wrong?",
            "solution": "Check these common causes:\n1. The adapter is not loaded - confirm you load it with PeftModel.from_pretrained and that the adapter is enabled.\n2. Wrong target modules - the names in target_modules must match the model's layer names, or no layers are adapted.\n3. Learning rate too low or too few steps - LoRA typically uses around 1e-4 to 2e-4.\n4. Prompt format mismatch - use the same chat template at inference as in training.\n5. Check that training loss actually decreased in the logs.",
        },
    },
    {
        "key": "qlora",
        "name": "QLoRA (Quantized LoRA)",
        "short": "QLoRA",
        "domain": "Generative AI",
        "question": "What is QLoRA?",
        "definition": (
            "QLoRA (Quantized LoRA) is a fine-tuning method that loads the frozen pre-trained "
            "model in 4-bit precision and trains LoRA adapters in higher precision on top of it. "
            "It introduced the 4-bit NormalFloat (NF4) data type, double quantisation and paged "
            "optimisers, making it possible to fine-tune large models on a single consumer GPU."
        ),
        "simple": (
            "QLoRA is LoRA plus compression. The big base model is squeezed into 4-bit numbers, "
            "which uses about a quarter of the memory of 16-bit weights, and it stays frozen. "
            "Small LoRA adapters are trained on top in 16-bit precision. During the forward pass "
            "the 4-bit weights are temporarily dequantised for computation. The result is "
            "fine-tuning quality close to 16-bit LoRA at a fraction of the memory."
        ),
        "eli10": (
            "Imagine you need to carry a huge encyclopedia on a trip, but your bag is small. You "
            "shrink the whole book into a tiny, pocket-sized version that is still readable, and "
            "then write your own notes on a few sticky notes. The tiny book is the 4-bit model, "
            "and the sticky notes are the LoRA adapters you learn."
        ),
        "analogy": (
            "QLoRA is like keeping a compressed ZIP archive of a huge reference library (the "
            "4-bit base model) and only unzipping the page you need at the moment you read it. "
            "Your own annotations (LoRA adapters) are written in full detail on separate sheets. "
            "You save enormous shelf space while still reading accurately."
        ),
        "how_it_works": [
            "Quantise the frozen base model weights to 4-bit NF4, a data type designed for normally distributed weights.",
            "Double quantisation: also quantise the per-block scaling constants to save more memory.",
            "Add LoRA adapters (kept in bf16/fp16) to the target layers; only they are trained.",
            "Forward/backward: 4-bit weights are dequantised on the fly to the compute dtype; gradients flow into the adapters only.",
            "Paged optimisers avoid out-of-memory spikes; save only the adapter at the end.",
        ],
        "example": (
            "A 1.5-billion-parameter model needs about 3 GB for its weights in 16-bit precision "
            "but only around 1 GB in 4-bit. In this project, QLoRA loads Qwen2.5-1.5B-Instruct in "
            "4-bit NF4 on an 8 GB laptop GPU, trains LoRA adapters with rank 16 on all "
            "attention and MLP projections, and saves an adapter of a few tens of megabytes."
        ),
        "applications": [
            "Fine-tuning 7B-70B LLMs on single GPUs",
            "Academic projects and prototyping on laptops",
            "Cost-efficient domain adaptation in industry",
            "Training many task-specific adapters cheaply",
        ],
        "advantages": [
            "About 4x less memory for base weights than 16-bit",
            "Quality close to 16-bit LoRA fine-tuning",
            "Makes LLM fine-tuning accessible on consumer hardware",
        ],
        "limitations": [
            "Slower per step than 16-bit LoRA because of dequantisation",
            "Needs a CUDA GPU and the bitsandbytes library",
            "Merging adapters back into a 4-bit model needs care (dequantise first)",
        ],
        "key_points": [
            "QLoRA = 4-bit quantised frozen base model + trainable LoRA adapters.",
            "Innovations: NF4 data type, double quantisation, paged optimisers.",
            "Computation happens in bf16/fp16 after on-the-fly dequantisation.",
            "Only the adapters are trained and saved.",
            "Introduced by Dettmers et al., 2023; implemented via bitsandbytes and PEFT.",
        ],
        "viva": [
            ("What is the difference between LoRA and QLoRA?", "LoRA trains low-rank adapters on a frozen base model kept in 16- or 32-bit precision. QLoRA does the same but loads the frozen base model in 4-bit precision, greatly reducing memory."),
            ("What is NF4?", "4-bit NormalFloat, a data type introduced by QLoRA whose quantisation levels are optimised for normally distributed neural network weights."),
            ("What is double quantisation?", "Quantising the quantisation constants (per-block scales) themselves, which saves roughly another 0.4 bits per parameter."),
            ("Are the LoRA adapters in QLoRA also 4-bit?", "No. The adapters are kept and trained in higher precision such as bf16; only the frozen base model is 4-bit."),
            ("Which library provides the 4-bit quantisation in QLoRA?", "bitsandbytes, used through Hugging Face Transformers' BitsAndBytesConfig."),
        ],
        "interview": [
            ("Explain QLoRA and the three innovations it introduced.", "QLoRA fine-tunes a model by freezing it in 4-bit precision and training LoRA adapters on top. Its three innovations are: NF4, a 4-bit data type that is information-theoretically suited to normally distributed weights; double quantisation, which also quantises the scaling constants; and paged optimisers, which use NVIDIA unified memory to avoid out-of-memory errors during memory spikes. Together they let a 65B model be fine-tuned on a single 48 GB GPU with quality close to 16-bit fine-tuning."),
            ("If QLoRA stores weights in 4-bit, how can training still be accurate?", "The weights are only stored in 4-bit; for each matrix multiplication they are dequantised to bf16. The gradients flow through these dequantised weights into the LoRA adapters, which are kept in bf16. The base weights never change, so quantisation error only affects the frozen part, and the adapters can learn to compensate for it."),
        ],
        "mcqs": [
            {"q": "In QLoRA, the frozen base model is stored in:", "options": ["32-bit", "16-bit", "4-bit", "1-bit"], "answer": "C", "explanation": "QLoRA quantises the frozen base weights to 4-bit NF4."},
            {"q": "Which data type did QLoRA introduce?", "options": ["FP8", "NF4", "INT32", "BF16"], "answer": "B", "explanation": "NormalFloat4 is optimised for normally distributed weights."},
            {"q": "In QLoRA, which parameters are updated during training?", "options": ["All 4-bit weights", "Only the LoRA adapter weights", "Only the embeddings", "None"], "answer": "B", "explanation": "The quantised base model is frozen; only the adapters learn."},
        ],
        "numerical": {
            "problem": "Estimate the memory needed to store the weights of a 1.5-billion-parameter model in 16-bit precision and in 4-bit precision.",
            "solution": "Step 1: 16-bit = 2 bytes per parameter: 1.5 x 10^9 x 2 = 3.0 x 10^9 bytes, about 3 GB.\nStep 2: 4-bit = 0.5 bytes per parameter: 1.5 x 10^9 x 0.5 = 0.75 x 10^9 bytes, about 0.75 GB.\nStep 3: Quantisation constants add a little overhead, so in practice about 1 GB.\nAnswer: About 3 GB in 16-bit vs about 1 GB in 4-bit - roughly a 4x reduction for the frozen weights.",
        },
        "troubleshooting": {
            "problem": "I get a CUDA out-of-memory error while fine-tuning with QLoRA on my laptop GPU. What should I change?",
            "solution": "Reduce memory in this order:\n1. Set the micro-batch size to 1 and increase gradient accumulation steps to keep the effective batch size.\n2. Reduce the maximum sequence length (e.g. 1024 -> 512).\n3. Make sure gradient checkpointing is enabled.\n4. Confirm the model is really loaded in 4-bit (load_in_4bit=True with NF4).\n5. Lower the LoRA rank or target fewer modules.\n6. Use a smaller base model and close other programs using the GPU.",
        },
    },
    {
        "key": "peft",
        "name": "PEFT (Parameter-Efficient Fine-Tuning)",
        "short": "PEFT",
        "domain": "Generative AI",
        "question": "What is PEFT (Parameter-Efficient Fine-Tuning)?",
        "definition": (
            "Parameter-Efficient Fine-Tuning (PEFT) is a family of methods that adapt a large "
            "pre-trained model by training only a small number of parameters - added modules or "
            "a selected subset - while keeping most of the original weights frozen. Examples "
            "include LoRA, QLoRA, adapters, prefix tuning, prompt tuning and IA3. PEFT is also "
            "the name of Hugging Face's library implementing them."
        ),
        "simple": (
            "Full fine-tuning updates billions of weights, which needs huge GPUs and creates a "
            "full copy of the model for every task. PEFT methods freeze the big model and train "
            "only a tiny set of extra parameters, often under 1%. You get most of the benefit of "
            "fine-tuning for a fraction of the cost, and each task is just a small add-on file."
        ),
        "eli10": (
            "Imagine you have a big LEGO castle and want it to become a pirate castle. Instead "
            "of rebuilding it from scratch, you just add a few pirate flags and a cannon. The "
            "castle stays the same, but now it is a pirate castle. PEFT adds small pieces to a "
            "big model instead of rebuilding it."
        ),
        "analogy": (
            "PEFT is like customising a smartphone with apps instead of building a new phone "
            "for every purpose. The phone (base model) stays the same; each app (adapter) is "
            "small, cheap to create and can be installed or removed to give the phone a new "
            "capability."
        ),
        "how_it_works": [
            "Load the pre-trained model and freeze all its parameters.",
            "Add or select a small set of trainable parameters: low-rank matrices (LoRA), bottleneck adapters, learned prompt vectors (prompt/prefix tuning) or scaling vectors (IA3).",
            "Train only these parameters on the task data.",
            "Save only the small trainable part as an adapter.",
            "At inference, load the base model and attach the adapter; different adapters can be swapped.",
        ],
        "example": (
            "In EduTune AI, the Hugging Face PEFT library wraps Qwen2.5-1.5B-Instruct with a "
            "LoraConfig. The training script prints the total parameters, the trainable "
            "parameters and the trainable percentage, which comes out at around 1%. That number "
            "is the practical demonstration of 'parameter-efficient'."
        ),
        "applications": [
            "Fine-tuning LLMs on limited hardware",
            "Multi-task serving with one base model and many adapters",
            "Personalised models per customer or user",
            "Diffusion model customisation",
        ],
        "advantages": [
            "Huge reduction in trainable parameters, gradient memory and optimiser memory",
            "Small checkpoints (MBs) instead of full model copies (GBs)",
            "Less catastrophic forgetting because base weights are frozen",
        ],
        "limitations": [
            "Can underperform full fine-tuning on tasks that need large changes",
            "Extra hyper-parameters (rank, target modules, prompt length)",
            "Some methods add inference latency unless merged",
        ],
        "key_points": [
            "PEFT trains a small number of parameters and freezes the rest.",
            "Methods: LoRA, QLoRA, adapters, prefix tuning, prompt tuning, IA3.",
            "Trainable parameters are often well under 1-2% of the model.",
            "Adapters are small, portable and swappable.",
            "Hugging Face 'peft' library: LoraConfig, get_peft_model, PeftModel.from_pretrained.",
        ],
        "viva": [
            ("What does PEFT stand for?", "Parameter-Efficient Fine-Tuning."),
            ("Name three PEFT methods.", "LoRA, prefix tuning and adapter layers. Prompt tuning, IA3 and QLoRA are others."),
            ("Why does PEFT need less GPU memory than full fine-tuning?", "Gradients and optimiser states (which for Adam are about twice the size of the trained weights) are only needed for the small set of trainable parameters."),
            ("What is prompt tuning?", "A PEFT method that learns a few continuous 'soft prompt' embedding vectors prepended to the input while the model itself stays frozen."),
        ],
        "interview": [
            ("Why is full fine-tuning of a 7B model impractical on a single consumer GPU, and how does PEFT solve it?", "Full fine-tuning in mixed precision needs the weights, gradients and Adam states - roughly 16 bytes per parameter, over 100 GB for 7B. PEFT freezes the base model, so gradients and optimiser states exist only for the adapter's few million parameters. Combined with 4-bit quantisation of the frozen weights (QLoRA), a 7B model fits in about 6-10 GB."),
        ],
        "mcqs": [
            {"q": "Which of these is NOT a PEFT method?", "options": ["LoRA", "Prefix tuning", "Full fine-tuning", "Adapters"], "answer": "C", "explanation": "Full fine-tuning updates all parameters, the opposite of PEFT."},
            {"q": "In PEFT, the pre-trained model weights are usually:", "options": ["Deleted", "Frozen", "Randomly reinitialised", "Doubled"], "answer": "B", "explanation": "PEFT keeps the base weights frozen and trains a small number of parameters."},
        ],
        "numerical": {
            "problem": "A model has 1,543,714,304 total parameters and LoRA adds 18,464,768 trainable parameters. What is the trainable percentage?",
            "solution": "Step 1: Trainable percentage = trainable / total * 100.\nStep 2: The total including the adapter is 1,543,714,304 + 18,464,768 = 1,562,179,072.\nStep 3: 18,464,768 / 1,562,179,072 * 100 = 1.18%.\nAnswer: About 1.18% of the parameters are trained - the rest stay frozen.",
        },
    },
    {
        "key": "instruction_tuning",
        "name": "Instruction Tuning",
        "short": "instruction tuning",
        "domain": "Generative AI",
        "question": "What is instruction tuning?",
        "definition": (
            "Instruction tuning is supervised fine-tuning of a language model on a dataset of "
            "(instruction, optional input, desired response) examples, so that the model learns "
            "to follow natural-language instructions and respond in the desired style and "
            "format."
        ),
        "simple": (
            "A pre-trained language model only knows how to continue text. Instruction tuning "
            "teaches it to follow requests. We show it many examples like 'Instruction: Explain "
            "RNN to a beginner' paired with a good answer. After training, it responds to new "
            "instructions in the same helpful way, even for topics that were not in the "
            "examples."
        ),
        "eli10": (
            "Imagine a parrot that can repeat lots of sentences but does not understand what you "
            "want. Now you train it: every time you say 'say hello', you show it the right "
            "response. After lots of practice, it learns to do what you ask. Instruction tuning "
            "trains an AI to follow instructions in the same way."
        ),
        "analogy": (
            "Instruction tuning is like training a new teaching assistant with a folder of model "
            "answers. Each page shows a student request ('Explain this simply', 'Give five viva "
            "questions') and an ideal response. After studying the folder, the assistant "
            "responds to new requests in the same style."
        ),
        "how_it_works": [
            "Build a dataset of instruction -> response pairs, often with task categories (explanations, analogies, MCQs...).",
            "Format each example with a consistent template or the model's chat template.",
            "Fine-tune with the next-token prediction loss, usually computed only on the response tokens.",
            "Often use PEFT (LoRA/QLoRA) to keep training affordable.",
            "Evaluate on held-out instructions and compare with the base model.",
        ],
        "example": (
            "This project is an instruction-tuning example. A training record looks like: "
            "{'instruction': 'Explain LSTM using a simple real-world analogy.', 'input': '', "
            "'output': 'Imagine you are reading a long story...'}. After QLoRA training on a few "
            "hundred such examples, the model answers new teaching requests in the same "
            "structured, analogy-rich style."
        ),
        "applications": [
            "Turning base LLMs into chat assistants (e.g. FLAN, Alpaca)",
            "Domain assistants for education, medicine or law",
            "Teaching consistent output formats (MCQs, reports, JSON)",
            "Improving zero-shot generalisation to unseen tasks",
        ],
        "advantages": [
            "Greatly improves instruction following and usefulness",
            "Relatively small datasets (hundreds to thousands of examples) can shape style",
            "Generalises to instructions not seen during training",
        ],
        "limitations": [
            "Quality depends heavily on the quality of the examples",
            "Does not add much new factual knowledge; it mainly teaches behaviour",
            "Can reduce diversity or overfit to the dataset's style",
        ],
        "key_points": [
            "Instruction tuning = supervised fine-tuning on instruction -> response pairs.",
            "Data format: instruction, optional input, output.",
            "Loss is usually computed only on the response tokens.",
            "It teaches the model how to respond, more than what facts to know.",
            "Often followed by preference alignment (RLHF or DPO).",
        ],
        "viva": [
            ("What is the format of an instruction-tuning dataset?", "Each example has an instruction, an optional input giving extra context, and the desired output (response)."),
            ("How is instruction tuning different from pre-training?", "Pre-training learns next-token prediction on raw text; instruction tuning fine-tunes on curated instruction-response pairs to teach the model to follow instructions."),
            ("Why is loss often computed only on the response?", "Because we want the model to learn to generate good responses, not to reproduce the prompts."),
            ("Is instruction tuning supervised or unsupervised?", "Supervised: every example includes the desired response."),
            ("Name a well-known instruction-tuned model or dataset.", "FLAN-T5, Alpaca and Dolly are well-known examples."),
        ],
        "interview": [
            ("What makes a good instruction-tuning dataset?", "High-quality, correct and consistent responses; diverse instruction phrasings and task types; a clear target style; no duplicates or near-duplicates; balanced categories; and a held-out validation split. Quality matters more than quantity - a few thousand excellent examples can beat a large noisy dataset."),
            ("Describe how you instruction-tuned a model in your project.", "I built a dataset of educational instruction-response pairs across AI, ML, DL and generative AI topics in categories like simple explanations, analogies, viva, interview and MCQs. I cleaned and de-duplicated it and split it 90/10. Then I fine-tuned Qwen2.5-1.5B-Instruct with QLoRA - a 4-bit frozen base model plus LoRA adapters - using TRL's SFTTrainer with loss on response tokens only. Finally I compared the base and fine-tuned models on validation loss, perplexity and side-by-side answers."),
        ],
        "mcqs": [
            {"q": "Instruction tuning trains a model on:", "options": ["Unlabelled raw text only", "Instruction-response pairs", "Reward signals from games", "Image-label pairs"], "answer": "B", "explanation": "Each training example pairs an instruction with the desired response."},
            {"q": "Instruction tuning is a form of:", "options": ["Supervised fine-tuning", "Clustering", "Reinforcement learning from scratch", "Data augmentation"], "answer": "A", "explanation": "It is supervised because the desired outputs are provided."},
        ],
    },
    {
        "key": "rlhf",
        "name": "RLHF (Reinforcement Learning from Human Feedback)",
        "short": "RLHF",
        "domain": "Generative AI",
        "question": "What is RLHF?",
        "definition": (
            "Reinforcement Learning from Human Feedback (RLHF) is a method for aligning language "
            "models with human preferences. Humans rank model responses, a reward model is "
            "trained to predict those preferences, and the language model is then optimised "
            "with reinforcement learning (typically PPO) to maximise the reward while staying "
            "close to the original model."
        ),
        "simple": (
            "After instruction tuning, a model can follow instructions, but its answers are not "
            "always the most helpful or safe. In RLHF, people compare pairs of answers and say "
            "which is better. A reward model learns to score answers like those people would. "
            "The language model is then trained to produce answers that get high scores."
        ),
        "eli10": (
            "Imagine training a puppy. When it does a trick well, it gets a treat; when it does "
            "something unwanted, it does not. Over time it learns what you like. In RLHF, "
            "people's preferences become the 'treats' that teach the AI which answers are good."
        ),
        "analogy": (
            "RLHF is like a chef improving a menu with customer feedback cards. Customers taste "
            "two versions of a dish and tick which they prefer. A head waiter learns to predict "
            "customers' preferences (reward model), and the chef keeps adjusting recipes to "
            "please the head waiter - without drifting so far that the dishes become unrecognisable "
            "(the KL penalty)."
        ),
        "how_it_works": [
            "Start from a supervised fine-tuned (instruction-tuned) model.",
            "Collect human comparisons: for a prompt, labellers rank two or more model responses.",
            "Train a reward model to predict which response humans prefer.",
            "Optimise the language model with reinforcement learning (PPO) to maximise the reward.",
            "Add a KL-divergence penalty to keep the model close to the SFT model and prevent reward hacking.",
        ],
        "example": (
            "For the prompt 'How do I fix a leaking tap?', the model produces two answers: A is "
            "clear with safety steps, B is vague. Labellers prefer A. After thousands of such "
            "comparisons, the reward model scores clear, safe answers highly, and PPO training "
            "pushes the language model to produce answers like A. This was a key step in "
            "InstructGPT and ChatGPT."
        ),
        "applications": [
            "Aligning chat assistants to be helpful, honest and harmless",
            "Reducing toxic or unsafe outputs",
            "Improving summarisation quality",
            "Tuning style, such as conciseness or politeness",
        ],
        "advantages": [
            "Optimises for what humans actually prefer, which is hard to write as a loss",
            "Significantly improved helpfulness and safety of chat models",
            "Learns from comparisons, which are easier for humans than writing ideal answers",
        ],
        "limitations": [
            "Complex and unstable pipeline: SFT, reward model and PPO",
            "Expensive human labelling; labellers may disagree or be biased",
            "Reward hacking: the model can exploit weaknesses in the reward model",
        ],
        "key_points": [
            "RLHF stages: SFT -> reward model from human comparisons -> RL optimisation (PPO).",
            "The reward model predicts human preference scores.",
            "A KL penalty keeps the policy close to the SFT model.",
            "Used in InstructGPT and ChatGPT.",
            "DPO is a simpler alternative that skips the explicit reward model and RL loop.",
        ],
        "viva": [
            ("What are the three main stages of RLHF?", "Supervised fine-tuning, training a reward model on human preference comparisons, and optimising the policy with reinforcement learning such as PPO."),
            ("What is a reward model?", "A model trained on human comparisons to output a score predicting how much humans would prefer a given response."),
            ("Why is a KL penalty used in RLHF?", "To stop the policy drifting too far from the supervised model, which prevents reward hacking and keeps outputs fluent."),
            ("Which RL algorithm is commonly used in RLHF?", "Proximal Policy Optimisation (PPO)."),
        ],
        "interview": [
            ("What is reward hacking and how is it mitigated in RLHF?", "Reward hacking is when the policy finds outputs that score highly on the learned reward model without actually being better - for example, overly long or flattering answers. Mitigations include a KL penalty to the reference model, improving and regularly retraining the reward model with fresh comparisons, reward model ensembles and human evaluation of the final policy."),
        ],
        "mcqs": [
            {"q": "In RLHF, the reward model is trained on:", "options": ["Raw web text", "Human preference comparisons", "Image labels", "Random noise"], "answer": "B", "explanation": "Humans rank responses, and the reward model learns to predict those rankings."},
            {"q": "Which algorithm is typically used for the RL step of RLHF?", "options": ["K-Means", "PPO", "Apriori", "Naive Bayes"], "answer": "B", "explanation": "Proximal Policy Optimisation is the standard choice."},
        ],
    },
    {
        "key": "dpo",
        "name": "DPO (Direct Preference Optimization)",
        "short": "DPO",
        "domain": "Generative AI",
        "question": "What is DPO (Direct Preference Optimization)?",
        "definition": (
            "Direct Preference Optimization (DPO) is a method for aligning language models with "
            "human preferences directly from preference pairs (prompt, chosen response, rejected "
            "response) using a simple classification-style loss. It removes the need for a "
            "separate reward model and a reinforcement learning loop."
        ),
        "simple": (
            "DPO does the same job as RLHF - making the model prefer better answers - but in a "
            "much simpler way. For each prompt we have a preferred answer and a worse answer. "
            "DPO directly trains the model to increase the probability of the preferred answer "
            "relative to the worse one, compared with a frozen reference model. No reward model "
            "and no PPO are needed."
        ),
        "eli10": (
            "Imagine a teacher showing you two drawings and saying 'this one is better than that "
            "one'. You do not need a scorecard; you just learn to draw more like the good one "
            "and less like the bad one. DPO teaches an AI by showing it pairs of good and bad "
            "answers."
        ),
        "analogy": (
            "If RLHF is hiring a food critic (reward model) and then repeatedly adjusting recipes "
            "to please the critic, DPO is skipping the critic and learning directly from customer "
            "choices: 'more of dish A, less of dish B'. Same goal, fewer moving parts."
        ),
        "how_it_works": [
            "Start from an SFT model; keep a frozen copy as the reference model.",
            "Collect preference data: (prompt x, chosen response y_w, rejected response y_l).",
            "Compute log-probability ratios of each response under the policy vs the reference.",
            "Loss = -log sigmoid(beta * [(log pi(y_w|x) - log ref(y_w|x)) - (log pi(y_l|x) - log ref(y_l|x))]).",
            "Minimising it raises the chosen response's likelihood relative to the rejected one; beta controls how far the model may move from the reference.",
        ],
        "example": (
            "Prompt: 'Explain overfitting.' Chosen: a clear explanation with an example and two "
            "remedies. Rejected: a one-line vague answer. DPO updates the model so the chosen "
            "answer's log-probability increases relative to the rejected one. It could be "
            "applied after this project's SFT stage to further prefer well-structured teaching "
            "answers."
        ),
        "applications": [
            "Aligning open-source chat models (e.g. Zephyr, many Llama fine-tunes)",
            "Preference tuning for style, helpfulness and safety",
            "A cheaper alternative to RLHF for academic and small-team projects",
            "Implemented in Hugging Face TRL as DPOTrainer",
        ],
        "advantages": [
            "Much simpler than RLHF: no reward model and no RL loop",
            "Stable training with a supervised-style loss",
            "Lower compute cost and easy to combine with LoRA",
        ],
        "limitations": [
            "Needs good-quality paired preference data",
            "Can overfit preference data or over-optimise without regularisation",
            "Offline: it does not explore new responses during training like online RL",
        ],
        "key_points": [
            "DPO aligns models directly from (prompt, chosen, rejected) pairs.",
            "No explicit reward model or PPO; the policy acts as an implicit reward model.",
            "A frozen reference model and the beta parameter control deviation.",
            "Typical pipeline: pre-training -> SFT -> DPO.",
            "Introduced by Rafailov et al., 2023.",
        ],
        "viva": [
            ("What data does DPO need?", "Preference pairs: a prompt, a chosen (preferred) response and a rejected response."),
            ("How is DPO different from RLHF?", "RLHF trains a separate reward model and then uses reinforcement learning such as PPO. DPO optimises the policy directly on preference pairs with a simple loss, without a reward model or RL."),
            ("What is the role of the reference model in DPO?", "It is a frozen copy of the starting model; DPO measures probability changes relative to it, which keeps the trained model from drifting too far."),
            ("What does the beta parameter control in DPO?", "How strongly the model is allowed to deviate from the reference model; it plays a role similar to the KL penalty in RLHF."),
        ],
        "interview": [
            ("Why has DPO become popular compared with PPO-based RLHF?", "DPO shows that the RLHF objective with a KL constraint can be optimised in closed form on preference pairs, so we can skip training a reward model and running unstable on-policy RL. It is a simple classification-like loss that is stable, cheap and easy to implement with LoRA, and in many benchmarks it matches or approaches PPO-based RLHF."),
        ],
        "mcqs": [
            {"q": "DPO removes the need for:", "options": ["Training data", "A separate reward model and RL loop", "A tokenizer", "The base model"], "answer": "B", "explanation": "DPO optimises preferences directly with a supervised-style loss."},
            {"q": "A DPO training example contains:", "options": ["An image and a label", "A prompt, a chosen response and a rejected response", "Only a prompt", "A reward number only"], "answer": "B", "explanation": "DPO learns from pairwise preferences."},
        ],
    },
]
