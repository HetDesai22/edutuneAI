"""Artificial Intelligence fundamentals."""

AI_TOPICS = [
    {
        "key": "artificial_intelligence",
        "name": "Artificial Intelligence (AI)",
        "short": "Artificial Intelligence",
        "domain": "Artificial Intelligence",
        "question": "What is Artificial Intelligence?",
        "definition": (
            "Artificial Intelligence (AI) is the field of computer science that builds systems "
            "able to perform tasks that normally require human intelligence, such as perceiving, "
            "reasoning, learning, planning, understanding language and making decisions."
        ),
        "simple": (
            "AI is about making computers do things that usually need human thinking. Instead of "
            "following only fixed step-by-step rules, an AI system can recognise patterns, make "
            "decisions and sometimes learn from experience. When your phone unlocks by recognising "
            "your face, or a map app finds the fastest route, that is AI at work."
        ),
        "eli10": (
            "Imagine teaching a robot to play a game. At first it knows nothing, but you show it "
            "lots of games and it slowly figures out which moves are good. AI is the science of "
            "making computers smart like that, so they can spot things, solve puzzles and make "
            "choices a bit like people do."
        ),
        "analogy": (
            "Think of AI as a new employee. Rule-based AI is an employee who follows a thick "
            "instruction manual exactly. Machine-learning AI is an employee who watches thousands "
            "of examples of experienced colleagues and learns the patterns on their own. Both "
            "get the job done, but the second one can handle situations the manual never "
            "described."
        ),
        "how_it_works": [
            "Define the task and how success is measured (for example, accuracy of a diagnosis).",
            "Represent the problem: states and actions for search, facts and rules for reasoning, or features for learning.",
            "Choose an approach: search and planning, logical reasoning, probabilistic models or machine learning.",
            "Build or train the system on knowledge or data.",
            "Evaluate it on new situations and improve it.",
        ],
        "example": (
            "A spam filter is a classic AI system. It looks at features of an email (words used, "
            "sender, links) and decides whether it is spam. A modern filter is trained on millions "
            "of emails labelled 'spam' or 'not spam', so it learns patterns such as 'contains "
            "\"urgent prize\" and a suspicious link' without a programmer writing every rule."
        ),
        "applications": [
            "Healthcare: medical image analysis and diagnosis support",
            "Transport: route planning and driver-assistance systems",
            "Language: translation, chatbots and speech recognition",
            "Finance: fraud detection and credit scoring",
            "Recommendation systems on streaming and shopping platforms",
        ],
        "advantages": [
            "Automates repetitive and large-scale decision tasks",
            "Can find patterns in data that are too large for humans to analyse",
            "Works continuously and consistently",
        ],
        "limitations": [
            "Learned systems can inherit bias from their training data",
            "Many models are hard to interpret (black boxes)",
            "Systems can fail in situations very different from what they were built or trained for",
        ],
        "key_points": [
            "AI builds systems that perform tasks requiring human-like intelligence.",
            "Narrow AI solves specific tasks; general AI (human-level on all tasks) does not yet exist.",
            "Main approaches: search, knowledge-based reasoning, probabilistic reasoning and machine learning.",
            "Machine learning is a subset of AI; deep learning is a subset of machine learning.",
            "The Turing Test is a classic proposal for judging machine intelligence through conversation.",
        ],
        "viva": [
            ("Define Artificial Intelligence.", "AI is the branch of computer science concerned with building systems that perform tasks requiring human intelligence, such as reasoning, learning, perception and decision making."),
            ("What is the difference between narrow AI and general AI?", "Narrow AI is designed for one specific task, like playing chess or recognising faces. General AI would match human ability across any intellectual task; it does not exist yet."),
            ("How are AI, ML and DL related?", "They are nested: machine learning is a subset of AI that learns from data, and deep learning is a subset of machine learning that uses multi-layer neural networks."),
            ("What is the Turing Test?", "Proposed by Alan Turing in 1950, it says a machine can be considered intelligent if a human judge, chatting by text, cannot reliably tell it apart from a human."),
            ("Give two ethical concerns in AI.", "Bias and unfairness learned from data, and lack of transparency or accountability in automated decisions. Privacy is another major concern."),
        ],
        "interview": [
            ("How would you explain the difference between AI and machine learning to a non-technical manager?", "AI is the overall goal of making machines act intelligently. Machine learning is one way of achieving it: instead of programming every rule, we give the system many examples and it learns the rules itself. So all machine learning is AI, but not all AI is machine learning; a hand-written chess engine is AI without learning."),
            ("What are the main risks of deploying an AI system in production?", "Data drift, where real-world data changes and accuracy drops; bias that leads to unfair outcomes; lack of explainability for regulated decisions; security issues like adversarial inputs; and over-reliance by users. I would mitigate these with monitoring, fairness checks, human review for high-stakes decisions and regular retraining."),
        ],
        "mcqs": [
            {"q": "Which statement best describes the relationship between AI, ML and DL?", "options": ["They are unrelated fields", "DL is a subset of ML, which is a subset of AI", "AI is a subset of ML", "ML is a subset of DL"], "answer": "B", "explanation": "Deep learning is a type of machine learning, and machine learning is one approach to AI."},
            {"q": "Which of these is an example of narrow AI?", "options": ["A human-level general problem solver", "A chess-playing program", "A conscious robot", "None of these"], "answer": "B", "explanation": "A chess program is specialised for one task, which is the definition of narrow AI."},
            {"q": "Who proposed the Turing Test?", "options": ["John McCarthy", "Geoffrey Hinton", "Alan Turing", "Marvin Minsky"], "answer": "C", "explanation": "Alan Turing proposed the 'imitation game' in his 1950 paper 'Computing Machinery and Intelligence'."},
        ],
    },
    {
        "key": "intelligent_agents",
        "name": "Intelligent Agents",
        "short": "intelligent agents",
        "domain": "Artificial Intelligence",
        "question": "What is an intelligent agent?",
        "definition": (
            "An intelligent agent is anything that perceives its environment through sensors and "
            "acts on that environment through actuators, choosing actions that maximise its "
            "expected performance measure."
        ),
        "simple": (
            "An agent is something that can sense what is around it and then do something about "
            "it. A robot vacuum is a good example: it senses dirt and walls, and it decides where "
            "to move and when to suck. It is called 'intelligent' when it picks actions that help "
            "it reach its goal, like cleaning the whole room quickly."
        ),
        "eli10": (
            "Think of a robot vacuum. It has 'eyes' (sensors) to notice dirt and walls, and "
            "'hands and feet' (wheels and a brush) to move and clean. It keeps looking around and "
            "deciding what to do next. Anything that notices things and then acts to reach a goal "
            "is called an agent."
        ),
        "analogy": (
            "An intelligent agent is like a thermostat with a brain. A simple thermostat senses "
            "the temperature and switches the heater on or off (a simple reflex agent). A smart "
            "thermostat also remembers your routine, predicts when you will come home and plans "
            "heating in advance to save energy (a goal- and utility-based agent)."
        ),
        "how_it_works": [
            "Percept: the agent receives input from sensors (camera, text, temperature).",
            "Agent function: maps the history of percepts to an action.",
            "The agent program implements this function, possibly using internal state, goals or a utility function.",
            "Actuators carry out the chosen action, which changes the environment.",
            "The cycle repeats: perceive, decide, act.",
        ],
        "example": (
            "A self-driving car is an agent. Its sensors are cameras, radar and GPS; its actuators "
            "are the steering, accelerator and brakes. Its performance measure includes safety, "
            "reaching the destination, speed and passenger comfort. Its environment is partially "
            "observable, dynamic, continuous and multi-agent, which makes it very challenging."
        ),
        "applications": [
            "Robot vacuum cleaners and warehouse robots",
            "Game-playing agents such as chess and Go engines",
            "Software agents like web crawlers and trading bots",
            "Virtual assistants that plan and carry out tasks",
        ],
        "advantages": [
            "Gives a unified framework to describe any AI system",
            "Separates the environment, percepts, actions and goals clearly",
            "Scales from simple reflex behaviour to learning agents",
        ],
        "limitations": [
            "Designing a correct performance measure is hard",
            "Partially observable, dynamic environments make decisions uncertain",
            "Simple reflex agents fail when the right action depends on history",
        ],
        "key_points": [
            "Agent = architecture + program; it perceives via sensors and acts via actuators.",
            "PEAS describes a task: Performance measure, Environment, Actuators, Sensors.",
            "Agent types: simple reflex, model-based reflex, goal-based, utility-based and learning agents.",
            "Environment properties: observable, deterministic, episodic, static, discrete, single/multi-agent.",
            "A rational agent chooses the action expected to maximise its performance measure.",
        ],
        "viva": [
            ("What is PEAS?", "PEAS stands for Performance measure, Environment, Actuators and Sensors. It is used to specify the task environment of an agent."),
            ("What is a rational agent?", "A rational agent selects the action that is expected to maximise its performance measure, given its percepts and built-in knowledge."),
            ("Name the five types of agents.", "Simple reflex, model-based reflex, goal-based, utility-based and learning agents."),
            ("What is the difference between a goal-based and a utility-based agent?", "A goal-based agent only checks whether a goal is reached. A utility-based agent measures how good each state is, so it can choose between several ways of reaching the goal."),
            ("What does 'partially observable' mean?", "The agent's sensors cannot see the complete state of the environment at each moment, so it must keep an internal model or memory."),
        ],
        "interview": [
            ("Describe the PEAS for an automated taxi.", "Performance: safety, legal driving, reaching the destination, speed, comfort and profit. Environment: roads, traffic, pedestrians, weather and customers. Actuators: steering, accelerator, brake, horn and display. Sensors: cameras, lidar/radar, GPS, speedometer, odometer and microphones."),
            ("Why is a simple reflex agent not enough for most real problems?", "It only reacts to the current percept using condition-action rules, so it fails when the environment is partially observable or when the correct action depends on the past. Model-based agents fix this by keeping internal state, and goal- or utility-based agents add planning."),
        ],
        "mcqs": [
            {"q": "In PEAS, what does the 'A' stand for?", "options": ["Algorithm", "Actuators", "Accuracy", "Agent"], "answer": "B", "explanation": "PEAS = Performance measure, Environment, Actuators, Sensors."},
            {"q": "Which agent type keeps an internal state to handle partially observable environments?", "options": ["Simple reflex agent", "Model-based reflex agent", "Random agent", "Table-driven agent"], "answer": "B", "explanation": "A model-based agent maintains an internal model of the world that tracks parts it cannot currently see."},
        ],
    },
    {
        "key": "search_algorithms",
        "name": "Search Algorithms in AI",
        "short": "search algorithms",
        "domain": "Artificial Intelligence",
        "question": "What are search algorithms in AI?",
        "definition": (
            "Search algorithms solve problems by systematically exploring a state space: starting "
            "from an initial state, they apply actions to generate new states until they reach a "
            "goal state, ideally along the lowest-cost path."
        ),
        "simple": (
            "Many AI problems can be turned into 'find a path' problems. A maze, a route on a map "
            "or a puzzle all have a start, a goal and possible moves. A search algorithm tries "
            "moves in an organised way until it finds the goal. Uninformed searches such as BFS "
            "and DFS explore blindly, while informed searches such as A* use a hint (a heuristic) "
            "about which direction looks promising."
        ),
        "eli10": (
            "Imagine you lost your ball somewhere in a big house. You could check every room one "
            "floor at a time (that is like breadth-first search), or go as deep as possible down "
            "one hallway before coming back (depth-first search). If you hear the dog barking "
            "near the ball, you can use that clue to look in the best rooms first. That clue is "
            "what AI calls a heuristic."
        ),
        "analogy": (
            "Search is like finding a route in a new city. BFS is checking every street one block "
            "away, then two blocks away, and so on. DFS is following one road as far as it goes "
            "before turning back. A* is using a map that tells you the straight-line distance to "
            "your destination, so you always try the street that looks closest overall."
        ),
        "how_it_works": [
            "Formulate the problem: initial state, actions, transition model, goal test and path cost.",
            "Put the initial state in the frontier (the list of states waiting to be explored).",
            "Repeatedly pick a state from the frontier; BFS uses a queue, DFS a stack, UCS and A* a priority queue.",
            "If it is the goal, return the path; otherwise expand it and add its unseen neighbours to the frontier.",
            "A* orders the frontier by f(n) = g(n) + h(n): cost so far plus estimated cost to the goal.",
        ],
        "example": (
            "Find a route from city A to city D. Roads: A-B (1), A-C (4), B-C (2), B-D (5), C-D (1). "
            "BFS finds A-B-D or A-C-D (fewest edges) but ignores cost. Uniform-cost search expands "
            "the cheapest path first and finds A-B-C-D with cost 1+2+1 = 4, which is optimal. "
            "A* with an admissible heuristic finds the same optimal path while expanding fewer nodes."
        ),
        "applications": [
            "GPS navigation and route planning",
            "Puzzle solving (8-puzzle, Rubik's cube, Sudoku)",
            "Game playing using minimax and alpha-beta pruning",
            "Robot motion planning and scheduling",
        ],
        "advantages": [
            "General: many problems can be framed as search",
            "BFS and UCS are complete and optimal under standard conditions",
            "Heuristics (A*) can reduce the explored space dramatically",
        ],
        "limitations": [
            "State spaces can grow exponentially with depth (combinatorial explosion)",
            "BFS needs a lot of memory; DFS may not find the shortest path",
            "A* is only as good as its heuristic",
        ],
        "key_points": [
            "Uninformed search (BFS, DFS, UCS, iterative deepening) uses no domain knowledge.",
            "Informed search (greedy best-first, A*) uses a heuristic h(n).",
            "A* uses f(n) = g(n) + h(n) and is optimal if h is admissible (never overestimates).",
            "BFS is complete and optimal for equal step costs; DFS uses little memory but is not optimal.",
            "Adversarial search (minimax, alpha-beta) is used for two-player games.",
        ],
        "viva": [
            ("What is the difference between BFS and DFS?", "BFS explores level by level using a queue and finds the shallowest goal; DFS goes as deep as possible along one branch using a stack and uses less memory but is not guaranteed to find the shortest path."),
            ("What is a heuristic?", "A heuristic is an estimate of the cost from a state to the nearest goal, used to guide informed search towards promising states."),
            ("When is A* optimal?", "A* tree search is optimal when the heuristic is admissible, meaning it never overestimates the true cost. For graph search the heuristic should also be consistent."),
            ("What is alpha-beta pruning?", "An optimisation of minimax that skips branches that cannot affect the final decision, giving the same result with fewer node evaluations."),
            ("What is iterative deepening search?", "It runs depth-limited DFS with increasing depth limits. It combines the low memory of DFS with the completeness of BFS."),
        ],
        "interview": [
            ("Why is A* preferred over Dijkstra's algorithm for route finding on a map?", "Both find optimal paths, but Dijkstra expands nodes in all directions based only on cost so far. A* adds a heuristic such as straight-line distance to the destination, so it focuses on nodes that lead towards the goal and usually expands far fewer nodes, while staying optimal if the heuristic is admissible."),
            ("What are the time and space complexities of BFS?", "With branching factor b and solution depth d, BFS takes O(b^d) time and O(b^d) space, because it stores the whole frontier. Memory is usually the bottleneck, which is why iterative deepening is often preferred."),
        ],
        "mcqs": [
            {"q": "Which data structure does BFS use for its frontier?", "options": ["Stack", "Queue", "Priority queue", "Hash map"], "answer": "B", "explanation": "BFS processes states in first-in first-out order, which a queue provides."},
            {"q": "In A*, what does f(n) = g(n) + h(n) represent?", "options": ["Cost so far plus estimated cost to goal", "Depth plus breadth", "Heuristic squared", "Number of expanded nodes"], "answer": "A", "explanation": "g(n) is the path cost from the start to n and h(n) estimates the remaining cost from n to the goal."},
            {"q": "A heuristic that never overestimates the true cost is called:", "options": ["Greedy", "Admissible", "Complete", "Adversarial"], "answer": "B", "explanation": "Admissible heuristics guarantee A* tree search returns an optimal solution."},
        ],
        "numerical": {
            "problem": "A* search: node n has path cost g(n) = 7 and heuristic h(n) = 5; node m has g(m) = 9 and h(m) = 2. Which node does A* expand first?",
            "solution": "Step 1: f(n) = g(n) + h(n) = 7 + 5 = 12.\nStep 2: f(m) = 9 + 2 = 11.\nStep 3: A* expands the node with the smallest f value.\nAnswer: node m is expanded first, because 11 < 12, even though it has a higher cost so far.",
        },
    },
    {
        "key": "knowledge_representation",
        "name": "Knowledge Representation",
        "short": "knowledge representation",
        "domain": "Artificial Intelligence",
        "question": "What is knowledge representation in AI?",
        "definition": (
            "Knowledge representation (KR) is the part of AI concerned with storing information "
            "about the world in a form a computer can use to reason, answer questions and solve "
            "problems. Common forms are logic, rules, semantic networks, frames and ontologies."
        ),
        "simple": (
            "A computer cannot reason about the world unless the facts are written down in a "
            "structured way. Knowledge representation is choosing that structure. For example, "
            "'All humans are mortal' and 'Socrates is a human' can be written in logic so the "
            "computer can conclude 'Socrates is mortal'."
        ),
        "eli10": (
            "Imagine making a family tree on paper. You draw boxes for people and lines showing "
            "who is whose parent. Now you can answer questions like 'Who is my cousin?' by "
            "following the lines. Knowledge representation is making 'family trees' like this "
            "for computers, so they can figure things out."
        ),
        "analogy": (
            "Knowledge representation is like a library's cataloguing system. The books (facts) "
            "are only useful if they are organised with titles, categories and cross-references. "
            "A good catalogue lets a librarian quickly answer 'which books are about birds that "
            "cannot fly?', just as a good representation lets an AI system reason efficiently."
        ),
        "how_it_works": [
            "Identify the objects, properties and relationships in the domain.",
            "Choose a formalism: propositional/first-order logic, production rules, semantic networks, frames or ontologies.",
            "Encode facts (e.g. Bird(Tweety)) and rules (e.g. Bird(x) -> CanFly(x)).",
            "Use an inference engine (forward or backward chaining, resolution) to derive new facts.",
            "Answer queries and explain the chain of reasoning.",
        ],
        "example": (
            "In first-order logic: Human(Socrates) and for all x, Human(x) -> Mortal(x). Using the "
            "rule Modus Ponens, the system infers Mortal(Socrates). In a semantic network, the same "
            "knowledge is a node 'Socrates' linked by an 'is-a' edge to 'Human', which has the "
            "property 'mortal', so the property is inherited."
        ),
        "applications": [
            "Expert systems for medical or technical diagnosis",
            "Knowledge graphs used by search engines",
            "Ontologies in the semantic web and biomedicine",
            "Planning systems and question answering",
        ],
        "advantages": [
            "Reasoning is explainable: you can trace which rules were used",
            "Knowledge can be edited directly by experts",
            "Works with little or no training data",
        ],
        "limitations": [
            "Manually encoding knowledge is slow (the knowledge acquisition bottleneck)",
            "Handling uncertainty and exceptions is difficult in pure logic",
            "Very expressive logics can make inference computationally expensive",
        ],
        "key_points": [
            "KR stores knowledge so a machine can reason with it.",
            "Main formalisms: logic, production rules, semantic networks, frames, ontologies.",
            "Inference methods: forward chaining (data-driven) and backward chaining (goal-driven).",
            "First-order logic adds objects, predicates and quantifiers to propositional logic.",
            "Knowledge graphs are a modern, large-scale form of semantic networks.",
        ],
        "viva": [
            ("What is the difference between propositional and first-order logic?", "Propositional logic works with whole statements that are true or false. First-order logic adds objects, predicates, functions and quantifiers (for all, there exists), so it can express general rules about many objects."),
            ("What is forward chaining?", "A data-driven inference method that starts from known facts and applies rules to derive new facts until the goal is reached or nothing new can be inferred."),
            ("What is backward chaining?", "A goal-driven method that starts from the goal and works backwards, looking for rules and facts that can prove it. Prolog uses backward chaining."),
            ("What is a semantic network?", "A graph where nodes represent concepts or objects and labelled edges represent relationships such as is-a or has-part."),
            ("What is a frame?", "A structured record representing a stereotyped object or situation, with slots (attributes) and default values, such as a 'Restaurant' frame with slots for menu and bill."),
        ],
        "interview": [
            ("When would you choose a knowledge-based approach over machine learning?", "When there is little labelled data, when decisions must be fully explainable or auditable, or when the rules are well known and stable, as in tax rules or regulatory checks. In practice, hybrid systems are common, for example a knowledge graph combined with a learned language model."),
            ("What is a knowledge graph?", "A knowledge graph stores entities as nodes and relationships as labelled edges, such as (Paris, capitalOf, France). It supports queries and reasoning over connected facts and is used in search engines, recommendation and retrieval-augmented LLM systems."),
        ],
        "mcqs": [
            {"q": "Which inference method starts from the goal and works backwards?", "options": ["Forward chaining", "Backward chaining", "Gradient descent", "Hill climbing"], "answer": "B", "explanation": "Backward chaining is goal-driven; forward chaining is data-driven."},
            {"q": "Which representation uses nodes and labelled edges such as 'is-a'?", "options": ["Frames", "Semantic network", "Truth table", "Decision tree"], "answer": "B", "explanation": "Semantic networks are graphs of concepts connected by relationships."},
        ],
    },
]
