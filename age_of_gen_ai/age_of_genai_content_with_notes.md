# Age of Gen AI - Slide Content

## **Slide 1: Title Slide**
- **Title:** Age of Gen AI
- **Subtitle:** Evolution, Capabilities, and Future Trends
- **Presenter:** Jit Ray Chowdhury
- **Date:** August 2025

- **Visual:** Abstract AI visualization with neural network patterns.

---

## **Slide 2: Table of Contents**
- **Part 1:** Foundation
- **Part 2:** Historical Context & Key Concepts
- **Part 3:** Technical Deep Dive
- **Part 4:** Current State & Future Trends

---
## **Part 1: Foundation**
---

## **Slide 3: GenAI Capabilities**
- **Text/Code Generation:**
    - Natural language understanding and generation
    - Code completion and generation
    - Document creation and summarization
- **Image/Video Generation:**
    - Text-to-image creation (DALL-E, Midjourney, Stable Diffusion)
    - Style transfer and image editing (NanoBanana, Qwen3-Image-Edit)
    - Video synthesis from prompts (Sora, Runway ML, Kling 2.1)
- **Actions/Tool/API Calls:**
    - Autonomous agents executing tasks
    - Tool use and reasoning capabilities
    - API integration for real-world actions
- **Visual:** Modern AI interface showing examples of each capability with input/output pairs.

---

## **Slide 4: What Makes GenAI Different?**
- **Foundation Models:** Pre-trained on vast datasets, adaptable to multiple tasks.
    - Traditional pre-trained models (ImageNet/COCO) were task-specific and required fine-tuning
    - Foundation models enable multi-task, zero-shot learning, and extend capabilities with prompt/context engineering
- **Multimodal Capabilities:** Processing and generating text, images, audio, and video simultaneously.
    - Previously, CNNs were common, text models were not as advanced, and combined text-image models were rare.
    - Custom token support for Actions, Graphs, and Road Networks
- **Natural Interfaces:** Conversational interaction using everyday language.
    - Significant advancement over earlier NLP models (LSTMs, RNNs)
- **Generation vs. Recognition:** Creating new content rather than just classifying it.
    - While GANs and translation models generated content previously, open-vocabulary image detection and any-text-to-image generation represent new capabilities
- **Scale:** Unprecedented model size (billions to trillions of parameters) and computational requirements.
- **Visual:** Split-screen comparison diagram showing traditional AI (rule-based, single-task) vs. GenAI approaches (foundation models, multi-task), with connecting arrows showing evolution.

---

## **Slide 5: Foundation Models vs. Pre-trained Models**

- **Traditional Pre-trained Models (2010s era e.g., ImageNet, COCO):**
    - **Task-specific datasets:** ImageNet (1000 categories), COCO (object detection)
    - **Limited scope:** Feature extractors for specific domain
    - **Transfer learning:** Required significant fine-tuning for new tasks
    - **ImageNet: The First "Common Sense" Dataset for Vision:** Revolutionized computer vision

- **Foundation Models (2020s breakthrough e.g., LLMs):**
    - **Trained on entire internet:** - trillions of tokens, self-supervised
    - **Emergent capabilities:** Exhibit reasoning, common sense understanding, Zero/few-shot learning not explicitly programmed
    - **Versatile adaptation:** Prompting, instruction tuning, minimal fine-tuning
    - **Multimodality:** Can process and generate information across text, images, audio, and more

- **Visual:** Split diagram showing ImageNet-style models vs. modern foundation models, highlighting the shift from task-specific to general-purpose AI.

---

## **Slide 6: Evolution from Traditional AI to GenAI**
- **AI: Artificial Intelligence (1950s)**
    - Expert systems for medical diagnostics from symptom databases
    - Rule-based systems for path planning (early GPS systems)
    - Logical reasoning and theorem proving using symbolic representation
- **ML: Machine Learning (1980s)**
    - Support Vector Machines (SVMs) for image classification and text categorization
    - Decision Trees/Random Forests for pose estimation (Kinect) and fraud detection
- **DL: Deep Learning (2010s)**
    - Convolutional Neural Networks (CNNs) for object detection in autonomous vehicles and medical image analysis.
    - Recurrent Neural Networks (RNNs) for language translation and sequence prediction
- **GenAI: Generative AI (2020s)**
    - GPT (Generative Pre-trained Transformer) to predict the next item in a sequence.
    - DALL-E for realistic image generation from natural language descriptions
- **Visual:** Interactive timeline or nested concentric circles showing AI evolution progression, with increasing complexity and model parameter counts visualized through size and color gradients. **Visual Timeline Segments:** https://www.infodiagram.com/slides/ai-timeline-history-machine-learning-deep-learning/

---

## **Slide 7: AI Evolution: From Narrow to Broad Capabilities**
- **AI (1950s):** Mimic human intelligence.
    - Hand-written code/algorithms by human domain experts
- **ML (1980s):** Learning trends from historical data
    - Hand-crafted features, handwritten per domain/application.
    - Function fitting on features
- **DL (2010s):** Learning features directly from large-scale data.
    - Same model architecture with automatic feature extraction per task
    - Different trained model weights per task
    - Trained from large-scale datasets
- **GenAI (2020s):** Large foundation models with multimodal capabilities.
    - Same trained model for zero-shot application across multiple tasks
    - Internet-scale training data (trillions of tokens)
    - Models with billions to trillions of parameters
    - Multimodal I/O (Text, Image, Audio, Actions, API Calls)
- **From Specialized to General:** Domain-specific → Foundation models.
- **Scaling Laws:** Performance improvements correlate with model size, data and compute
- **Visual:** Combined dashboard showing: (1) Exponential growth chart of model parameters over time, (2) ImageNet accuracy progression, (3) Scaling laws correlation between compute/data/performance, all in consistent modern styling.

---
## **Slide 8: AI, Uncertainty, and the Intelligence Trade-off**
-  **Key Insight:** As we embrace higher levels of intelligence and autonomy, we must also accept more uncertainty. 
    - You can't expect a rule-based algorithm to work well on unstructured data, nor can you expect a smart model that handles unstructured data to always follow the same rules without creativity.


- **AI's Embrace of Uncertainty:**
    - **Traditional AI (1950s):** Deterministic, probabilistic, provable, and introspectable            
    - **ML (1980s):** Deterministic, probabilistic, provable, and introspectable. Learn decision making/rules from extracted features, data driven.
    - **Deep Learning (2010s):** "Black box" models with statistical proof but limited explainability. Solved many more corner cases by learning them from data.
    - **GenAI (2020s):** Handles much more ambiguity, like unstructured data. Much smarter but hallucinates.
- **Probabilistic Robotics:**: Models uncertainty (sensor noise, errors) for robust behavior in unpredictable environments.
- **Organizational Hierarchy Analogy:** More responsibility with more Autonomy and Empowerment.
    - **Entry Level (Traditional AI):** Strict rules, clear instructions, limited adaptability
    - **Middle Management (ML/DL):** Independent problem-solving with measurable results
    - **Executive Leadership (GenAI):** High autonomy, creativity, strategic decision-making with incomplete information
- **Programming Language  Analogy:**
    - more optimization and control in Assembly and C than with Python. 
    - GenAI proving more intuitive natular interface, but prompt engineering becomes neccasry is needed as we've embraced a more ambiguous interface.

- **Visual:** Conceptual diagram showing increasing abstraction and uncertainty with AI evolution, with subtle hierarchy visual showing progression from rigid rules to creative autonomy.

---
## **Part 2: Historical Context & Key Concepts**
---

## **Slide 9: A Personal Journey: Early Insights**
- **2000: Automate everything**
    - "I am lazy, can't stand inefficiency, and repeating the same task."
- **2005: Operate at speed of thought**
    - Software accelerated many tasks (Simulation, e-commerce)
    - "If you can think about it, you can build it."
    - **Challenge: "It still takes a long time to build (find info, write code, take action)."**
- **2008: Make Robots**
    - Online tasks (banking/e-commerce) becoming solved and less challenging
    - "I need robots to solve real-world problems (like caring for elderly parents remotely)"
- **2011: Common sense for robots**
    - How to collect and structure all human knowledge?
- **2013: Dynamic compute allocation**
    - Impossible to process all 360-degree sensors in detail; compromises are made like masking, subsampling, and classification before planning.
    - Human analogy: On slippery roads, focus on controls and only nearby obstacles.
- **2014: No fixed interfaces.**
    - Adaptive control systems (subconscious vs. precise leg control for dancing)
    - Transfer learning between similar devices (cameras), learn transfer effeciently from demo, manual, video.

---

## **Slide 10: How GenAI Addressed These Challenges**
- **2000: ✗ Automate everything**
    - Progress with personal AI assistants and robots, but not fully achieved
- **2005: ✗ Operate at speed of thought**
    - Coding agents helping, but no perfect finance/travel agents yet
- **2008: ✗ Make Robots**
    - Self Driving is reality, but not general home/factory robots
- **2011: ✓ Common sense for robots**
    - ImageNet: The first common sense Database, choose images only to represnt knowledge, limited vocabulary
    - Internet-scale text training with next-word prediction
- **2013: ✓ Dynamic compute allocation**
    - Attention mechanisms, Mixture of Experts (MoE), Model Routing, Multi layer LLM with early stop 
    - Inference-time reasoning, Chain of Thought (CoT).
- **2014: ✓ No fixed interfaces.**
    - Transfer learning in early DL models (e.g., if you change the camera).
    - Foundation models, in-context learning, Zero-shot learning, prompt engineering, and RAG (Retrieval Augmented Generation).
    - Natural language interfaces accessible to all humans


---

## **Slide 11: Types of Learning in AI**
- **Supervised Learning:**
    - Uses labeled data for training
    - Historically most successful approach
    - Examples: Image classification, speech recognition
- **Unsupervised Learning:**
    - No labels required, leverages larger datasets
    - Pattern discovery (clustering, dimensionality reduction), like (KNN, KMeans)
- **Reinforcement Learning:**
    - Trial-and-error learning with reward feedback
    - **[DEBATE: Some believe path to AGI, others cite sample inefficiency limitations]**
    - Renaissance in combination with LLMs for reasoning tasks
- **Self-Supervised Learning:**
    - Best of Supervised and Unsupervised, Not traditional
    - Learn as well as Supervised with psudo-labels but large scale like Unsupervised
    - **Critical for foundation models like GPT, CLIP, DinoV3**
- **Visual:** Venn diagram showing convergence of learning approaches in modern GenAI systems.

---

## **Slide 12: Classical AI Concepts: Making a Comeback**
- **Why Classical Methods Failed Initially:**
    - **Reinforcement Learning:** Sample inefficient, lacked common sense
    - **Agent-Based Systems:** Brittle, domain-specific rule systems
    - **Search Algorithms:** Limited to well-defined state spaces, like A*
- **Why They Work Now with LLMs:**
    - **[KEY ENABLER: LLMs provide common sense reasoning and world knowledge]**
    - **Better representations:** Natural language descriptions vs. hand-coded features
    - **Hybrid approaches:** Neural reasoning combined with classical algorithms
- **Modern Success Examples:**
    - **Planning:** LLMs decompose complex goals into actionable sub-tasks
    - **Search:** Semantic search using embeddings vs. keyword matching
    - **RL:** Language feedback instead of sparse numerical rewards
- **Visual:** Split image showing old CS textbook vs. modern AI system architecture combining both approaches.

---

## **Slide 13: The Bitter Lesson: Advice for AI Developers**
 
**(From Rich Sutton's Essay, 2019)**

- **Bet on Scaling, Not on Human Knowledge**
    - General methods that leverage computation are ultimately the most effective, and by a large margin
    - The biggest lesson from 70 years of AI is that general methods leveraging massive computation consistently win. The real world is endlessly complex; instead of trying to code this complexity directly into your models, build the meta-methods that can discover and capture it on their own.
    - > *"The biggest lesson that can be read from 70 years of AI research is that general methods that leverage computation are ultimately the most effective, and by a large margin."*

- **Embrace General-Purpose Methods: Search and Learning**
    - **Search and Learning** are the only approaches that scale with computation
    - Focus on the two methods that scale with computation: search and learning. History shows they eventually outperform systems designed with human expertise.
    - > *"We have to learn the bitter lesson that building in how we think we think does not work in the long run."*
 - **Source:** http://www.incompleteideas.net/IncIdeas/BitterLesson.html

### **Proof in Practice: Computation Outperforms Human Knowledge**

- **Computer Chess (Deep Blue, 1997)**
    - Massive brute-force search (200 million positions/second) defeated world champion Kasparov
    - Hand-coded grandmaster strategies proved inferior to computational search

- **Computer Go (AlphaGo, 2016)**
    - Deep learning + self-play discovered superior strategies
    - Human intuition and traditional Go knowledge surpassed by computation
    - Human intuition was thought essential, but AlphaGo mastered the game through deep learning and extensive self-play, discovering superior strategies.

- **Natural Language Processing (Large Language Models, e.g., GPT-3)**
    - Brittle, hand-coded grammar rules were replaced by models that learn complex patterns from vast text datasets, achieving nuanced human-like communication.
    - Statistical models replaced brittle hand-coded grammar rules
    - Achieved nuanced human-like communication through scale

- **Computer Vision (DINOv2, 2023)**
    - Self-supervised learning on unlabeled data surpassed supervised methods
    - No hand-designed features required, just computational scale
---

## **Slide 14: Yann LeCun's Cake Analogy (Information Density Analysis)**
### **Information Content During Learning:**
- **Reinforcement Learning (the cherry - 1%):**
    - Scalar reward signal given intermittently
    - **Few bits per sample**
- **Supervised Learning (the icing - 10%):**
    - Category or numerical predictions for each input
    - **10 to 10,000 bits per sample**
- **Self-Supervised Learning (the cake - 89%):**
    - **[KEY INSIGHT: Predicts any part of input from any observed part]**
    - Future frame prediction, masked language modeling
    - **Millions of bits per sample**

**[SOURCE: Yann LeCun presentations and papers on self-supervised learning]**

---

## **Slide 15: The CNN Era (2012-2017)**
- **[2012 BREAKTHROUGH: AlexNet won ImageNet, triggering deep learning revolution]**
- **Strong inductive bias:** Convolution operation perfectly suited for image structure
- **Everything-to-image paradigm:** Audio spectrograms, road networks as semantic layers
- **Key datasets:** ImageNet (classification), COCO (detection/segmentation)
- **Architecture evolution:** Deeper networks (ResNet skip connections), wider networks, pyramid architectures
- **[CURRENT STATUS: Being superseded by Vision Transformers (ViT) and DINOv2/v3]**

---

## **Slide 16: RL Excitement and Difficulty**
- **Early promise:** Deep RL on Atari games, AlphaGo breakthrough
- **Persistent challenges:** Reward hacking, sparse reward signals
- **Sample inefficiency:** Millions of interactions needed for simple tasks
- **[LIMITATION: Worked well in constrained environments, struggled in open worlds]**
- ***Visual: Image of a boat racing video game.***

---

## **Slide 17: Deep RL for AGI Ambitions (2016 Peak)**
- **Key players:** DeepMind (AlphaGo), OpenAI (Dota 2, robotics)
- **Major projects:** 
    - **AlphaGo defeats Lee Sedol (2016)**
    - **Atari game suite benchmarks**
    - **OpenAI Gym standardized environments**
- **[HISTORICAL NOTE: This period saw peak optimism for RL as path to AGI]**

---
## **Part 3: Technical Deep Dive**
---

---

## **Slide 18: Transformers Changed Everything (2017)**
**[SEMINAL PAPER: "Attention Is All You Need" - Vaswani et al., Google, 2017]**

- **Self-attention mechanism:** Direct modeling of relationships between all sequence positions
- **Revolutionary insight:** Eliminated recurrence, enabled full parallelization
- **[PERFORMANCE: Superior translation quality with significantly less training time]**

- **Easy to Scale** Faster, shallower network that can handle more scale. 

---

## **Slide 19: GPT Emerges, RL Winter Begins (2019 Inflection Point)**
- **RL Criticism Peak - "Deep Reinforcement Learning is a waste of time" (2019):**
    - Still sample-inefficient despite years of research
    - Limited to toy problems and constrained action spaces
    - No common sense reasoning, excessive exploration in open environments
- **GPT-1/GPT-2 Breakthrough:**
    - **[KEY INSIGHT: Next-token prediction showed emergent intelligence]**
    - Initially limited to text completion and basic coding assistance
    - **[SURPRISE: Scaling simple objective yielded complex behaviors]**

---

## **Slide 20: How ChatGPT is Better than GPT**
- Supervised Fine-Tuning (SFT) with Reinforcement Learning from Human Feedback (RLHF).
- Also called: instruction tuned, Human alignment, Preference optimization.
- Basically tuned for humans to like it and be able to communicate easily.
- Without this GPT was like a human expert with a lot of knowledge but not good social communication skill, which we sometimes see in human geniuses as well.

---


## **Slide 21: RL Renaissance: The Cherry on Top (2024-2025)**
- RL for reasoning: DeepSeek, RLVR (Reinforcement Learning with Verifiable Rewards).
- The role of RL continues to grow.
- RL is important for code debugging and Agentic problem solving with exploration and learning

---

## **Slide 22: GenAI Architecture Overview**
- **Transformer Architecture:** Self-attention enabling parallel sequence processing
- **Diffusion Models:** Gradual denoising for high-quality image/video generation
- **Retrieval Augmented Generation (RAG):** Enhancing generation with external knowledge retrieval.
- **Fine-Tuning:** From full fine-tuning to LoRA to prompt engineering.
- **Mixture of Experts (MoE):** For efficient scaling
- **Visual:** Split diagram showing: (1) Transformer self-attention mechanism with query/key/value visualization, (2) Diffusion model denoising process steps, both using clean technical illustration style with consistent color coding.

---

## **Part 4: Current State & Future Trends**

---

## **Slide 23: Current GenAI Trends (2025)**
- **Multi-modal AI:** Systems processing and generating multiple data types simultaneously (text, images, audio, video, actions).
- **AI-First Applications:** Software built with AI at its core rather than as an add-on
- **Specialized AI Agents:** Task-specific AI systems with enhanced capabilities
- **Generative User Interfaces:** AI-assisted interface design and interaction, creating dynamic UIs.
- **Hyper-Personalization:** Tailored experiences based on individual user preferences and behavior patterns.
- **Conversational AI:** More advanced and natural language interactions across platforms.
- **Service as Software:** AI agents tackling broader range of applications and driving productivity gains.
- **Healthcare Integration:** AI enhancing operations and engaging stakeholders in healthcare sector.
- **Workplace Transformation:** AI empowering employees and unlocking productivity potential.
- **Real-time AI Reasoning:** Enhanced voice interfaces and real-time responsiveness for enterprise decision-making.
- **Visual:** Trend graph showing growth trajectories, or dashboard-style layout with icons representing each trend.

### **Next**
- **Perception AI (2012 AlexNet):** Speech Recognition, Deep Recognition, Medical Imaging
- **Generative AI (2020s):** Digital Marketing, Content Creation  
- **Agentic AI (Current):** Coding Assistant, Customer Service, Patient Care
- **Physical AI (Next):** Self-Driving Cars, General Robotics
- **Visual:** Image showing a diagram of AI progression***


---

## **Slide 24: LLM-powered Autonomous Agent Systems**
-  **Systems Thinking:** LLM models are like powerful engines; you need systems like airplanes, cars, or factories to take advantage of them and produce value.

- **Components:**
    - **Foundation model:** Intelligence/reasoning engine
    - **Tools:** Calculator, code interpreter, web search, APIs
    - **Planning:** Chain-of-thought, self-reflection, subgoal decomposition
    - **Memory:** Short-term context + long-term storage systems
    - **Action execution:** Real-world task completion
- **Applications:**
    - Customer service automation
    - Business process optimization
    - Code generation and debugging
    - Research and data analysis
    - Personal assistants
- **Visual:** Agent architecture flowchart showing foundation model connected to tools, memory, and action systems. https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/#reflection

---

## **Slide 25: Next Challenges**
- **World Model:**
    - Next-token prediction insufficient for 3D spatial reasoning
    - Scarcity of "boring" everyday activity data (cooking, cleaning, commuting)
    - 3D spatial data extremely limited compared to text/images
- **Persistent Memory:**
    - ANALOGY: Don't want employees acting like it's their first day every day
    - What to store (importance, recency, relevance)
    - How to store it (Graph structures, temporal sequences, hierarchical storage, etc.)
- **Continuous Learning:**
    - Models need to build "muscle memory" and continuously learn or retrain.
    - Just as when you write code, you don't look up API docs every time; looking up common things can be inefficient.
    - Just like self-reflection, mistakes need to be learned from and not repeated.
    - We need a "sleeping over it" effect on learning capabilities.
- **Specialization:**
    - Although we've strived for generalization to achieve our AGI dreams, we mostly need it for common sense (the foundation).
    - Not all humans know everything, not even experts.
    - Big models are large and inefficient.
    - We need specialized, expert, efficient, smaller, faster models fine-tuned for particular tasks, personalities, and opinions. We can have a bunch of them for diversity, which is still better than having no opinions in many applications.

---

## **Slide 26: Risks and Responsible Development**
- **Demonstrated Capabilities and Risks:**
    - **[FACT: LLM-generated content passes Turing tests, enabling sophisticated misinformation]**
    - **Evolution:** Sandboxed → read-only internet access → agentic write capabilities
- **Current Risk Acceleration:**
    - **[CONCERN: Agentic deployment without adequate guardrails due to commercial pressure]**
    - **Vision-Language-Action (VLA) models:** Extending AI control to physical robots
    - **[HISTORICAL PARALLEL: Autonomous vehicles required extensive testing; AI deployment lacks similar caution]**
- **Accessibility vs. Safety Trade-off:**
    - **[RISK: Easy deployment without training expertise or computational barriers]**
    - No accountability frameworks for widespread AI deployment
    - **[RECOMMENDATION: Careful evaluation before factory and home deployment]**

---

## **Slide 27: Appendix**

### **A1: Challenges of Reinforcement Learning (RL)**
- **Difficulty of Reward Specification:**
    - Designing effective reward functions without incentivizing unintended behavior (reward hacking).
    - Hard to specify "good" outcomes precisely and comprehensively for complex tasks.
- **Sample Inefficiency:**
    - Requires vast number of environment interactions to learn optimal policies.
    - Expensive, time-consuming, or dangerous in real-world applications (robotics, autonomous driving).
- **Limited State and Action Spaces:**
    - Historically successful only in constrained environments (board games, simple control systems).
    - Real-world problems involve high-dimensional, continuous spaces making exploration extremely challenging.
- **Need for Common Sense and Guided Exploration:**
    - Early RL struggled without prior knowledge or domain-specific heuristics.
    - Agents would "wander aimlessly" in vast, unstructured environments without guidance.

### **A2: Unsupervised Learning: Images vs. Language**
- **Challenges with Images:**
    - Raw pixel data lacks explicit, discoverable hierarchical structures consistent across diverse images.
    - Pixel meaning highly context-dependent (red pixel could be car, flower, or brick wall).
    - Unsupervised models often learned noisy or uninterpretable features without human-annotated labels.
- **Success with Language:**
    - **Sequential and Contextual Nature:** Words appear in specific order conveying meaning; context provides strong learning signals.
    - **Statistical Regularities:** Rich patterns in word co-occurrence and grammatical structures.
    - **Hierarchical Structure:** Natural progression from characters → words → phrases → sentences → paragraphs.
    - **Predictive Tasks:** Masked language modeling (BERT) and next-token prediction (GPT) leverage inherent language structure.

### **A3: AI Agents: Resurgence of Early Concepts**
- **Early AI Limitations:**
    - **Complexity:** Real-world environments overwhelmed available computational power.
    - **Uncertainty:** Assumed perfect knowledge of environment (rarely realistic).
    - **Brittleness:** Rule-based systems failed when encountering unprogrammed situations.
    - **Lack of Common Sense:** Missing broad, intuitive understanding of the world.
- **LLMs Re-enabling Classical Methods:**
    - **Broad World Knowledge:** Unprecedented breadth of general knowledge and common sense from vast training data.
    - **Natural Language Interface:** Human-like text understanding enabling intuitive goal specification.
    - **Reasoning and Planning:** Emergent capabilities for breaking down complex tasks into sub-goals.
    - **Tool Use:** Can be augmented with external tools (APIs, search engines, code interpreters).
    - **Handling Unstructured Information:** Excel at processing real-world's dominant unstructured text format.

---

## **Miscellaneous and References**

### **Initial Thoughts**
- **Wish to execute a lag:**
    - I want to travel -> travel planning.
    - I want an application -> writing code.
- [NVIDIA - Generative Physical AI](https://www.nvidia.com/en-us/glossary/generative-physical-ai/)
- [Understanding Generative AI in Layman's Terms](https://pub.aimind.so/understanding-generative-ai-in-laymans-term-46e8d088659a)

### **References 1: The Bitter Lesson**
1. [The Bitter Lesson by Rich Sutton (PDF)](https://www.cs.utexas.edu/~eunsol/courses/data/bitter_lesson.pdf)
2. [Medium - The Bitter Lesson](https://medium.com/towards-nesy/the-bitter-lesson-1a1d282ae1b9)
3. [arXiv - The Bitter Lesson of Scaling](https://arxiv.org/abs/2410.09649v1)
4. [Cognitive Medium - The Bitter Lesson](https://cognitivemedium.com/bitter-lesson)
5. [Xander.ai - Robotics and The Bitter Lesson](https://xander.ai/robotics-and-the-bitter-lesson)
6. [Exxact Corp - Revisiting Sutton's Bitter Lesson](https://www.exxactcorp.com/blog/Deep-Learning/compute-goes-brrr-revisiting-sutton-s-bitter-lesson-for-artificial-intelligence)
7. [Hacker News Discussion 1](https://news.ycombinator.com/item?id=42672790)
8. [Hacker News Discussion 2](https://news.ycombinator.com/item?id=40134071)
9. [Reddit - The Bitter Lesson 2.0](https://www.reddit.com/r/MachineLearning/comments/10ag9id/d_bitter_lesson_20/)

### **References 2: AI History**
1. [IBM - History of AI](https://www.ibm.com/think/topics/artificial-intelligence)

### **References 3: Cake Analogy & Reinforcement Learning**
**Cake Analogy:**
1. [SyncedReview - Yann LeCun’s Cake Analogy 2.0](https://medium.com/syncedreview/yann-lecun-cake-analogy-2-0-a361da560dae)
2. [YouTube - Yann LeCun on Self-Supervised Learning](https://www.youtube.com/watch?v=OuntI2Y4qxQo)

**"Deep Reinforcement Learning is a waste of time":**
1. [Jtoy.net Blog](http://www.jtoy.net/blog/deep-reinforcement-learning-is-a-waste-of-time.html)
2. [Hacker News Discussion 1](https://news.ycombinator.com/item?id=27794248)
3. [Hacker News Discussion 2](https://news.ycombinator.com/item?id=27868539)
4. [Medium - Why Reinforcement Learning Became Uncool](https://machine-learning-made-simple.medium.com/why-reinforcement-learning-became-uncool-and-how-it-might-come-back-7e791efbac24)
5. [Plain English - Reinforcement Learning is Dead, Long Live the Transformer](https://plainenglish.io/blog/reinforcement-learning-is-dead-long-live-the-transformer)

**RL is Not Dead:**
1. [Medium - Reinforcement Learning Isn’t Dead, It’s Evolving](https://medium.com/illumination/reinforcement-learning-isnt-dead-it-s-evolving-f8702b101f9e)

### **References 4: Next Generation Reasoning Models**
- **Traits of next-generation reasoning models:**
    - [06/05/2025, AIEWF] Next-generation RLVR models

---

## **Design Elements**
- **Color Palette:** 
    - Primary Background: Deep navy (#0a192f)
    - Accent Color: Bright teal (#64ffda) 
    - Text: Light blue-white (#e6f1ff)
    - Secondary: Muted blue-gray (#8892b0)
- **Typography:** 
    - Font Family: Roboto (Google Fonts)
    - Headings: 36px bold, teal accent color
    - Body Text: 18px regular, high contrast
    - Code/Technical: Monospace font for code examples
- **Visual Style:** 
    - Minimalist design with ample white space
    - Interactive elements where possible (hover states, animations)
    - Data visualizations using Chart.js or D3.js
    - SVG icons and diagrams for scalability
- **Layout:** 
    - 1280x720 slide dimensions
    - 40px padding, consistent spacing
    - Maximum 4-5 bullet points per slide
    - Left-aligned content with visual elements on right when applicable
- **Images:** 
    - High-quality PNG/SVG visuals
    - Consistent illustration style
    - Evolution roadmaps and flowcharts
    - Model architecture diagrams