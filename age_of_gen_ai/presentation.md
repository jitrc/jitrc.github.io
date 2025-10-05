---
marp: true
theme: default
paginate: true
style: |
  .divider {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    text-align: center;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .title {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    text-align: center;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
  }
---

<!-- _class: title -->

![bg opacity:0.3](https://raw.githubusercontent.com/tensorflow/playground/master/preview.png)

# Age of Gen AI

**Evolution, Capabilities, and Future Trends**

**Jit Ray Chowdhury**
2025

---

## Table of Contents

- **Part 1:** Foundation
- **Part 2:** Historical Context & Key Concepts
- **Part 3:** Technical Deep Dive
- **Part 4:** Current State & Future Trends

---

<!-- _class: divider -->

# Part 1: Foundation

---

## GenAI Capabilities

![bg right:35%](https://raw.githubusercontent.com/ashishpatel26/Tools-to-Design-or-Visualize-Architecture-of-Neural-Network/master/images/AI_banner.png)

**Text/Code Generation:**
- Natural language understanding and generation
- Code completion and generation
- Document creation and summarization

**Image/Video Generation:**
- Text-to-image creation (DALL-E, Midjourney, Stable Diffusion)
- Style transfer and image editing (NanoBanana, Qwen3-Image-Edit)
- Video synthesis from prompts (Sora, Runway ML, Kling 2.1)

**Actions/Tool/API Calls:**
- Autonomous agents executing tasks
- Tool use and reasoning capabilities
- API integration for real-world actions

---

## What Makes GenAI Different?

![bg right:40%](https://miro.medium.com/v2/resize:fit:1400/1*7bRSYH6g6Eec2T3Nh_J73w.png)

**Foundation Models:** Pre-trained on vast datasets, adaptable to multiple tasks
- Traditional pre-trained models (ImageNet/COCO) were task-specific
- Foundation models enable multi-task, zero-shot learning

**Multimodal Capabilities:** Processing and generating text, images, audio, video
- Custom token support for Actions, Graphs, and Road Networks

**Natural Interfaces:** Conversational interaction using everyday language
- Advancement over earlier NLP models (LSTMs, RNNs)

**Generation vs. Recognition:** Creating new content rather than classifying
- Open-vocabulary image detection and any-text-to-image generation

**Scale:** Billions to trillions of parameters

---

## Foundation Models vs. Pre-trained Models

**Traditional Pre-trained Models (2010s):**
- ImageNet (1000 categories), COCO - first "common sense" dataset for vision
- Supervised learning, required extensive fine-tuning per task
- Task-specific: Separate models per application
- Expensive: New labeled datasets per task

**Foundation Models (2020s):**
- Internet-scale training (trillions of tokens), self-supervised
- Emergent capabilities: Reasoning, common sense, zero/few-shot learning
- Multimodal: Text, images, audio, code, actions

**Key Shift:**
- **Scale:** 1000 categories → trillions of tokens
- **Adaptation:** Fine-tuning → prompting (natural language programming)
- **Capability:** Single task → universal multi-task

---

## Evolution from Traditional AI to GenAI

![bg right:40%](https://miro.medium.com/v2/resize:fit:1400/1*VNpCOLa4zmzS5sDGSNcGjw.png)

**AI: Artificial Intelligence (1950s)**
- Expert systems for medical diagnostics
- Rule-based systems for path planning (early GPS)
- Logical reasoning and theorem proving

**ML: Machine Learning (1980s)**
- SVMs for image classification and text categorization
- Decision Trees/Random Forests for pose estimation (Kinect)

**DL: Deep Learning (2010s)**
- CNNs for object detection in autonomous vehicles
- RNNs for language translation and sequence prediction

**GenAI: Generative AI (2020s)**
- GPT: Predict the next item in a sequence
- DALL-E: Realistic image generation from text

---

## AI Evolution: From Narrow to Broad Capabilities

![bg right:40%](https://ourworldindata.org/uploads/2024/12/AI-parameters-Epoch-1536x1096.png)

**AI (1950s):** Mimic human intelligence
- Hand-written code/algorithms by domain experts

**ML (1980s):** Learning trends from historical data
- Hand-crafted features per domain/application
- Function fitting on features

**DL (2010s):** Learning features directly from data
- Same architecture with automatic feature extraction
- Different trained weights per task
- Large-scale datasets

**GenAI (2020s):** Large foundation models
- Same trained model for zero-shot application across tasks
- Internet-scale training (trillions of tokens)
- Billions to trillions of parameters
- Multimodal I/O (Text, Image, Audio, Actions, API Calls)

---

## AI, Uncertainty, and the Intelligence Trade-off

> **Key Insight:** As we embrace higher levels of intelligence and autonomy, we must also accept more uncertainty. You can't expect a rule-based algorithm to work well on unstructured data, nor expect a smart model to always follow the same rules without creativity.

**AI's Embrace of Uncertainty:**
- **Traditional AI (1950s):** Deterministic, probabilistic, provable, introspectable
- **ML (1980s):** Data-driven decision making from extracted features
- **Deep Learning (2010s):** "Black box" models, limited explainability
- **GenAI (2020s):** Handles ambiguity and unstructured data, but hallucinates

**Organizational Hierarchy Analogy:**
- **Entry Level:** Strict rules, clear instructions
- **Middle Management:** Independent problem-solving
- **Executive Leadership:** High autonomy, strategic decision-making

**Probabilistic Robotics:** Models uncertainty (sensor noise, errors) for robust behavior in unpredictable environments

**Programming Language Analogy:**
- More optimization and control in Assembly/C than Python
- GenAI provides intuitive natural interface, but prompt engineering becomes necessary as we've embraced a more ambiguous interface

---

## Part 1 Recap

**What We Covered:**
- GenAI capabilities: Text, images, video, actions
- Foundation models vs. traditional pre-trained models
- Evolution: AI → ML → DL → GenAI
- Trade-off: Intelligence vs. uncertainty

**Key Takeaway:** GenAI represents a paradigm shift from task-specific to universal models

---

<!-- _class: divider -->

# Part 2: Historical Context & Key Concepts

---

## A Personal Journey: Early Insights

```
2000 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
     Automate everything
     "I am lazy, can't stand inefficiency"

2005 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
     Operate at speed of thought
     Software accelerates tasks... but still takes long to build

2008 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
     Make Robots
     Need robots for real-world problems

2011 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
     Common sense for robots
     How to collect and structure all human knowledge?

2013 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
     Dynamic compute allocation
     Can't process all sensors in detail; compromises needed

2014 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
     No fixed interfaces
     Adaptive control, transfer learning between devices
```

---

## How GenAI Addressed These Challenges

```
2000  ✗  Automate everything
         Progress with AI assistants, but not fully achieved

2005  ✗  Operate at speed of thought
         Coding agents helping, but not perfect yet

2008  ✗  Make Robots
         Self-driving is reality, but not general home robots

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

2011  ✓  Common sense for robots
         ImageNet: First common sense database
         Internet-scale text training with next-word prediction

2013  ✓  Dynamic compute allocation
         Attention mechanisms, MoE, Chain of Thought

2014  ✓  No fixed interfaces
         Foundation models, zero-shot learning, prompt engineering, RAG
```

---

## Types of Learning in AI

![bg right:35%](https://www.researchgate.net/publication/373535780/figure/fig1/AS:11431281210621453@1705501116569/Types-of-machine-learning-Machine-learning-encompasses-three-main-types-supervised.png)

**Supervised Learning:**
- Uses labeled data for training
- Historically most successful approach
- Examples: Image classification, speech recognition

**Unsupervised Learning:**
- No labels required, leverages larger datasets
- Pattern discovery (clustering, dimensionality reduction)

**Reinforcement Learning:**
- Trial-and-error learning with reward feedback
- **DEBATE:** Some believe path to AGI, others cite sample inefficiency
- Renaissance in combination with LLMs for reasoning tasks

**Self-Supervised Learning:**
- Best of Supervised and Unsupervised
- Learn with pseudo-labels but at large scale
- **Critical for foundation models like GPT, CLIP, DinoV3**

---

## Classical AI Concepts: Making a Comeback

![bg right:40%](https://miro.medium.com/v2/resize:fit:1400/1*bhFifratH9DjKqMBTeQG5A.gif)

**Why Classical Methods Failed Initially:**
- **RL:** Sample inefficient, lacked common sense
- **Agent-Based Systems:** Brittle, domain-specific rules
- **Search Algorithms:** Limited to well-defined state spaces

**Why They Work Now with LLMs:**
- **KEY ENABLER:** LLMs provide common sense reasoning and world knowledge
- **Better representations:** Natural language vs. hand-coded features
- **Hybrid approaches:** Neural reasoning + classical algorithms

**Modern Success Examples:**
- **Planning:** LLMs decompose complex goals into sub-tasks
- **Search:** Semantic search using embeddings
- **RL:** Language feedback instead of sparse numerical rewards

---

## The Bitter Lesson: Advice for AI Developers

**(From Rich Sutton's Essay, 2019)**

**Bet on Scaling, Not on Human Knowledge**
- General methods leveraging computation are most effective
- The real world is endlessly complex; build meta-methods that discover complexity

> *"The biggest lesson from 70 years of AI research is that general methods that leverage computation are ultimately the most effective, and by a large margin."*

**Embrace General-Purpose Methods: Search and Learning**
- Search and Learning are the only approaches that scale with computation
- History shows they outperform systems designed with human expertise

> *"We have to learn the bitter lesson that building in how we think we think does not work in the long run."*

---

## The Bitter Lesson: Proof in Practice

**Computer Chess (Deep Blue, 1997)**
- Brute-force search (200M positions/second) defeated Kasparov
- Hand-coded strategies proved inferior to computational search

**Computer Go (AlphaGo, 2016)**
- Deep learning + self-play discovered superior strategies
- Human intuition surpassed by computation

**Natural Language Processing (GPT-3)**
- Statistical models replaced hand-coded grammar rules
- Achieved nuanced human-like communication through scale

**Computer Vision (DINOv3, 2025)**
- Self-supervised learning surpassed supervised methods
- No hand-designed features required

---

## Yann LeCun's Cake Analogy

![bg right:35%](https://miro.medium.com/v2/resize:fit:1400/1*VT1TDnzhP4E9HhzkG6tZyA.png)

**Information Content During Learning:**

**Reinforcement Learning (the cherry - 1%):**
- Scalar reward signal given intermittently
- **Few bits per sample**

**Supervised Learning (the icing - 10%):**
- Category or numerical predictions for each input
- **10 to 10,000 bits per sample**

**Self-Supervised Learning (the cake - 89%):**
- **KEY INSIGHT:** Predicts any part of input from any observed part
- Future frame prediction, masked language modeling
- **Millions of bits per sample**

---

## The CNN Era (2012-2017)

![bg right:40%](https://miro.medium.com/v2/resize:fit:1400/1*vkQ0hXDaQv57sALXAJquxA.jpeg)

- **2012 BREAKTHROUGH:** AlexNet won ImageNet, deep learning revolution begins
- **Strong inductive bias:** Convolution perfectly suited for image structure
- **Everything-to-image:** Audio spectrograms, road networks as layers
- **Key datasets:** ImageNet, COCO
- **Architecture evolution:** ResNet, wider networks, pyramids
- **Current status:** Superseded by Vision Transformers (ViT, DINOv3)

---

## RL Excitement and Difficulty

![bg right:40%](https://miro.medium.com/v2/resize:fit:1400/1*nTdVlhOp-h2NfRoVnHYW3g.png)

- **Early promise:** Deep RL on Atari games, AlphaGo breakthrough
- **Persistent challenges:** Reward hacking, sparse reward signals
- **Sample inefficiency:** Millions of interactions needed for simple tasks
- **LIMITATION:** Worked well in constrained environments, struggled in open worlds
- **Example:** CoastRunners boat racing - agent farms powerups in circles instead of finishing race

---

## Deep RL for AGI Ambitions (2016 Peak)

**Key players:** DeepMind (AlphaGo), OpenAI (Dota 2, robotics)

**Major projects:**
- AlphaGo defeats Lee Sedol (2016)
- Atari game suite benchmarks
- OpenAI Gym standardized environments

**HISTORICAL NOTE:** This period saw peak optimism for RL as path to AGI

---

## Part 2 Recap

**What We Covered:**
- Personal journey: From automation dreams to GenAI reality
- Learning types: Self-supervised learning dominates (Yann's Cake)
- The Bitter Lesson: Computation scales, human knowledge doesn't
- Classical AI comeback: RL + agents work with LLM foundation

**Key Takeaway:** General methods + scale > hand-coded expertise

---

<!-- _class: divider -->

# Part 3: Technical Deep Dive

---

## Transformers Changed Everything (2017)

![bg right:45%](https://jalammar.github.io/images/t/transformer_resideual_layer_norm_3.png)

**SEMINAL PAPER:** "Attention Is All You Need" - Vaswani et al., Google, 2017

- **Self-attention mechanism:** Direct modeling of relationships between all positions
- **Revolutionary insight:** Eliminated recurrence, enabled full parallelization
- **PERFORMANCE:** Superior translation quality with less training time
- **Easy to Scale:** Faster, shallower network that handles more scale

---

## GPT Emerges, RL Winter Begins (2019)

**RL Criticism Peak - "Deep RL is a waste of time" (2019):**
- Still sample-inefficient despite years of research
- Limited to toy problems and constrained action spaces
- No common sense reasoning, excessive exploration

**GPT-1/GPT-2 Breakthrough:**
- **KEY INSIGHT:** Next-token prediction showed emergent intelligence
- Initially limited to text completion and basic coding
- **SURPRISE:** Scaling simple objective yielded complex behaviors

---

## How ChatGPT is Better than GPT

- Supervised Fine-Tuning (SFT) with RLHF
- Also called: instruction tuned, human alignment, preference optimization
- Basically tuned for humans to like it and communicate easily
- Without this, GPT was like a human expert with knowledge but poor social skills

---

## RL Renaissance: The Cherry on Top (2024-2025)

**Why RL Works Now (vs. 2019 criticism):**
- **Foundation models provide common sense:** LLMs solve the "cold start" problem
- **Language as reward:** Natural feedback instead of sparse numerical rewards
- **Verifiable environments:** Code execution, math proofs enable RLVR

**Modern Success Stories:**
- **OpenAI o1:** Extended thinking time via RL, 83% → 94% on AIME with more compute
- **DeepSeek R1:** Open-source reasoning model matching o1 performance
- **Code generation:** RL improves debugging, test-driven refinement
- **Contrast with 2019:** Same algorithms, but LLM backbone enables practical deployment

**Key Insight:** RL wasn't a waste—it needed the foundation model "cake" first

---

## GenAI Architecture Overview

![bg right:40%](https://miro.medium.com/v2/resize:fit:1400/1*nbiCv7Vxqd5P6y58NvwM5g.png)

- **Transformer Architecture:** Self-attention enabling parallel sequence processing
- **Diffusion Models:** Gradual denoising for high-quality image/video generation
- **Retrieval Augmented Generation (RAG):** Enhancing generation with external knowledge
- **Fine-Tuning:** From full fine-tuning to LoRA to prompt engineering
- **Mixture of Experts (MoE):** For efficient scaling


---

## Part 3 Recap

**What We Covered:**
- Transformers (2017): Self-attention revolutionized sequence modeling
- GPT breakthrough: Next-token prediction → emergent intelligence
- RLHF/SFT: Making models human-friendly (ChatGPT)
- RL Renaissance: Foundation models enable practical RL deployment
- Architecture: Transformers, diffusion, RAG, MoE

**Key Takeaway:** Transformers + scale + RL + optimization = practical GenAI

---

<!-- _class: divider -->

# Part 4: Current State & Future Trends

---

## Current GenAI Trends (2025)

![bg right:35%](https://images.ctfassets.net/fi0zmnwlsnja/23Zd1l5HaPb2J7tJkIHXTz/fe5c7e1dcfd27d1a5cfe0f1c9b0dd1e3/Gen_AI_Adoption.png)

- **Multi-modal AI:** Processing multiple data types simultaneously
- **AI-First Applications:** Software built with AI at its core
- **Specialized AI Agents:** Task-specific AI with enhanced capabilities
- **Generative User Interfaces:** AI-assisted interface design
- **Hyper-Personalization:** Tailored experiences
- **Conversational AI:** Advanced natural language interactions
- **Service as Software:** AI agents driving productivity gains
- **Healthcare Integration:** AI enhancing operations
- **Workplace Transformation:** AI empowering employees
- **Real-time AI Reasoning:** Enhanced voice interfaces

**Key Stat:** 71% of organizations regularly use Gen AI in at least one function (2025)


---

## AI Evolution Stages

![bg right:35%](https://www.nvidia.com/content/nvidiaGDC/us/en_US/glossary/generative-physical-ai/_jcr_content/root/responsivegrid/nv_container_392816145/nv_container/nv_image.coreimg.100.850.jpeg/1735078887580/gen-physical-ai-progression.jpeg)

**Perception AI (2012 AlexNet) - MATURE**
- Speech Recognition, Object Detection, Medical Imaging
- **Status:** Production-ready, widely deployed

**Generative AI (2020s) - MAINSTREAM**
- Digital Marketing, Content Creation, Image/Video Generation
- **Status:** Rapid adoption, improving quality

**Agentic AI (Current) - EMERGING**
- Coding Assistant, Customer Service, Patient Care
- **Status:** Early deployment, variable reliability, human oversight required
- **Maturity levels:** Coding (high) > Customer service (medium) > Complex reasoning (low)

**Physical AI (Next) - EARLY RESEARCH**
- Self-Driving Cars (L4/L5 in limited areas), General Robotics
- **Status:** Experimental, safety testing, limited deployment

---

## LLM-powered Autonomous Agent Systems

![bg right:40%](https://lilianweng.github.io/posts/2023-06-23-agent/agent-overview.png)

> **Systems Thinking:** LLM models are like powerful engines; you need systems like airplanes, cars, or factories to take advantage of them and produce value.

**Components:**
- **Foundation model:** Intelligence/reasoning engine
- **Tools:** Calculator, code interpreter, web search, APIs
- **Planning:** Chain-of-thought, self-reflection, subgoal decomposition
- **Memory:** Short-term context + long-term storage
- **Action execution:** Real-world task completion

**Applications:** Customer service, business optimization, code generation, research, personal assistants

---

## Next Challenges

**World Model:**
- **Problem:** Next-token prediction insufficient for 3D spatial reasoning
- **Data gap:** Scarcity of "boring" everyday activity data, limited 3D spatial data
- **Efforts:** GAIA-1 (Wayve), Sora's physics understanding, Genie 3
- **Need:** Video prediction models understanding object permanence, physics

**Persistent Memory:**
- **Problem:** Don't want AI "acting like it's their first day every day"
- **What to store:** Importance, recency, relevance (forgetting curve)
- **How to store:** Knowledge graphs, temporal sequences, hierarchical memory
- **Examples:** Mem0, MemGPT, long-term context architectures

**Continuous Learning:**
- **Need:** Models with "muscle memory" and continuous learning
- **Challenge:** Catastrophic forgetting vs. learning from mistakes
- **Analogy:** "Sleeping over it" effect for consolidation
- **Research:** Elastic Weight Consolidation, Progressive Neural Networks

**Specialization:**
- **Need:** Specialized, efficient, smaller, faster models
- **Approaches:** Fine-tuning, LoRA, distillation
- **Examples:** Domain-specific (medical, legal), task-specific (coding, math)

---

## Risks and Responsible Development

**Demonstrated Capabilities and Risks:**
- **FACT:** LLM-generated content passes Turing tests, enabling misinformation
- **Evolution:** Sandboxed → read-only internet → agentic write capabilities

**Current Risk Acceleration:**
- **CONCERN:** Agentic deployment without adequate guardrails
- **Vision-Language-Action (VLA) models:** Extending AI to physical robots
- **HISTORICAL PARALLEL:** Autonomous vehicles required extensive testing

**Accessibility vs. Safety Trade-off:**
- **RISK:** Easy deployment without training expertise
- No accountability frameworks for widespread AI deployment
- **RECOMMENDATION:** Careful evaluation before factory and home deployment

---

## References

**Additional Resources:**
- [NVIDIA - Generative Physical AI](https://www.nvidia.com/en-us/glossary/generative-physical-ai/)
- [Understanding Generative AI in Layman's Terms](https://pub.aimind.so/understanding-generative-ai-in-laymans-term-46e8d088659a)
- [IBM - History of AI](https://www.ibm.com/think/topics/artificial-intelligence)

---

## References: The Bitter Lesson

1. [The Bitter Lesson by Rich Sutton (PDF)](https://www.cs.utexas.edu/~eunsol/courses/data/bitter_lesson.pdf)
2. [Medium - The Bitter Lesson](https://medium.com/towards-nesy/the-bitter-lesson-1a1d282ae1b9)
3. [arXiv - The Bitter Lesson of Scaling](https://arxiv.org/abs/2410.09649v1)
4. [Cognitive Medium - The Bitter Lesson](https://cognitivemedium.com/bitter-lesson)
5. [Xander.ai - Robotics and The Bitter Lesson](https://xander.ai/robotics-and-the-bitter-lesson)
6. [Exxact Corp - Revisiting Sutton's Bitter Lesson](https://www.exxactcorp.com/blog/Deep-Learning/compute-goes-brrr-revisiting-sutton-s-bitter-lesson-for-artificial-intelligence)
7. [Hacker News Discussion 1](https://news.ycombinator.com/item?id=42672790)
8. [Hacker News Discussion 2](https://news.ycombinator.com/item?id=40134071)
9. [Reddit - The Bitter Lesson 2.0](https://www.reddit.com/r/MachineLearning/comments/10ag9id/d_bitter_lesson_20/)

---

## References: Self-Supervised Learning & RL

**Yann LeCun's Cake Analogy:**
- [SyncedReview - Yann LeCun's Cake Analogy 2.0](https://medium.com/syncedreview/yann-lecun-cake-analogy-2-0-a361da560dae)
- [YouTube - Yann LeCun on Self-Supervised Learning](https://www.youtube.com/watch?v=OuntI2Y4qxQo)

**"Deep Reinforcement Learning is a waste of time" (2019):**
- [Jtoy.net Blog](http://www.jtoy.net/blog/deep-reinforcement-learning-is-a-waste-of-time.html)
- [Hacker News Discussion 1](https://news.ycombinator.com/item?id=27794248)
- [Hacker News Discussion 2](https://news.ycombinator.com/item?id=27868539)
- [Medium - Why RL Became Uncool](https://machine-learning-made-simple.medium.com/why-reinforcement-learning-became-uncool-and-how-it-might-come-back-7e791efbac24)
- [Plain English - RL is Dead, Long Live the Transformer](https://plainenglish.io/blog/reinforcement-learning-is-dead-long-live-the-transformer)

**RL Renaissance:**
- [Medium - RL Isn't Dead, It's Evolving](https://medium.com/illumination/reinforcement-learning-isnt-dead-it-s-evolving-f8702b101f9e)

---

<!-- _class: divider -->

# Appendix

---

## A1: Challenges of Reinforcement Learning (RL)

**Difficulty of Reward Specification:**
- Designing effective reward functions without incentivizing unintended behavior (reward hacking)
- Hard to specify "good" outcomes precisely and comprehensively for complex tasks

**Sample Inefficiency:**
- Requires vast number of environment interactions to learn optimal policies
- Expensive, time-consuming, or dangerous in real-world applications (robotics, autonomous driving)

**Limited State and Action Spaces:**
- Historically successful only in constrained environments (board games, simple control systems)
- Real-world problems involve high-dimensional, continuous spaces making exploration extremely challenging

**Need for Common Sense and Guided Exploration:**
- Early RL struggled without prior knowledge or domain-specific heuristics
- Agents would "wander aimlessly" in vast, unstructured environments without guidance

---

## A2: Unsupervised Learning: Images vs. Language

**Challenges with Images:**
- Raw pixel data lacks explicit, discoverable hierarchical structures consistent across diverse images
- Pixel meaning highly context-dependent (red pixel could be car, flower, or brick wall)
- Unsupervised models often learned noisy or uninterpretable features without human-annotated labels

**Success with Language:**
- **Sequential and Contextual Nature:** Words appear in specific order conveying meaning; context provides strong learning signals
- **Statistical Regularities:** Rich patterns in word co-occurrence and grammatical structures
- **Hierarchical Structure:** Natural progression from characters → words → phrases → sentences → paragraphs
- **Predictive Tasks:** Masked language modeling (BERT) and next-token prediction (GPT) leverage inherent language structure

---

## A3: AI Agents: Resurgence of Early Concepts

**Early AI Limitations:**
- **Complexity:** Real-world environments overwhelmed available computational power
- **Uncertainty:** Assumed perfect knowledge of environment (rarely realistic)
- **Brittleness:** Rule-based systems failed when encountering unprogrammed situations
- **Lack of Common Sense:** Missing broad, intuitive understanding of the world

**LLMs Re-enabling Classical Methods:**
- **Broad World Knowledge:** Unprecedented breadth of general knowledge and common sense from vast training data
- **Natural Language Interface:** Human-like text understanding enabling intuitive goal specification
- **Reasoning and Planning:** Emergent capabilities for breaking down complex tasks into sub-goals
- **Tool Use:** Can be augmented with external tools (APIs, search engines, code interpreters)
- **Handling Unstructured Information:** Excel at processing real-world's dominant unstructured text format

---

<!-- _class: title -->

# Thank You

**Questions?**

Jit Ray Chowdhury
2025
