---
marp: false
---

> **Video Title:** The Age of Gen AI: From AlexNet to ChatGPT

> **Video Description:**
> What is Generative AI, really? How did we get from basic AI to tools like ChatGPT, Gemini, and Midjourney so fast? In this video, I share my perspective on the complete history of modern AI—the key breakthroughs, the philosophical shifts, and where I think we're heading next.
>
> **Links mentioned in this video:**
> - [Link to presentation slides]
> - [Link to Rich Sutton's "The Bitter Lesson"]
> - [Link to Yann LeCun's Cake Analogy]
> - ... and other resources.

---
### Introduction [Slides 1-2: Title + Table of Contents]

What is Generative AI, really? It feels like ChatGPT, Midjourney, and these tools appeared overnight—but the story behind them is decades in the making.

I want to share my perspective on that journey: how we got here, where I think this is heading, and some insights I've developed from working in this space for over a decade.

---
### Part 1: The Foundation [Slides 3-4: Divider + GenAI Capabilities]

Let me quickly set the stage before we get to the interesting parts.

GenAI has three core capabilities. **Text and Code Generation**—tools like ChatGPT and Copilot. **Image and Video Generation**—Midjourney, DALL-E, Sora. You've probably seen that viral image of the Pope in a puffy jacket—completely AI-generated. And most interesting to me: **Actions and Tool Use**, where AI stops talking and starts *doing* things in the real world.

---
### What Makes GenAI Different? [Slide 5]

What's behind all this? A few key things.

First, **Model Scale**—billions to trillions of parameters. This enables **Foundation Models**—as you can see in this diagram, massive external data flows into one model, which can then be prompted for almost any task. No more building separate AIs for each application.

These models are **Multimodal**—text, images, audio, video. And you interact through **Natural Language**, not code.

The biggest shift is **Generation vs. Recognition**. For decades, AI recognized what was in an image. Now it creates entirely new images, text, code—things that never existed before.

---
### Foundation Models vs. Pre-trained Models [Slide 6]

This is worth understanding. In the 2010s, we had pre-trained models—trained on labeled datasets like ImageNet. Expensive, slow, and you needed a new model for every task.

Today's Foundation Models are trained on the entire internet using **self-supervised learning**—the model learns without human labels. In 2012, building a cat-vs-dog classifier took weeks. Today, you just ask.

---
### The Evolution of AI [Slide 7]

This diagram shows how it all fits together—a 70-year evolution where each layer builds on the last.

**1950s: Classical AI**—hand-written rules.
**1980s: Machine Learning**—learned from data, but humans specified the features.
**2010s: Deep Learning**—learned features automatically, but still one model per task.
**2020s: Generative AI**—one model, almost any task. And at the center: Large Language Models.

Each step traded manual effort for learned capability. But this progression also comes with a trade-off that I think is underappreciated.

---
### The Intelligence Trade-off [Slides 8-9]

This evolution brings up something I think about a lot: the Intelligence Trade-off. As AI systems become more capable, we lose direct control and interpretability. You can't have both.

Think about it:
- A simple calculator is 100% controllable and predictable.
- A deep learning model that recommends movies is a bit of a "black box."
- A generative AI that writes a movie script is even less predictable, but far more powerful. It can be creative, but it can also make things up—what we call "hallucination."

It's like an organizational hierarchy. You give an entry-level employee strict instructions. You give a CEO high-level goals and trust them to figure out the details. You trade control for capability.

Think of it like programming languages. With Assembly, you have total control over every instruction. With Python, you give that up for productivity. And now with AI coding agents, you describe what you want in plain English and trust the system to figure it out. Each step trades control for capability.

So when people ask, "Shouldn't AI be explainable?" The answer is: it's a spectrum. For a critical medical diagnosis, you want a simple, interpretable model. For discovering new drugs or creating art, you embrace the trade-off to unlock its full potential. You still want AI to explain its reasoning and make meaningful choices—but you also need to give it agency and creative freedom when you're setting high-level goals, tackling hard unstructured problems, or operating in unpredictable environments over extended periods.

---
### Part 2: The History [Slides 10-11: Divider + Personal Journey]

Now that we have the foundations, let's look at the history. To really understand why GenAI works now, you have to understand why other approaches *didn't* work before.

I want to share my own journey through some of these challenges. Here's a timeline of problems I've been thinking about for over two decades.

In **2000**, I realized I'm lazy—I can't stand inefficiency or monotonous tasks—and I wanted to automate everything in my life.

By **2005**, I could automate a lot with software, but it took forever to write. Going from idea to implementation was painfully slow, and I wanted to operate at the speed of thought.

Then I got into **robotics**. We'd been automating things in the virtual space for decades, but we kept ignoring the hard problems in the physical world. I felt a need to build robots that could solve real problems—take care of elderly parents when I'm not around, save lives in hospitals and on the roads.

Then I hit a wall. In **2011**, I realized robots need common sense. I started thinking about how to structure all of human knowledge into a database of common sense for machines.

In **2013**, while working on self-driving cars, I had another realization: computers can't process everything. I naively thought we could see 360 degrees all the time to make cars safe, and plan much faster than humans. But that wasn't true. Everyone in the industry was masking parts of sensors, subsampling data, planning at only 10 Hz. As humans, we dynamically shift our mental workload—when walking on slippery roads, we focus on balance and control more intentionally, less on obstacles far ahead. But most other times, we're focused ahead and balance runs on autopilot. Dynamic allocation of resources is essential.

In **2014**, I noticed something else: interfaces in computer systems are too rigid and brittle. As humans, we can learn to drive one car and easily transfer to another, or learn to operate a camera just by reading the manual. Computers, on the other hand, need fixed interfaces—and that wouldn't work long-term for robotics and intelligence.

---
### How GenAI Addressed These Challenges [Slide 12]

Now here's the interesting part—how did GenAI actually address these challenges?

**Automate everything** isn't solved, but it's accelerating. We're seeing real progress with AI assistants across many domains.

**Operate at the speed of thought**—this is actually becoming possible with coding agents like Claude Code. It's incredibly exciting. Still a long way to go, but the gap between idea and implementation is shrinking fast.

**Robots** are becoming a reality. Self-driving cars are here and scaling up. General home robots are being built and gaining capability—expect a lot of demos in 2026.

Now the last three—these are where GenAI really delivered.

**Common sense**—to my surprise, it started with ImageNet. Just image-based common sense, but that's what sparked the deep learning revolution. And later, in a very similar way, we did the same thing with internet-scale text instead of internet-scale images. No additional structuring needed—just letting efficient models learn.

**Dynamic compute allocation** has been key. Without the Attention mechanism dynamically picking which parts of the input to focus on—and enabling parallelization—we wouldn't have been able to scale and learn common sense from internet-scale text. Mixture of Experts and Chain of Thought reasoning are more ways we're expanding on this idea.

And finally, **no fixed interfaces**. Foundation models with zero-shot learning and prompt engineering have freed us from rigid APIs. Today you can dynamically call tools, use protocols like MCP, learn from huge sets of documents using RAG—and with multimodal models, you can input and output images, videos, and audio seamlessly.

It's remarkable to see these once-theoretical ideas become core features of tools we use every day.

---
### The Comeback of Classical AI [Slides 13-15: AIMA TOC + Book + Classical AI Comeback]

Here's something that might surprise you. Take a look at this table of contents—I've highlighted the key topics. Intelligent Agents. Planning. Reinforcement Learning. These are all the hot topics of today.

What year do you think this was published?

This is from "Artificial Intelligence: A Modern Approach"—the definitive AI textbook—and it was published in **1995**.

The core ideas behind AI agents are decades old. So why did they fail back then? Because they were brittle, they lacked common sense, and they only worked in extremely limited environments.

Why do they work now? Because Large Language Models provide the missing ingredient: a massive database of common sense and world knowledge. Classical AI concepts are making a comeback because LLMs can serve as their intelligent "brain."

---
### The Four Types of Learning [Slide 16]

To understand the next breakthrough, we need to talk about the four types of learning in AI.

We have **Supervised Learning** (learning from labeled examples), **Unsupervised Learning** (finding patterns in unlabeled data), and **Reinforcement Learning** (learning from trial and error).

But the real hero of this story is the fourth one: **Self-Supervised Learning**. It's a clever way for a model to learn from unlabeled data—like text from the internet—by creating its own goals, such as predicting the next word in a sentence. This is what enables training at planetary scale.

---
### The Rise and Fall of CNNs [Slides 17-20: CNN Era + Everything as Image + Scale + Can't Understand]

The modern AI era kicked off in 2012. Look at this graph—you can see traditional computer vision improving slowly, then AlexNet comes along and the accuracy jumps dramatically. That was the moment deep learning took over.

CNNs were so good at understanding images that for years, we tried to turn every problem into an image problem. We encoded road networks and audio spectrograms as images just to feed them into a CNN.

This led to a key discovery, captured well in this chart from Andrew Ng: with deep learning, scale was everything. Bigger neural networks trained on more data kept performing better—unlike traditional algorithms that plateaued. But there was a bottleneck: this required *labeled* data, which is expensive to create.

And even with all that scale, CNNs could never truly *understand*. They could tell you a face was in a picture, but not why the expression was funny or sad.

---
### The Hype and Heartbreak of Reinforcement Learning [Slides 21-23: RL Difficulty + Deep RL for AGI + AlphaGo]

At the same time, Reinforcement Learning was having its moment. In 2016, DeepMind's AlphaGo beat the world's best Go player, and many thought RL was the path to general intelligence.

But the hype met reality quickly. RL was plagued by problems. One was "reward hacking"—in one famous example, an AI supposed to race a boat figured out it could score more points by driving in circles collecting powerups instead of finishing the race.

RL was also incredibly inefficient. By 2019, the frustration was so high that you'd see influential engineers writing posts titled "Deep Reinforcement Learning is a waste of time." And at that moment, they weren't entirely wrong.

---
### The Cake Analogy [Slide 24]

This brings me to one of my favorite concepts, from AI pioneer Yann LeCun: the Cake Analogy. Look at this image—it explains why self-supervised learning is so powerful.

If intelligence is a cake, then **Reinforcement Learning is the cherry**. It provides a tiny signal—a single reward number every once in a while. Very few bits of information.

**Supervised Learning is the icing**. You get a label for every example—much more information.

But **Self-Supervised Learning is the génoise—the entire cake**. By predicting missing parts of its input—like the next word in a sentence—the model gets millions of bits of information from every sample.

This is why GPT works. Predicting the next word on a massive chunk of the internet is self-supervised learning at an almost unimaginable scale. It's how we finally built the "cake" of world knowledge.

---
### The Bitter Lesson [Slides 25-26]

There's another foundational idea I want to share—a famous essay called "The Bitter Lesson" by Rich Sutton.

The lesson is this: for 70 years, we've learned that trying to hand-code human knowledge into AI doesn't work in the long run. The methods that win are general-purpose methods like **Search** and **Learning**, because they get better as computers get faster.

History proves this repeatedly. In chess, brute-force search beat human strategy. In Go, massive self-play beat human intuition. In language, scale beat hand-coded grammar rules. The lesson is controversial, but history suggests that scale wins.

---
### Part 3: The Technical Breakthrough [Slides 27-28: Divider + Transformers]

So we had the history and the philosophy—but what was the actual technical invention that sparked the generative AI explosion?

It came in 2017, in a paper from Google titled "Attention Is All You Need." It introduced the **Transformer architecture**—this is the diagram that launched a thousand AI companies.

This was revolutionary for three reasons:
1. **Self-Attention:** Every word in a sentence could look at every other word simultaneously, solving a key limitation of older models.
2. **Parallelization:** It eliminated sequential processing, enabling incredibly fast training on modern GPUs.
3. **Token-based Design:** It treated everything as a "token"—a general-purpose idea that later allowed the architecture to extend beyond text to images, audio, and more.

---
### The Birth of ChatGPT and the RL Renaissance [Slides 29-31: GPT Emerges + ChatGPT + RL Renaissance]

By 2019, while everyone was still criticizing Reinforcement Learning, the first GPT models showed something remarkable. The simple, self-supervised goal of "predict the next word," combined with the Transformer architecture and massive scale, led to surprising, *emergent* intelligence. No one had explicitly taught it to reason or code—it just learned.

So if the base GPT model was the "cake" of knowledge, what made ChatGPT special? It added the icing and the cherry. It was fine-tuned on human conversations and optimized with **Reinforcement Learning from Human Feedback (RLHF)** to be helpful and safe. Think of it like a brilliant expert in a field—someone with deep knowledge but poor social skills. They know everything, but they can't explain it in a way that's useful to others. With a bit of training on how to communicate, suddenly that expertise becomes accessible to everyone.

And this brings us full circle. Remember when RL was called a "waste of time"? It turns out RL wasn't dead—it was waiting for the cake. 

Think about it this way: when learning to ride a bicycle, you can use pure trial and error—RL works fine. But learning to fly a commercial plane? You need to learn a lot of unintuitive things first—physics, airspace regulations, aircraft performance limits, emergency procedures. You build that foundation of knowledge, and then you still need many hours of flight experience at the end to calibrate and develop well-rounded expertise in both understanding and execution. That's exactly what happened with AI. RL alone wasn't enough. But RL on top of a foundation model? That's powerful.

We're now in an RL Renaissance, where it's used to create powerful reasoning systems and coding agents. It just needed that foundation of world knowledge to work.

---
### Part 4: Today & The Future [Slides 32-35: Divider + Trends + NVIDIA Stages + Agent Architecture]

So where are we now, and what's next?

In 2025, the capabilities are remarkable. We have real-time conversational AI, multi-modal models that understand everything, and coding agents that are becoming true partners. Industries from healthcare to software are being transformed.

This diagram from NVIDIA captures the trajectory well. We started with **Perception AI** in 2012—speech recognition, object detection. Then came **Generative AI**—content creation, digital marketing. Now we're entering **Agentic AI**—coding assistants, customer service agents. And on the horizon: **Physical AI**—self-driving cars and general robotics.

Agentic systems are really just systems built around LLMs. Think of LLMs as intelligence engines. Just like we design cars and airplanes to derive value from combustion or jet engines, we need to build agentic systems around LLMs to unlock their potential.

Here are the common components: **Tools**—calculators, code interpreters, web search. **Memory**—both short-term context and long-term recall. **Sub-agents**—specialized modules that handle specific tasks. And **planning**—chain of thought reasoning, self-reflection.

These systems are already proving valuable in real applications—coding assistants like Cursor and GitHub Copilot, customer service agents, research assistants that can browse and synthesize information. This architecture will power the next wave of AI applications.

---
### The Next Great Challenges [Slide 36]

I'm actually more excited about what comes next. AI agents are great—they'll drive a lot of commercial value. But there's growing consensus on what we need to solve to take intelligence to a whole new level.

1. **World Models:** Next-word prediction doesn't teach AI about physics or 3D space. We need models with true, intuitive understanding of the real world—likely trained from video. This is essential for safer robotics, but also for any application where spatial reasoning matters—making movies, physics simulations, even modeling climate change.

2. **Persistent Memory:** Without it, it's like having smart employees who act like it's their first day at the company every single day. Just like employees build up knowledge about how a company operates, AI agents need effective memory to be useful for personal assistants, virtual employees, and long-running tasks.

3. **Continuous Learning:** You don't want to keep storing everything to memory and retrieving it, or have bloated context windows. Just like we build new skills into muscle memory for better efficiency and reliability over time, we need LLMs that can self-reflect and consolidate information—a kind of "sleeping on it" effect, distilling daily experiences into lasting capability.

---
### Jagged Performance [Slide 37]

These models have what I'd call a "jagged frontier" of performance. But here's the thing—this isn't new. ML and deep learning have always had jagged performance. It didn't start with LLMs. The difference now is that these systems are being deployed in high-stakes environments where failures are more visible and consequential.

---
### Risks [Slides 38-40: Robot Crash + AI Minister]

Responsible development is essential. It was fine to have agents do read-only operations—like searching the web. But as we give them more power to write, to pay, to interact with other agents, we need robust guardrails, rigorous testing, and clear accountability. Don't underestimate this—AI agents with large-scale reach across the internet can cause significant damage too.

And for physical AI, the stakes are even higher. Look at the robot crashing in a warehouse. These systems can be superhuman one moment and fail comically the next. Physical AI needs even more testing and guardrails—the same level of scrutiny we've applied to self-driving cars—because failures in the real world have real consequences.

And we're already seeing real-world consequences with LLMs too. Albania appointed an AI as a government minister—raising serious questions about accountability and oversight. And Deloitte had to issue refunds after their AI-generated report contained fabricated citations. These aren't hypotheticals—they're happening now, and they demand responsible oversight. 

So, what are the key takeaways?
1. **Scale is about efficiency, not just raw resources.** Algorithmic breakthroughs are important—sometimes the key enabler.
2. **Natural language is the new universal interface** for computers. It's democratized access to intelligence—everyone can vibe code, ask for advice, non-ML engineers can start integrating ML solutions. So sit tight and expect many more AI agent innovations to come and transform everything, improving quality of life.
3. **There is a fundamental trade-off between intelligence and control.** We need to learn to understand, characterize, and trust these systems more, but not blindly and not without guardrails. We need to learn where and how much to trust, and how to safeguard.
4. The next frontiers are **world models, memory, and continuous learning.**
5. And **responsible deployment is paramount.**

---
### Closing Thoughts [Slide 42: Thank You]

The Age of GenAI is a fundamental shift in how we interact with technology and with intelligence itself. The question isn't *if* it will transform your work and life, but how quickly you can adapt.

I hope this gave you a clearer picture of this field. If you found it useful, I'd appreciate you sharing it. And if you have thoughts on the biggest challenges or most exciting applications of AI, I'd genuinely like to hear them.

Thanks for watching.

---
<!-- Slides 43-49: References and Appendix - not covered in video narration -->
