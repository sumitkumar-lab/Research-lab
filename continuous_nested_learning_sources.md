# Research Sources: Continuous & Nested Learning in AI
*Compiled for AI Research on Breaking Limited/One-Time Training Paradigm*

---

## 🎯 Core Breakthrough: Nested Learning (2025)

### **Primary Source: Google's Nested Learning Paper**
**Title:** "Nested Learning: The Illusion of Deep Learning Architectures"  
**Published:** NeurIPS 2025  
**Authors:** Ali Behrouz, Meisam Razaviyayn, Peilin Zhong, Vahab Mirrokni (Google Research)

**Key Innovation:** Treats ML models as nested, multi-level optimization problems rather than single monolithic systems.

**Links:**
- arXiv: https://arxiv.org/abs/2512.24695
- Google Research Blog: https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/
- NeurIPS Poster: https://neurips.cc/virtual/2025/poster/116123
- OpenReview: https://openreview.net/forum?id=nbMeRvNb7A
- Full PDF: https://abehrouz.github.io/files/NL.pdf

**Key Concepts:**
1. **Continuum Memory System (CMS):** Memory as a spectrum of modules updating at different frequencies
2. **Hope Architecture:** Self-modifying recurrent architecture built on Nested Learning principles
3. **Deep Optimizers:** Treats optimizers as learnable associative memory modules
4. **Multi-scale Updates:** Different components learn at different rates (like human brain)

**Why It Matters:**
- Addresses catastrophic forgetting through architectural redesign
- Models can self-improve and continuously learn
- Moves beyond "more layers = better" paradigm
- Provides theoretical framework for continual learning

---

## 📚 Foundational Papers on Continual Learning

### **1. Catastrophic Forgetting - The Core Problem**

**Elastic Weight Consolidation (EWC)**
- Paper: "Overcoming catastrophic forgetting in neural networks" (PNAS, 2017)
- Authors: Kirkpatrick et al.
- Link: https://www.pnas.org/doi/10.1073/pnas.1611835114
- Method: Selectively constrains updates to parameters crucial for previous tasks
- Impact: One of the most cited solutions to catastrophic forgetting

### **2. Comprehensive Review Papers**

**"Continual Learning and Catastrophic Forgetting"**
- Authors: Gido M. van de Ven et al.
- Date: March 2024
- Link: https://arxiv.org/html/2403.05175v1
- Covers: 6 main computational approaches (replay, parameter regularization, functional regularization, optimization-based, context-dependent processing, template-based)

**"The Future of Continual Learning in the Era of Foundation Models"**
- Date: June 2025
- Link: https://arxiv.org/html/2506.03320v1
- Focus Areas:
  - Continual Pre-Training
  - Continual Fine-Tuning
  - Continual Compositionality & Orchestration (CCO)
- Key Insight: CCO offers most promising path via modular, decentralized model ecosystems

**"Recent Advances of Continual Learning in Computer Vision"**
- Publisher: IET Computer Vision (Wiley)
- Date: March 2025
- Link: https://ietresearch.onlinelibrary.wiley.com/doi/abs/10.1049/cvi2.70013
- Focus: Computer vision applications of continual learning

---

## 🧠 Neuroscience-Inspired Approaches

### **Hybrid Neural Networks (Corticohippocampal Circuits)**
- Paper: "Hybrid neural networks for continual learning inspired by corticohippocampal circuits"
- Publisher: Nature Communications (Feb 2025)
- Link: https://www.nature.com/articles/s41467-025-56405-9
- Innovation: Emulates dual memory representations (specific vs. generalized) from biological systems
- Key Feature: Task-agnostic, no memory increase required

### **Personalized AGI via Neuroscience Principles**
- Paper: "Personalized Artificial General Intelligence (AGI) via Neuroscience-Inspired Continuous Learning Systems"
- Date: April 2025
- Link: https://arxiv.org/abs/2504.20109
- Covers:
  - Synaptic Pruning
  - Hebbian Plasticity
  - Sparse Coding
  - Dual Memory Systems
- Application: Edge AI, mobile assistants, humanoid robots

### **Spiking Neural Networks (SNNs) for Continual Learning**
- Review: "Review of deep learning models with Spiking Neural Networks"
- Link: https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1623497/pdf
- Focus: Energy-efficient, biologically-inspired temporal processing
- Applications: Medical imaging, neuroimaging

---

## 🔧 Practical Methods & Techniques

### **Six Main Computational Approaches:**

1. **Replay Methods**
   - Store subset of previous data
   - Periodically retrain on old + new data
   - Example: CORE (Cognitive Replay) - Zhang et al., 2024

2. **Parameter Regularization**
   - EWC (Elastic Weight Consolidation)
   - Protect important weights for old tasks
   - Theoretical foundation well-established

3. **Functional Regularization**
   - Preserve output behavior rather than parameters
   - Knowledge distillation approaches

4. **Optimization-Based Approaches**
   - UniGrad/UniGrad-FS - Li et al., 2024
   - C-Flat (Continual Flatness) - Bian et al., 2025
   - SharpSeq - Le et al., 2024
   - Modify gradient updates to reduce conflicts

5. **Context-Dependent Processing**
   - Different network parts for different tasks
   - Orthogonal Weights Modification (OWM) - Zeng et al., 2018

6. **Template-Based Classification**
   - Use exemplars/prototypes
   - Efficient memory management

---

## 💡 Key Research Resources

### **Bing Liu's Lifelong Learning Lab (UIC)**
- Website: https://www.cs.uic.edu/~liub/lifelong-learning.html
- Comprehensive resource collection
- Books: "Lifelong Machine Learning" (Morgan & Claypool, 2018)
- Recent papers on:
  - Continual Compositionality & Orchestration
  - Learning after deployment
  - Open-world continual learning

### **2025 Deep Learning Roadmap**
- Blog: Medium - "The 2025 Deep Learning RoadMap"
- Link: https://medium.com/javarevisited/the-2024-deep-learning-roadmap-f4179458e1e3
- Practical learning path for deep learning fundamentals

---

## 🎯 First Principles Approach

### **Understanding the Problem:**

1. **Stability-Plasticity Dilemma**
   - Stability: Retain old knowledge
   - Plasticity: Learn new information
   - Challenge: Balance both without catastrophic forgetting

2. **Why Traditional Networks Fail:**
   - Weights optimized for Task A get overwritten for Task B
   - No mechanism for importance-based preservation
   - Static architecture can't adapt to changing data distributions

3. **Human Brain Analogy:**
   - Neuroplasticity enables continuous learning
   - Multi-timescale memory consolidation
   - Hierarchical processing at different speeds

### **Solution Directions:**

**A. Architecture-Level (Nested Learning approach)**
- Treat model + optimizer as unified learning system
- Multiple nested optimization levels
- Each level updates at different frequency
- Memory as continuum, not discrete layers

**B. Training-Level (Traditional CL approaches)**
- Selective parameter updates (EWC)
- Memory replay mechanisms
- Task-specific network components
- Meta-learning for fast adaptation

**C. Hybrid Approaches**
- Combine architectural innovation with training strategies
- Use biological inspiration (SNNs, dual memory)
- Modular, compositional systems (CCO)

---

## 📊 Key Metrics & Evaluation

### **Performance Metrics:**
- Average accuracy across all tasks
- Backward transfer (performance on old tasks)
- Forward transfer (benefit for new tasks)
- Memory efficiency
- Computational cost

### **Benchmark Datasets:**
- Split MNIST (permuted versions)
- CIFAR-100 (class-incremental)
- Tiny-ImageNet
- Task-specific benchmarks

---

## 🚀 Emerging Trends (2025)

1. **Foundation Model Continual Learning**
   - Adapting LLMs without full retraining
   - Parameter-efficient fine-tuning (LoRA, AutoLoRA)
   - Continual pre-training strategies

2. **Federated & Decentralized Learning**
   - Distributed continual learning
   - Privacy-preserving approaches
   - Edge AI deployment

3. **Self-Modifying Architectures**
   - Hope (Google's implementation)
   - Learning to learn (meta-learning++)
   - Automatic architecture search for CL

4. **Multi-Modal Continual Learning**
   - Cross-modal knowledge transfer
   - Unified representations across modalities

---

## 📖 Recommended Reading Order

### For Building First Principles Understanding:

1. **Start Here:**
   - Van de Ven et al. (2024) - "Continual Learning and Catastrophic Forgetting"
   - Comprehensive overview with clear taxonomy

2. **Core Problem:**
   - Kirkpatrick et al. (2017) - EWC paper
   - Understanding catastrophic forgetting fundamentally

3. **New Paradigm:**
   - Behrouz et al. (2025) - "Nested Learning" paper
   - Paradigm shift in thinking about learning systems

4. **Practical Applications:**
   - Liu's Lifelong Learning website
   - Recent papers on real-world deployment

5. **Neuroscience Connection:**
   - Hybrid NN paper (Nature, 2025)
   - Personalized AGI paper (2025)

---

## 🔬 Research Opportunities

### **Unexplored Areas:**

1. **Scaling Nested Learning**
   - How does it perform at GPT-4 scale?
   - Computational efficiency at large scales
   - Integration with existing LLM architectures

2. **Hybrid Approaches**
   - Combining nested learning + neuroscience principles
   - SNNs with nested optimization
   - Multi-agent nested learning systems

3. **Theoretical Foundations**
   - Mathematical guarantees for forgetting bounds
   - Convergence properties of nested optimizers
   - Information theory perspectives

4. **Practical Deployment**
   - Edge device continual learning
   - Real-time adaptation in production
   - Human-in-the-loop continual learning

---

## 💻 Implementation Resources

### **Tools & Frameworks:**
- Avalanche: Comprehensive continual learning library
- Continuum: Lightweight CL framework
- PyTorch/TensorFlow: Base frameworks

### **Code Repositories:**
- Check paper GitHub links (when available)
- Avalanche library examples
- Continual learning benchmarks

---

## 📌 Key Takeaways

1. **Nested Learning is the newest paradigm** (NeurIPS 2025) - most relevant to your research
2. **Catastrophic forgetting is the core challenge** - must be addressed at architectural level
3. **Multi-timescale updates are crucial** - inspired by biological systems
4. **The field is rapidly evolving** - 2024-2025 seeing major breakthroughs
5. **Practical deployment still challenging** - gap between theory and production systems

---

## 🎓 Next Steps for Your Research

1. **Deep dive into Nested Learning paper** - This is most aligned with your hypothesis
2. **Understand EWC and replay methods** - Baseline approaches to compare against
3. **Explore neuroscience connections** - Dual memory, synaptic consolidation
4. **Implement small-scale experiments** - Test ideas on MNIST/CIFAR-100
5. **Identify specific gap** - What aspect will your research uniquely address?

---

*Last Updated: February 2026*
*Focus: Breaking the one-time training paradigm through continuous/nested learning approaches*
