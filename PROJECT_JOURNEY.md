# Project Journey: From Problem to Paper

This document tells the story behind the project: the problem we started with, why we chose Retrieval-Augmented Generation (RAG), how we built and tested the system, what we found, and how those findings became the paper:

> **Knowledge Retrieval-Based Intelligent Question and Answer Generation Framework for Education**
> Anay Pardasani, Karen Maleski, Sayantan Roy. *Journal of Student Research*, Vol. 14, Issue 1 (2025).

---

## 1. The problem

Students prepare for exams by practising questions, but good practice material is scarce. Quality questions with correct answers have traditionally been **written by hand by educators**. That work is slow and needs a strong command of both the subject and academic standards, so the variety and availability of practice material for students stay limited.

Large Language Models (LLMs) looked like an obvious fix. They can read large amounts of text and write fluent, human-like questions and answers on demand. When we tried them directly, though, we saw the problem the paper is built around: on their own, LLMs **are not consistently accurate**, and their output **doesn't reliably match a specific syllabus or academic standard**. A question that sounds plausible but has a wrong answer is worse than no question at all for a student.

So we narrowed the problem down to three challenges:

1. **Coverage:** questions and answers should span the whole subject matter, not just what the model happens to know well.
2. **Quality:** questions *and* answers should be correct and match educational standards at different difficulty levels.
3. **Evaluation:** we needed objective metrics to show whether the system actually works, instead of judging by eye.

---

## 2. Why RAG?

There were broadly three ways to make an LLM produce trustworthy, syllabus-specific content:

| Option | Why we did / didn't choose it |
|---|---|
| **Prompting a general LLM alone** | Fast, but answers come from the model's memory. It can't be tied to a particular textbook, and it hallucinates. |
| **Fine-tuning a model on the curriculum** | Expensive, needs lots of labelled data, and has to be retrained whenever the material changes. |
| **Retrieval-Augmented Generation (RAG)** | Keeps the LLM's language ability but **grounds every answer in a curated knowledge base**. Updating the content just means adding PDFs, and every answer can be traced back to its source passage. |

RAG fit the problem best. The curated textbooks provide the **accuracy and alignment**, and the LLM provides the **language and reasoning**. Because each response comes with the retrieved context that produced it, RAG also makes **evaluation possible**: we can check whether an answer is actually supported by the source material.

---

## 3. Building the system

### 3.1 Getting the data out of PDFs
Educational material lives in PDFs: textbooks, notes and exam papers. We ran **OCR** to make them machine-readable and used **LlamaParse** to extract the harder elements such as figures and tables. Basic NLP cleaning followed, and the cleaned text was saved to separate files, with **multilingual content stored separately**, because the system was meant to be **bilingual**.

### 3.2 Choosing a chunking strategy
Documents are too long to retrieve whole, so they have to be split into chunks. We explored several approaches:

- **Fixed-size chunking:** simple and predictable, but can cut ideas in half.
- **Context-aware chunking:** respects document structure.
- **Recursive chunking:** splits hierarchically by separators.
- **Semantic chunking:** splits where the meaning changes.

Each trades **preserving context** against **retrieval precision**. We settled on LlamaIndex's **sentence splitter with fixed-size, overlapping chunks**. The overlap keeps context continuous across chunk boundaries, which improved retrieval, consistent with other RAG implementations.

### 3.3 Representing meaning as vectors
To search by meaning rather than keywords, every chunk was encoded with **multilingual BERT (mBERT)**. Its self-attention captures semantics across languages, so English and non-English content share one vector space. That was essential for a bilingual system.

### 3.4 Storing knowledge as "nodes"
Each chunk was stored in **MongoDB** as a *node*: the vector, the original text, and **metadata** such as subject, chapter or category:

```json
{ "_id": 1, "sentence": "Organisms that convert decaying matter into nutrients", "vector": [0.23, 0.89, -0.10, 0.54, ...] }
```

The metadata means results can later be filtered or ranked by subject or chapter.

### 3.5 Retrieval and generation
At query time the question is embedded with the same model, and **cosine similarity** finds the closest node. We used **top-k = 1**: only the single best-matching chunk is passed to the LLM. That chunk is inserted into a strict prompt that tells the model to use only the context, to say "I don't know" when the answer isn't there, and to keep answers to three sentences.

### 3.6 Comparing LLMs
We plugged the same pipeline into three LLMs: **GPT-4**, **GPT-4o mini** and **Gemini 1.5**. Comparing them showed whether the framework holds up across models, and whether a larger model is actually necessary.

---

## 4. Measuring it

To avoid judging the outputs subjectively, we built an **evaluation set of 100 questions with ground-truth answers**, drawn from a **separate set of PDFs** so the test wasn't trivially easy.

We scored both halves of the pipeline with the **RAGAS** framework:

| Stage | Metric | Question it answers |
|---|---|---|
| Retrieval | **Context precision** | Are the retrieved chunks relevant? |
| Retrieval | **Context recall** | Does the retrieved context contain what the ground-truth answer needs? |
| Generation | **Faithfulness** | Is every claim in the answer supported by the context? |
| Generation | **Answer relevancy** | Does the answer actually address the question asked? |

The paper uses an example to make faithfulness concrete. Take the question *"How can acceleration be determined using a velocity-time graph?"* and the answer *"the slope gives acceleration and the area gives distance."* It contains one supported claim and one wrong one (the area gives **displacement**), so faithfulness = 1/2 = **0.5**.

---

## 5. Results

| Metric | GPT-4 | GPT-4o mini | Gemini 1.5 |
|---|---|---|---|
| Faithfulness | 0.6516 | **0.7044** | 0.4583 |
| Answer relevancy | 0.9779 | **0.9838** | 0.1201 |
| Context recall | 0.6154 | 0.6575 | **0.6875** |
| Context precision | **0.8756** | 0.7676 | 0.5000 |

What stood out:

- **The OpenAI models clearly beat Gemini 1.5** on faithfulness, answer relevancy and context precision. Gemini's answer relevancy (0.12) was especially low.
- **GPT-4o mini, the smaller model, matched or beat GPT-4** on faithfulness and answer relevancy. This was the most interesting result: once the knowledge comes from retrieval, the model mostly needs to read the context and summarise it faithfully, and a smaller model does that well.
- **Retrieval is the bottleneck.** Context recall stayed at about 0.6–0.7 for every model. With top-k = 1, a single chunk often doesn't contain everything the ground truth needs, and that limits how good the final answer can be no matter which LLM is used.
- **Answer relevancy was very high** for the OpenAI models (about 0.98). The strict prompt kept answers focused on the question.

---

## 6. From findings to paper

We built the paper around these findings:

1. **Introduction:** framed the problem: manual question writing doesn't scale, and plain LLMs aren't reliable enough for education.
2. **Methods:** described the pipeline we built and the design choices we made along the way: OCR and parsing, the chunking comparison, mBERT embeddings, MongoDB nodes with metadata, cosine top-k retrieval, the prompt template, and the multi-LLM setup.
3. **Results:** defined each RAGAS metric (with the faithfulness worked example) and reported the comparison table.
4. **Discussion:** explained why the OpenAI models outperformed Gemini 1.5 for this context-heavy task.
5. **Conclusion:** highlighted the main takeaway: **small language models can be just as effective for task-specific, retrieval-grounded work**, which points toward cheaper and more accessible educational AI.
6. **Limitations:** recorded what we observed: remaining hallucinations, dependence on retrieval quality, compute cost, contextual drift over longer interactions, and the black-box nature of LLMs.

---

## 7. What's next

The findings suggest several directions:

- **Better retrieval:** increase top-k, try hybrid (keyword + vector) search, or add re-ranking to raise context recall.
- **Small language models:** build on the GPT-4o mini result and test local SLMs for low-cost or offline classroom use.
- **Richer question generation:** control difficulty and Bloom's-taxonomy level, and have educators review generated questions at scale.
- **Hallucination checks:** automatically reject generated Q&A pairs whose faithfulness score falls below a threshold.

---

## About this repository

The code here is a **simplified, runnable re-implementation** of the framework described above, kept small so it's easy to read and extend. It differs from the original setup in a few ways: `pypdf` instead of OCR + LlamaParse, a multilingual sentence-transformer instead of raw mBERT, and OpenAI models only. See the [README](README.md#how-it-maps-to-the-paper) for the full mapping.
