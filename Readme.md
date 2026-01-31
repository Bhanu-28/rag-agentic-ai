# IBM RAG AND GENERATIVE AI

Description: https://www.coursera.org/professional-certificates/ibm-rag-and-agentic-ai

# Foundations of Generative AI and prompt Engineering

 This hands-on course is designed for aspiring AI developers, machine learning engineers, data scientists, AI researchers, or someone in a related role.

### **Pre-requisites**

Python programming skills and experience are essential for this course, as it dives straight into coding and application development. Fundamental AI and web development knowledge are highly recommended. 

### Next Evolution of AI

Retrieval Augmented Generation, or RAG, enhances AI's ability to provide accurate, context-aware responses by integrating real-time information retrieval. 

Multimodal AI is another advancement that allows systems to process and integrate various types of data – text, images, audio, and video – enabling more dynamic and interactive user experiences. 

Agentic AI represents a further shift, equipping systems with the ability to reason, plan, and autonomously execute tasks. 

While each of these approaches can work independently, they can also be combined to create more powerful and adaptable AI systems. 

welcome to the **Develop Generative AI Applications:** 

This course is part of the

[**IBM RAG and Agentic AI** **Professional Certificate**](https://www.coursera.org/professional-certificates/ibm-rag-and-agentic-ai)

, designed to provide you with the practical skills and knowledge to excel in developing advanced AI applications that leverage retrieval-augmented generation (RAG), multimodal AI, and agentic AI systems.

## **Objectives**

After completing this course, you will be able to:

- Examine foundational concepts of generative AI and the LangChain framework, focusing on how prompt engineering and in-context learning enhance AI interactions
- Apply prompt templates, chains, and agents to create flexible and context-aware AI applications using LangChain's modular approach
- Develop a generative AI web application with Flask, integrating advanced features such as JSON output parsing for structured AI responses
- Evaluate and compare different language models to select the most suitable option for specific use cases, ensuring optimal performance and reliability

This course consists of three modules.

## **Module 1: Foundations of Generative AI and Prompt Engineering**

- Lesson 0: Welcome
- Lesson 1 [Optional]: Introduction to Generative AI
- Lesson 2: Working with Prompt Engineering and Prompt Templates
- Lesson 3: Module Summary and Evaluation

## **Key Topics:**

- Generative AI Models
- NLP (Natural Language Processing)
- Prompt Engineering and In-Context Learning
- Advanced Methods of Prompt Engineering
- LangChain LCEL Chaining Method
- LangChain Prompt Templates

## **Module 2: Introduction to LangChain in GenAI Applications**

- Lesson 1: LangChain Core Components and Advanced Features
- Lesson 2: Module Summary and Evaluation

## **Key Topics:**

- LangChain: Core Concepts
- LangChain Advanced Features: Chains and Agents
- Build Smarter AI Apps: Empower LLMs with LangChain

## **Module 3: Build a Generative AI Application with LangChain**

- Lesson 1: Application Development Workflow with Generative AI
- Lesson 2: Module Summary and Evaluation
- Lesson 3: Course Wrap-Up

## **Key Topics:**

- Choose the Right AI Model for Your Use Case
- From Idea to AI: Building Applications with Generative AI
- Python with Flask for Large-Scale Projects
- Hands-on with GenAI: Choosing the Right Model for Your Application

### **Tools/Software**

In this course, you will explore and utilize a variety of tools and platforms, including:

- LangChain to design and implement structured AI workflows, integrate LLMs with external data sources, and optimize prompt engineering techniques
- Flask, HTML5 and CSS3 to develop AI-powered web applications with interactive features
- LangChain's JsonOutputParser to generate structured JSON outputs from LLM responses for enhanced data usability
- LLMs (Llama 3, Granite, Mixtral) to experiment with different AI models, compare their performance, and select the best fit for specific applications
- Python for coding AI applications, integrating APIs, and implementing LangChain's functionalities effectively

The creative skills of generative AI come from generative AI models such as generative adversarial networks or GANs, variational autoencoders or VAEs, transformers, and diffusion models. 

These models can be considered as the building blocks of generative AI. 

For instance, you have ChatGPT and Gemini for text generation, DALL-E2 and MidJourney for image generation, Synthesia for video generation, and Copilot and AlphaCode for 

code generation. 

And even though these models are trained to perform at its core a generation task, predicting the next word in the sentence, we actually can take these models and if you introduce a small amount of label data to the equation, you can tune them to perform traditional NLP tasks, things like classification or named entity recognition, things that you don't normally associate as being a generative-based model or capability. This process is called tuning, where you can tune your foundation model by introducing a small amount of data, you update the parameters of your model, and now I can perform a very specific natural language task. If you don't have data or have only very few data points, you can still take these foundation models and they actually work very well in low label data domains. 

**Discriminative AI —> Classification of data**

**Generative AI → Discriminative models model the boundary between classes, while generative models model how data is generated through data distributions.** 

### Natural language processing.

Foundation models can be tuned with minimal labeled data to perform various NLP tasks.

Translating from unstructured to structured data.

Natural language understanding —> unstructured to structured

Natural language generation —> Structured to unstructured.

1. Machine translation—> understand overall context.
2. Virtual Assistant/CHAT BOT
3. Sentiment analysis. Positive/Serious/Sarcastic.  
4. SPAM DETECTION.

It has multiple algorithms 

 The first stage of NLP is called **tokenization**. This is about taking a string and breaking it down into chunks. So if we consider the unstructured text we've got here, "add eggs and milk to my shopping list,” that's eight words, and that could be eight tokens. And from here on in, we are going to work one token at a time as we traverse through this. Now, the first stage, once we've got things down into tokens that we can perform, is called **stemming**. And this is all about deriving the word stem for a given token. So for example, running, runs, and ran—the word stem for all three of those is run. We're just kind of removing the prefix and the suffixes and normalizing the tense, and we're getting to the word stem. But stemming doesn't work well for every token. For example, "universal" and "university"—well, they don't really stem down to universe. For situations like that, there is another tool that we have available, and that is called **lemmatization**.

And **lemmatization** takes a given token and learns its meaning through a dictionary definition, and from there it can derive its root or its lem. So take better, for example. Better is derived from good, so the root or the lem of better is good. The stem of better would be bet. So you can see that it is significant whether we use stemming or we use lemmatization for a given token.Now, next thing we can do is we can do a process called part of **speech tagging**.  And what this is doing is, for a given token, it's looking where that token is used within the context of a sentence. So take the word "make,” for example. If I say I'm going to make dinner, "make” is a verb. But if I ask you what the make of your laptop, well, make is now a noun. So where that token is used in the sentence matters. Part of speech tagging can help us derive that context. And then finally, another stage is named **entity recognition**. And what this is asking is, for a given token, is there an entity associated with it? So, for example, a token of Arizona has an entity of a U.S. state, whereas a token of Ralph has an entity of a person's name. 

# **Foundational Concepts**

| **Term** | **Definition** | **Examples/Use Cases** |
| --- | --- | --- |
| **LLM** | A type of AI model trained on vast amounts of text data to understand and generate human-like language. | GPT-o1, Claude, LLaMA |
| **Prompting** | A technique for designing input instructions to guide LLM outputs. | "Write a summary in 3 sentences," "Answer as a cybersecurity expert." |
| **Prompt Templates** | Reusable, structured prompts with placeholders for dynamic inputs. | `"Explain {concept} like I'm 5 years old."` |
| **RAG (Retrieval-Augmented Generation)** | Combines retrieval from external knowledge sources with LLM generation to enhance factual accuracy. | Answering questions with real-time data (for example, [RAG Paper](https://arxiv.org/abs/2005.11401)) |
| **Retriever** | A system component designed to fetch relevant information from a dataset or database. | Vector similarity search using FAISS, Elasticsearch |
| **Agent** | An autonomous AI system that can plan, reason, and execute tasks using tools. | AutoGPT, LangChain Agents |
| **Multi-Agent System** | A framework in which multiple AI agents collaborate to solve complex tasks. | Microsoft AutoGen, CrewAI |
| **Chain-of-Thought** | A prompting technique that encourages models to decompose problems into intermediate steps. | "Let's think step by step…" |
| **Hallucination Mitigation** | Strategies to reduce incorrect or fabricated outputs from LLMs. | RAG, fine-tuning, prompt constraints |
| **Vector Database** | A database optimized for storing and querying vector embeddings. | Pinecone, Chroma, Weaviate |
| **Orchestration** | Tools to manage and coordinate workflows involving multiple AI components. | LangChain, LlamaIndex |
| **Fine-tuning** | Adapting pre-trained models for specific tasks using domain-specific data. | LoRA (Low-Rank Adaptation), QLoRA (quantized fine-tuning) |

---

# **Tools & Frameworks**

# **Model Development & Deployment**

| **Tool/Framework** | **Definition** | **Examples/Use Cases** | **Reference Link** |
| --- | --- | --- | --- |
| **Hugging Face** | A platform hosting pre-trained models and datasets for NLP tasks. | Accessing GPT-2, BERT, Stable Diffusion | [Hugging Face](https://huggingface.co/) |
| **LangChain** | A framework for building applications with LLMs, agents, and tools. | Creating chatbots with memory and web search | [LangChain](https://python.langchain.com/) |
| **AutoGen** | A library for creating multi-agent conversational systems. | Simulating debates between AI agents | [AutoGen](https://microsoft.github.io/autogen/) |
| **CrewAI** | A framework for assembling collaborative AI agents with role-based tasks. | Task automation with specialized agents | [CrewAI](https://www.crewai.com/) |
| **BeeAI** | A lightweight framework to build production-ready multi-agent systems | Distributed problem-solving systems | [BeeAI](https://github.com/i-am-bee/beeai-framework/tree/main) |
| **LlamaIndex** | A tool to connect LLMs to structured or unstructured data sources. | Building Q&A systems over private documents | [LlamaIndex](https://www.llamaindex.ai/) |
| **LangGraph** | A library for building stateful, multi-actor applications with LLMs. | Cyclic workflows, agent simulations | [LangGraph](https://www.langchain.com/langgraph) |

# **Retrieval & Infrastructure**

| **Tool/Framework** | **Definition** | **Examples/Use Cases** | **Reference Link** |
| --- | --- | --- | --- |
| **FAISS** | A library for efficient similarity search of dense vectors. | Retrieving top-k documents for RAG | [FAISS](https://faiss.ai/) |
| **Pinecone** | A managed cloud service for vector database operations. | Storing embeddings for real-time retrieval | [Pinecone](https://www.pinecone.io/) |
| **Haystack** | An end-to-end framework for building RAG pipelines. | Deploying enterprise search systems | [Haystack](https://haystack.deepset.ai/) |

# **Advanced Prompting Techniques**

| **Concept** | **Definition** | **Example** |
| --- | --- | --- |
| **Few-Shot Prompting** | Providing examples in the prompt to guide the model's output format. | "Translate to French: 'Hello' → 'Bonjour'; 'Goodbye' → __” |
| **Zero-Shot Prompting** | Directly asking the model to perform a task without examples. | "Classify this tweet as positive, neutral, or negative: {tweet}” |
| **Chain-of-Thought** | Encouraging step-by-step reasoning. | "First, calculate X. Then, compare it to Y. Final answer: ___” |
| **Prompt Chaining** | Breaking complex tasks into smaller prompts executed sequentially. | Prompt 1: Extract keywords → Prompt 2: Generate summary from keywords. |

# **Key Architectures & Workflows**

# **RAG Pipeline**

1. **Retrieval**: Query vector database (for example, Pinecone) for context.
2. **Augmentation**: Combine context with user prompt.
3. **Generation**: LLM (for example, GPT-4) produces final output.

# **Multi-Agent System**

- **Agents**: Specialized roles (for example, researcher, writer, critic).
- **Orchestration**: LangGraph for cyclic workflows, AutoGen for conversations, and so on.
- **Tools**: Web search, code execution, API integrations, and so on.

### In-context Learning.

In-context learning is a specific method of prompt engineering where demonstrations of the task are provided to the model as a part of the prompt in natural language. 

 A new task is learned from a small set of examples presented within the context or prompt at inference time. 

In-context learning doesn't require the model to be fine tuned on specific datasets. This can drastically reduce the resources and time needed to adapt LLMs for specific tasks while improving their performance  it's constrained by what can realistically be provided in the context. Complex tasks could require gradient steps or more traditional machine learning training approaches, which involve adjusting the model's weights based on error gradients

Essentially, prompts are instructions or inputs given to an LLM designed to guide it toward performing a specific task or generating a desired output.  There are two main components to a prompt. **Instructions** are clear, direct commands that tell the AI what to do. They need to be specific to ensure the LLM understands the task. **Context** includes the necessary information or background that helps the LLM make sense of the instruction. It can be data, parameters, or any relevant details that shape the AI's response. By combining these elements effectively, you can tailor LLMs like those developed by IBM, OpenAI, Google, or Meta to perform tasks ranging from answering queries and analyzing data to generating content. Prompt engineering is a specialized process where you design and refine the questions, commands, or statements you use to interact with the AI systems, particularly LLMs. 

The goal is not just about asking a question, it's about how to ask it in its best way possible. This involves carefully crafting clear, contextually rich prompts tailored to get the most relevant and accurate responses from the AI. This process is fundamental in fields ranging from customer service automation to advanced research and computational linguistics. 

### **Why Prompt engineering**

Prompt engineering boosts effectiveness and accuracy by directly influencing how effectively and accurately LLMs function. It ensures relevance by enabling LLMs to generate precise and perfectly suited responses to the context. It facilitates meeting user expectations through clearer prompts and reduced misunderstandings.
It eliminates the need for continual fine-tuning, allowing the model to adapt and learn within its context.

Let's break down the components that make up a well structured prompt. Instructions tell the LLM what needs to be done.
For example, classify the following customer review into neutral, negative, or positive sentiment. This is straightforward and directs the LLMs action. Context helps the LLM understand the scenario or the background in which it operates. Here it's indicated that this review is part of feedback for a recently launched product. This can help the LLM weigh the sentiment analysis in light of the products novelty. Input data is the actual data the LLM will process. In the prompt, it's the customer review.
The product arrived late but the quality exceeded my expectations. The LLM uses this data to perform the task specified by the instructions. The output indicator is the part of the prompt where the LLM's response is expected. It's a clear marker that tells the AI where to deliver its analysis. In this example, sentiment indicates waiting for the LLM to append its classification.

### **Recap**.

In this video, you learned that in-context learning is a method of prompt engineering where task demonstrations are provided to the model as a part of the prompt. 

Prompts are inputs given to an LLM to guide it towards performing a specific task.

 They consist of instructions and context. 

Prompt engineering is a process where you design and refine the prompts to get relevant and accurate responses from AI. Prompt engineering has several advantages. It boosts the effectiveness and accuracy of LLMs. It ensures relevant responses. It facilitates meeting user expectations. It eliminates the need for continual fine tuning. A prompt consists of four key elements, instructions, context, input data, and output indicator.

## **Introduction to LangChain**

The LangChain open-source Python framework streamlines the development of large language model (LLM) applications. LangChain provides developers with components and interfaces to assist in integrating LLMs into their AI applications.

LangChain is a Python framework for pinpointing relevant information and text and providing methods for responding to complex prompts. Benefits include **modularity, extensibility, decomposition capabilities**, and easy integration with **vector databases**. Several practical applications include **deciphering complex legal documents**, extracting key statistics from reports, customer support, and automating routine writing tasks. **LangChain** can be used with other data types by using external libraries and models.

It chains together the retrieval, extraction, processing, and generation operations from the large amounts of text and multiple sources, hence the word chain as part of its name. 

**Modularity** 

LangChain's design allows application developers to piece together different components like building blocks. This modularity also encourages component reuse, reducing development time and effort.

**Extensibility**

LangChain's extensible design allows developers to readily add new features, adapt to existing components, integrate with external systems, and make only minimal changes to their codebases

**Decomposition capabilities**

 LangChain mimics the human problem-solving process by breaking down complex queries or tasks into smaller, manageable steps. 

This decomposition capability enables it to make accurate inferences from context, resulting in relevant, precise responses. 

**Easy integration with vector databases**

 LangChain integrates with vector databases for efficient semantic searches and information retrieval.

 When used in conjunction with vector databases, it provides applications quick access to relevant information within extensive data sets.

### **Advanced Methods of Prompt engineering.**

**zero-shot prompt**.

 This type of prompt instructs an LLM to perform a task without any prior specific training or examples.  This task requires the LLM to understand the context and information without any previous tuning for the specific query

**one-shot prompt** gives the LLM a single example to help it perform a similar task.

**few-shot prompting** where the AI learns from a small set of examples before tackling a similar task. This helps the AI generalize from a few instances to new data. For example, the LLM is shown three statements, each labeled with an emotion. These examples teach the LLM to classify emotions based on context. Then it classifies a new statement. 

 **Chain-of-thought or COT prompting** is a technique used to guide LLM through complex reasoning step-by-step. This method is highly effective for problems requiring multiple intermediate steps or reasoning that mimics human thought processes.

You can view the model output for the CoT prompt query by breaking down the calculation into clear sequential steps. The model arrives at the correct answer and provides a transparent explanation. 

**Self-consistency** is a technique for enhancing the reliability and accuracy of outputs. It involves generating multiple independent answers to the same question and then evaluating these to determine the most consistent result. 

 This approach demonstrates how self-consistency can verify the reliability of the responses from LLMs by cross-verifying multiple paths to the same answer.

### Tools and applications used in prompt engineering

OpenAI's Playground, LangChain, Hugging Face's Model Hub, and IBM's AI Classroom

 They enable real-time tweaking and testing of prompt to see immediate effect on outputs. 

they provide access to various pre-trained models suitable for different tasks and languages. They also facilitate the sharing and collaborative editing of prompts among teams or communities. Finally, they offer tools to track changes, analyze results, and optimize prompt based on performance metrics.

Let's learn more about LangChain, a tool for prompt engineering. LangChain uses prompt templates, predefined recipes for generating effective prompt for LLMs. These templates include instructions for the language model, a few-shot examples to help the model understand contexts and expected responses, and a specific question directed at the language model.

 An agent can perform complex tasks across domains using different prompts. 

### **LangChain Expression language LCEL Chaining Method**

 LangChain Expression Language (or LCEL) is a pattern for building LangChain applications that utilizes the pipe (|) operator to connect components. This approach ensures a clean, readable flow of data from input to output. 

This modern method provides better composability, clearer visualization of data flow, and greater flexibility when constructing complex chains.  . To create a typical LCEL pattern, you need to Define a template with variables and curly braces Create a prompt template instance. Build a chain using the pipe operator to connect components. Invoke the chain with input values.

 In LangChain, runnables serve as an interface and building blocks that connect different components like LLMs, retrievers, and tools into a pipeline. 

There are two main runnable composition primitives. Runnable sequence chains components sequentially, passing the output from one component as input to the next. RunnableParallel runs multiple components concurrently while using the same input for each. However, LCEL provides elegant syntax shortcuts. For example, instead of using runnable sequence, the same sequential chain can be created by simply connecting runnable 1 and runnable 2 with a pipe, making the structure more readable and intuitive. LCEL also handles type coercion automatically. This means it converts regular code into runnable components. 

When you use a dictionary, it becomes a runnable parallel, which runs multiple tasks simultaneously. When you use a function, it becomes a RunnableLambda, which transforms inputs.

RunnableLambda in this chain wraps the format_prompt function, transforming it into a runnable component that LangChain can work with. When the chain runs, RunnableLambda takes the input dictionary, containing adjective and content keys, passes this dictionary to the format_prompt function. The function formats the prompt template with these variables. The formatted prompt is then passed to the next component, the LLM. 

The pipe operator creates a sequence by connecting runnable components together.

LCEL is best suited for simpler orchestration tasks. For more complex workflows, consider using LangGraph while still leveraging LCEL within individual nodes. 

LCEL pattern structures workflows use the pipe operator for clear data flow. Prompts are defined using templates with variables and curly braces. Components can be linked using RunnableSequence for sequential execution. RunnableParallel allows multiple components to run concurrently with the same input. LCEL provides a more concise syntax by replacing RunnableSequence with the pipe operator.
Type coercion in LCEL automatically converts functions and dictionaries into compatible components.

Example : The result will contain three keys, summary, translation, and sentiment, each with the output from the respective LLM call. 

## **What Is Prompt Engineering and Why Do We Care?**

If you've ever used a chatbot or virtual assistant online, you may have mixed feelings about the experience. Most chatbots are designed to respond to a predefined set of queries programmatically. While AI allows some flexibility in how we phrase our messages, these chatbots are essentially programmed with specific responses.

It all starts with user input, our **prompt**. A simple text box is all we have to control this powerful tool. The quality of our prompt significantly impacts the results (the output). As the saying goes, 'garbage in, garbage out.' If we provide a vague or poorly structured prompt, the AI won't have much to work with, limiting its ability to deliver valuable responses and showcase its full potential.

### **What is prompt engineering?’**

> Prompt engineering refers to the process of designing and refining prompts or instructions given to a language model, such as GPT-3.5, to generate desired outputs. It involves carefully crafting the input text provided to the model to elicit the desired response or behavior.
> 

> Prompt engineering can be used to fine-tune the output of language models by specifying explicit instructions, adding context, or using specific formatting techniques to guide the model's generation. By tweaking the prompts, developers and users can influence the output to be more accurate, coherent, or aligned with a specific goal.
> 

> Effective prompt engineering requires understanding the underlying language model and its capabilities. It involves experimenting with different prompts, iterating on them, and analyzing the model's responses to achieve the desired results. Prompt engineering is particularly important Prompt engineering is crucial when working with large language models like GPT-3.5, as their responses can occasionally be unpredictable or need extra context to produce the desired results.
> 

For example, when we asked, 'What is Prompt Engineering?', we used a zero-shot prompt.

In contrast, a **one-shot** prompt provides the model with a single example, while a **few-shot** prompt includes multiple examples to guide its output. These techniques help shape the model's responses, especially when it hasn't been explicitly trained for a particular task. Users can steer the model toward a specific output or format by offering examples.

**Chain-of-thought (CoT)**, a more advanced prompting technique explored later in the course, is an example of **one-shot** prompting (or **few-shot** prompting, depending on how we phrase our prompt).

[Module1 - Lab1](https://www.notion.so/Module1-Lab1-2e5fdd05003e8049b73beeb48c0e22f1?pvs=21)

# Summary and Highlights: Foundations of Generative AI and Prompt Engineering

Congratulations! You have completed this module. At this point in the course, you know:

- In-context learning is a prompt engineering method where demonstrations of the task are provided to the model as part of the prompt.
- Prompts are inputs given to an LLM to guide it toward performing a specific task.
- Prompt engineering is a process where you design and refine the prompt questions, commands, or statements to get relevant and accurate responses.
- Advantages of prompt engineering include that it boosts the effectiveness and accuracy of LLMs, ensures relevant responses, facilitates user expectations, and eliminates the need for continual fine-tuning.
- A prompt consists of four key elements: the instructions, the context, the input data, and the output indicator.
- Advanced methods for prompt engineering include zero-shot prompts, few-shot prompts, chain-of-thought prompting, and self-consistency.
- Prompt engineering tools can facilitate interactions with LLMs.
- LangChain uses 'prompt templates,' which are predefined recipes for generating effective prompts for LLMs.
- An agent is a key component in prompt applications that can perform complex tasks across various domains using different prompts.
- LCEL pattern structures workflows use the pipe operator (|) for clear data flow.
- Prompts are defined using templates with variables in curly braces {}.
- Components can be linked using RunnableSequence for sequential execution.
- RunnableParallel allows multiple components to run concurrently with the same input.
- LCEL provides a more concise syntax by replacing RunnableSequence with the pipe operator.
- Type coercion in LCEL automatically converts functions and dictionaries into compatible components.

## **Introduction to LangChain**

In the rapidly evolving landscape of artificial intelligence (AI), the LangChain framework has emerged as a beacon for developers and researchers keen on harnessing the power of large language models (LLMs) for practical applications. LangChain is an open-source framework designed to streamline the creation and deployment of language AI applications, focusing on extracting specific data points from extensive texts and facilitating complex language-based operations.

### **Benefits:**

LangChain is useful for several reasons:

- **Modularity:** LangChain's modular design allows users to piece together different components like building blocks, fostering an environment of innovation and flexibility. This modularity not only simplifies the development process but also encourages the reuse of components, reducing the time and effort required to bring new ideas to fruition.
- **Chain of thought processing:** LangChain employs a "chain of thought" processing model that breaks down complex queries or tasks into smaller, manageable steps and enhances the model's understanding of context and its ability to make accurate inferences, resulting in more relevant and precise responses. With this ability, LangChain mimics human problem-solving processes, making the interactions with AI more natural and intuitive.
- **Integration with vector databases:** LangChain offers seamless integration with vector store databases that enables efficient semantic search and information retrieval, essential for applications requiring quick access to relevant data points within extensive datasets. The ability to query and retrieve information based on semantic similarity opens new possibilities for knowledge management and information discovery.

**LangChain extensibility**

A crucial aspect of LangChain's design is its extensibility, which allows it to add new features, adapt to existing components, or integrate with external systems to meet specific project requirements easily. With its flexible design, LangChain ensures that applications built on it can evolve alongside emerging technologies and changing business needs.

**Practical applications of LangChain**

LangChain finds its use in various aspects:

- **Content summarization:** LangChain can automatically summarize articles, reports, and documents, highlighting key information for quick consumption that helps users stay informed about developments in their field without dedicating hours to reading.
- **Data extraction:** The LangChain framework's ability to retrieve specific information from unstructured texts for data analysis and management. It can extract key financial figures from reports or identify relevant case law in legal documents, simplifying the process of turning text into actionable insights.
- **Question answering systems:** Building sophisticated QA systems with LangChain can transform customer support and information retrieval services. By understanding and responding to queries with contextually relevant answers, these systems can provide a higher level of service and efficiency.

• **Automated content generation:** LangChain's capabilities extend to content creation, enabling the automatic generation of written materials. The framework opens new possibilities for automating routine writing tasks, from drafting emails to generating creative writing or technical documentation.

LangChain represents a pivotal development in the field of AI, offering a comprehensive toolset to tackle the complexities of language-based applications.

### **LangChain: Core Concepts**

LangChain is an open-source interface that simplifies the application development process using LLMs. The core components of LangChain are The language models in LangChain use text input to generate text output. The chat model understands the question or prompts and responds like a human. The chat model handles various chat messages such as 

human message, 

AI message, 

system message,

function message, 

and tool message.
The prompt templates in LangChain translate the questions or messages into clear instructions. You've also learned about the example selector which instructs the model for the inserted context and guides LLM to generate the desired output. Lastly, you learned about the Output Parsers that transform the output from an LLM into a suitable format.

LangChain is a platform that embeds APIs for developing applications. Chains is the sequence of calls. In chains, the output from one step becomes the input for the next step. In LangChain, chains first define the template string for the prompt, then creates a prompt template using the defined template, and creates an LLM chain object name. 

In LangChain, memory storage is important for reading and writing historical data. Agents in LangChain are dynamic systems where a language model determines and sequences actions such as pre-defined chains. Agents integrate with tools such as search engines, databases, and websites to fulfill user requests. 

**Create LCEL Chain**

 To create a typical LCEL pattern, you need to Define a template with variables and curly braces Create a prompt template instance. Build a chain using the pipe operator to connect components. Invoke the chain with input values. 

In LangChain, runnables serve as an interface and building blocks that connect different components like LLMs, retrievers, and tools into a pipeline.

There are two main runnable composition primitives. Runnable sequence chains components sequentially, passing the output from one component as input to the next. RunnableParallel runs multiple components concurrently while using the same input for each. However, LCEL provides elegant syntax shortcuts. For example, instead of using runnable sequence, the same sequential chain can be created by simply connecting runnable 1 and runnable 2 with a pipe, making the structure more readable and intuitive. LCEL also handles type coercion automatically. This means it converts regular code into runnable components.  

When you use a **dictionary**, it becomes a **runnable parallel**, which runs multiple tasks simultaneously. When you use a **function**, it becomes a **RunnableLambda**, which transforms inputs. This happens behind the scenes, so you don't have to handle the conversion manually.

The RunnableLambda in this chain wraps the Format_prompt function, transforming it into a runnable component that LangChain can work with. When the chain runs, RunnableLambda takes the input dictionary, containing adjective and content keys, passes this dictionary to the Format_prompt function. The function formats the prompt template with these variables. The formatted prompt is then passed to the next component, the LLM. 

The pipe operator creates a sequence by connecting runnable components together. In this joke chain, first, the RunnableLambda formats the prompt with variables. The pipe operator passes the formatted prompt to the LLM. Another pipe passes the LLM's response to the StrOutputParser.

[Module 2 Lab 1](https://www.notion.so/Module-2-Lab-1-2f6fdd05003e804198def85e92aa795b?pvs=21)

## Module 3

### Choose the right model

LangChain is an open-source framework designed to develop applications that leverage large language models (LLMs). LangChain stands out by providing essential tools and abstractions that enhance the customization, accuracy, and relevance of the information generated by these models.

LangChain offers a generic interface compatible with nearly any LLM. This generic interface facilitates a centralized development environment so that data scientists can seamlessly integrate LLM-powered applications with external data sources and software workflows. This integration is crucial for organizations looking to harness AI's full potential in their processes.

One of LangChain's most powerful features is its module-based approach. This approach supports flexibility when performing experiments and the optimization of interactions with LLMs. Data scientists can dynamically compare prompts and switch between foundation models without significant code modifications. These capabilities save valuable development time and enhance the developer's ability to fine-tune applications.

If you want your AI garden to grow, you need to ensure that you're using a variety of vegetables, or in this case, multiple models. That's what we call a multi-model approach, where you have a variety of models for your AI use cases. This means you can pick and choose from different models to find the right one for the right use case, which gives you the opportunity to look at how each of those models is designed as you find the right fit. You need to ask specific, important questions. Who built it? What data was it trained on? What guardrails are in place for it? 

What risks and regulations do you need to consider and account for? The other challenge when it comes to finding the right model for the right use case is identifying the best use case to fit your business needs. And that begins with a prompt. A prompt is a textual input or instruction that goes into a large language model to set up the basics of the AI. What a good prompt does is clearly articulate your use case and the problem you're solving with AI. So, the first step in the process of choosing a model for your use case is writing a very specific prompt that captures use case, user problem, the ask of the technology, and the guardrails for what good looks like. Next, you'll research the available models, looking at things like model size, performance, costs, risks, and deployment methods. 

You can then use the information you've collected to evaluate those models against your prompt and identify which of them you first want to test. Start with a large model and work with it until you satisfy your original prompt. Then, try to duplicate the result using smaller models. You're essentially passing the same prompt through different models to experiment and see which works best.

Now, all throughout this process, you want to be constantly thinking about the factors that affect your choice of model. In addition to the three elements of performance, accuracy, reliability, and speed, you also want to consider size, deployment method, transparency, and any potential risks.

 All of those need to be considered as you choose the right model for your use case and then start to implement it. That implementation is going to require a team that not only crosses disciplines, but also crosses lines of business. 

Don't think of it as proprietary to any one department, but treat it as a distinctly collaborative project that takes multiple teams to get up and running. Ensure that this team is ready and able to diagnose performance benchmarks, each of which measures something unique and produces a dataset that shows how everything is being calculated. Without this, you can't make informed decisions about this in future models and use cases. And remember, even once your little AI model crop is growing happily, you need to keep taking care of it. In this case, that means continuous testing, governance, and optimization, all of which are essential to keep that model up to date and running optimally. Remember, models evolve, so your strategy and choices need to do the same. You want to keep growing towards the sun instead of withering on the vine. 

**Ideation** : 

first step is ideating around exploration and proof of concepts, which I can break down in a few simple steps. So firstly, remember that your use case is specialized, so you need a specialized model that can do the job as well. You'll start from researching and evaluating models from popular repositories like Hugging Face or the open source community. And that's a great start.
But you also need to think about different factors such as the model size, or say, for example, its performance, and understand the benchmarking through popular benchmark tools that are out there as well. For example, there's a couple different ground rules that you need to understand. Generally, self hosting a large language model will be cheaper than a cloud-based service. And small language models, SLMs, versus large language models, LLMs, will generally perform better with lower latency, and they're specialized for a specific task. Now, you should also understand about various prompting techniques when you're actually working with the model. For example, zero-shot prompting. Now what this is, is essentially asking a model a question without any examples of how to respond.
Now, we can also do this a little bit differently with what's known as few-shot prompting, where we're actually giving a few different examples of how to respond, the behavior you want the LLM to have as we're working with the AI, and also chain of thought, which is actually asking the model to explain its thinking, its process, step by step  Now that we've evaluated models for our use case, it's time to build our application. Now, just as we can locally run databases and different services on our machine, well, we can actually do the same with **AI** to actually **serve it locally** from our machine and be able to make requests to its API from local host. Plus, you also get the added benefit of knowing that your data is secure and **private** on **premise**. 

But what if you want to use that data with your large language model as well? Well, there's a few different methods to do so, starting with what's known as retrieval, augmented generation, or RAG, where you actually take a large language model, a pre-trained foundational model, and supplement it with relevant and accurate data. But what you can also do is fine tune the model. So take the large language model and include the data with it. So we're actually baking this information, how we want it to behave, the different styles and intuition that we want it to react with, actually into the model itself. 

**Building**

Now, just as we can locally run databases and different services on our machine, well, we can actually do the same with AI to actually serve it locally from our machine and be able to make requests to its API from local host. Plus, you also get the added benefit of knowing that your data is secure and private on premise.

 That's really important nowadays. But what if you want to use that data with your large language model as well? Well, there's a few different methods to do so, starting with what's known as retrieval, augmented generation, or RAG, where you actually take a large language model, a pre-trained foundational model, and supplement it with relevant and accurate data. And this can help provide better and more accurate responses.fine tune the model. So take the large language model and include the data with it. So we're actually baking this information, how we want it to behave, the different styles and intuition that we want it to react with, actually into the model itself. 

Now, this can be done through sequences of prompts and model calls to actually accomplish more complex tasks. So that means you're going to need to break down problems into smaller, more manageable steps. 

And during this process, be able to evaluate the flows during these model calls, now, but also in a production environment, which brings us to the final step of operationalizing these AI-powered applications. So finally, you've got that application powered by AI or a large language model, and you want to deploy it to production to be able to scale things up. And this actually falls under the umbrella of something known as machine learning operations, or MLOps.

**Deployment** 

So first off, your infrastructure needs to be able to handle efficient model deployment and scaling. So using technologies such as **containers** and **orchestrators**, such as **Kubernetes**, are going to help you do this. Being able to auto scale and balance the traffic for your application. 

MLOPS

And you can also use production-ready runtime, such as vLLM, for the model serving. And what we're also seeing right now is that organizations are taking a hybrid approach, both with their models and their infrastructure. So having this multi-model Swiss Army knife approach to different models for different use cases, as well as a combination of on-prem and cloud infrastructure to make the most out of your resources and your budget. So with all of these new AI-powered applications out there, say you have something in production, well, the job isn't done. You still need to benchmark, monitor, and be able to handle different exceptions that are coming from your application. And similar to how we have DevOps, well, we also have MLOps for ensuring models go into production in a smooth fashion

## **FLASK**

[**FLASK**](https://www.notion.so/FLASK-2f9fdd05003e805aa05ec82a8bd28f56?pvs=21)