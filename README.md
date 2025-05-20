# Study Planner Assistant

A sophisticated LLM-powered study planner assistant built with LangChain that helps manage study materials, create personalized study schedules, and generate quizzes through natural language processing.

## 🌟 Features

- **Study Material Management**: Upload and organize your PDFs and lecture notes
- **Smart Scheduling**: Create personalized study schedules on your calendar
- **Material Search**: Search through your study materials using natural language
- **Quiz Generation**: Generate custom quizzes based on your study materials
- **Study Plans**: Create structured study plans for specific topics
- **Content Summarization**: Summarize your study materials for quick review
- **RAG (Retrieval Augmented Generation)**: Context-aware responses using vector similarity search

## 🛠️ Technologies

- **LangChain**: Framework for building LLM applications
- **Google Gemini 2.0**: Core language model powering the assistant
- **FAISS**: Vector database for efficient similarity search
- **HuggingFace Embeddings**: Sentence transformers for document embedding
- **PyPDF Loader**: For processing PDF documents
- **Agents & Tools**: Task-specific functionality with ReAct prompting

## 📋 System Architecture

The assistant implements a comprehensive study planning system with:

1. **Document Processing Pipeline**: Handles PDF and text documents
2. **Vector Store**: Maintains collections for study materials with semantic search
3. **Calendar Integration**: Manages study sessions and schedules
4. **Quiz System**: Generates and evaluates quizzes based on your materials
5. **Study Plan Generator**: Creates customized study plans
6. **Intent Classification**: Routes user queries to appropriate handling logic
7. **RAG Pipeline**: Retrieves relevant context to enhance responses

## 🚀 Getting Started

### Prerequisites

```bash
pip install langchain langchain_openai langchain_community langchain_google_genai faiss-cpu sentence-transformers python-dateutil PyPDF2
```

### Directory Structure

```
study-planner-assistant/
├── data/
│   ├── study_materials/      # Place your PDFs and notes here
│   ├── calendar_events.txt
│   └── quiz_history.txt
├── study_planner_assistant.py
└── README.md
```

### Configuration

1. Replace `YOUR_GOOGLE_API_KEY` in the code with your actual Google Gemini API key
2. Place your study materials (PDFs, text files) in the `data/study_materials` directory

## 💻 Usage Examples

### Uploading Study Materials

Place your PDFs and text files in the `data/study_materials` directory and the system will automatically process them on startup.

### Scheduling Study Sessions

```
> Schedule a study session on neural networks tomorrow at 3pm for 2 hours
Processing your request...

Scheduled: Neural Networks for 2025-05-21 at 15:00 for 2 hours
```

### Checking Schedule

```
> What am I studying next week?
Processing your request...

Found the following study sessions:
Topic: Neural Networks - Date: 2025-05-21 - Time: 15:00 - Duration: 2 hours
Topic: Data Structures - Date: 2025-05-23 - Time: 10:00 - Duration: 1 hour
```

### Searching Study Materials

```
> Find information about backpropagation
Processing your request...

### Result 1 from Neural Networks
Source: neural_networks.pdf

Backpropagation is an algorithm used for training neural networks by updating weights...
```

### Generating Quizzes

```
> Generate a quiz on data structures
Processing your request...

# Quiz on Data Structures

### Question 1: Which data structure uses LIFO ordering?
A. Queue
B. Stack
C. Linked List
D. Array

Correct answer: B

Explanation: A Stack is a Last-In-First-Out (LIFO) data structure...
```

### Creating Study Plans

```
> Create a study plan for machine learning
Processing your request...

# 7-Day Study Plan for Machine Learning

## Day 1: Fundamentals
**Objectives:** Understand basic ML concepts
**Content to Cover:** Introduction, types of learning
**Activities:** Review key definitions, simple examples
**Time Allocation:** 2 hours

...
```

## 🧠 Core Components

### DocumentProcessor

Handles the processing of PDFs and text files, splitting them into appropriate chunks and extracting metadata.

### StudyMaterialStore

A vector store system maintaining indices for study materials, providing efficient semantic search capabilities and topic organization.

### CalendarTool

Manages study session scheduling, calendar queries, and suggests optimal study times based on your existing schedule.

### QuizGenerator

Creates custom quizzes based on your study materials and can evaluate your answers to test your knowledge.

### SummaryTool

Generates concise summaries of your study materials and creates structured study plans for specific topics.

### Intent Classification

Uses a dedicated LLM chain to classify incoming queries into specific categories:
- upload_material
- schedule_study
- check_schedule
- search_materials
- generate_quiz
- answer_quiz
- summarize_content
- create_study_plan
- get_topics
- suggest_study_times
- general_inquiry

### Agent System

Implements a Zero-Shot ReAct agent for reasoning about tool selection and use, with specialized prompting to guide the assistant's behavior.

## 🔍 Technical Details

- **Vector Embeddings**: Utilizes all-MiniLM-L6-v2 model for efficient semantic representations
- **Persistent Storage**: Maintains study data in simple text files for portability
- **Error Handling**: Robust exception management with fallback to RAG-based responses
- **Conversation Flow**: Maintains context through well-formatted interaction patterns

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- LangChain framework developers
- Google Gemini API team
- Sentence Transformers project