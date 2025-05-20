import sys
import os
import datetime
import re
import time
import numpy as np
from typing import List, Dict, Any, Tuple
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain, SimpleSequentialChain
from langchain_openai import OpenAI
from langchain.agents import AgentType, initialize_agent, Tool
from langchain.agents import AgentExecutor
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.vectorstores import FAISS
from dateutil import parser
from dateutil.relativedelta import relativedelta
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd

# Initialize LLM and embeddings
model = ChatGoogleGenerativeAI(
    google_api_key="AIzaSyAFw2hwd10KH089Nlus29HmKQcLiOSrCbI",  # Replace with your API key
    model="gemini-2.0-flash",
    temperature=0.0,
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

class DocumentProcessor:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
    
    def process_pdf(self, file_path):
        """Process a PDF file and convert it to document chunks"""
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # Extract metadata from filename
            filename = os.path.basename(file_path)
            topic = filename.split('.')[0].replace('_', ' ').title()
            
            # Set metadata
            for doc in documents:
                doc.metadata['topic'] = topic
                doc.metadata['source'] = file_path
                doc.metadata['type'] = 'pdf'
            
            # Split documents
            split_docs = self.text_splitter.split_documents(documents)
            
            return split_docs
        except Exception as e:
            print(f"Error processing PDF {file_path}: {e}")
            return []
    
    def process_text(self, file_path):
        """Process a text file and convert it to document chunks"""
        try:
            loader = TextLoader(file_path)
            documents = loader.load()
            
            # Extract metadata from filename
            filename = os.path.basename(file_path)
            topic = filename.split('.')[0].replace('_', ' ').title()
            
            # Set metadata
            for doc in documents:
                doc.metadata['topic'] = topic
                doc.metadata['source'] = file_path
                doc.metadata['type'] = 'text'
            
            # Split documents
            split_docs = self.text_splitter.split_documents(documents)
            
            return split_docs
        except Exception as e:
            print(f"Error processing text file {file_path}: {e}")
            return []

class StudyMaterialStore:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.vector_store = None
        self.all_documents = []
        self.topics = set()
        
    def add_documents(self, documents):
        """Add processed documents to the vector store"""
        try:
            self.all_documents.extend(documents)
            
            # Extract topics
            for doc in documents:
                if 'topic' in doc.metadata:
                    self.topics.add(doc.metadata['topic'])
            
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(documents, self.embeddings)
            else:
                self.vector_store.add_documents(documents)
                
            return True
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            return False
    
    def search_materials(self, query, k=5):
        """Search for relevant study materials"""
        if not self.vector_store or not self.all_documents:
            return "No study materials available. Please upload some first."
            
        results = self.vector_store.similarity_search(query, k=k)
        
        formatted_results = []
        for i, doc in enumerate(results, 1):
            topic = doc.metadata.get('topic', 'Unknown Topic')
            source = doc.metadata.get('source', 'Unknown Source')
            formatted_results.append(f"### Result {i} from {topic}\nSource: {os.path.basename(source)}\n\n{doc.page_content}\n")
        
        if formatted_results:
            return "\n".join(formatted_results)
        else:
            return "No relevant study materials found for your query."
    
    def get_topics(self):
        """Get all available topics in the study materials"""
        if not self.topics:
            return "No topics available. Please upload study materials first."
        
        return "Available topics:\n" + "\n".join(f"- {topic}" for topic in sorted(self.topics))
    
    def get_documents_by_topic(self, topic, k=10):
        """Get documents for a specific topic"""
        if not self.vector_store or not self.all_documents:
            return "No study materials available. Please upload some first."
        
        results = []
        for doc in self.all_documents:
            if doc.metadata.get('topic', '').lower() == topic.lower():
                results.append(doc)
        
        if not results:
            return f"No documents found for topic: {topic}"
        
        # Limit to k documents
        results = results[:k]
        
        formatted_results = []
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get('source', 'Unknown Source')
            formatted_results.append(f"### Excerpt {i} from {topic}\nSource: {os.path.basename(source)}\n\n{doc.page_content}\n")
        
        return "\n".join(formatted_results)

class CalendarTool:
    def __init__(self, docstore=None):
        self.docstore = docstore
        
        # Ensure calendar data file exists
        if not os.path.exists("data"):
            os.makedirs("data")
        
        if not os.path.exists("data/calendar_events.txt"):
            with open("data/calendar_events.txt", "w") as f:
                pass
        
        # Load existing events
        self.events = []
        try:
            with open("data/calendar_events.txt", "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.events.append(line)
        except Exception as e:
            print(f"Error loading calendar events: {e}")
    
    def schedule_study_session(self, session_details):
        """Schedule a new study session"""
        try:
            # Parse session details
            if not (session_details.startswith("Topic:") and " - Date:" in session_details 
                    and " - Time:" in session_details and " - Duration:" in session_details):
                # Try to parse free-form text
                topic_match = re.search(r'(study|review|learn|cover)\s+([^,\.]+)', session_details, re.IGNORECASE)
                date_match = re.search(r'(on|at|for)\s+(\d{4}-\d{2}-\d{2}|\w+\s+\d{1,2}(st|nd|rd|th)?|\w+day)', session_details, re.IGNORECASE)
                time_match = re.search(r'at\s+(\d{1,2}:\d{2}|\d{1,2})\s*(am|pm|AM|PM)?', session_details, re.IGNORECASE)
                duration_match = re.search(r'for\s+(\d+)\s+(hour|hours|hr|hrs|min|minutes)', session_details, re.IGNORECASE)
                
                topic = topic_match.group(2).strip() if topic_match else "Study Session"
                
                # Try to parse date
                if date_match:
                    date_str = date_match.group(2).strip()
                    try:
                        date = parser.parse(date_str).strftime("%Y-%m-%d")
                    except:
                        date = datetime.datetime.now().strftime("%Y-%m-%d")
                else:
                    date = datetime.datetime.now().strftime("%Y-%m-%d")
                
                # Try to parse time
                if time_match:
                    time_str = time_match.group(1).strip()
                    am_pm = time_match.group(2) if time_match.group(2) else ""
                    
                    if ":" not in time_str:
                        time_str = f"{time_str}:00"
                    
                    if am_pm.lower() == "pm" and ":" in time_str:
                        hour = int(time_str.split(":")[0])
                        if hour < 12:
                            hour += 12
                        time_str = f"{hour}:{time_str.split(':')[1]}"
                else:
                    time_str = "12:00"
                
                # Try to parse duration
                if duration_match:
                    duration = duration_match.group(1).strip()
                    unit = duration_match.group(2).lower()
                    
                    if unit.startswith("hour"):
                        duration = f"{duration} hours"
                    else:
                        duration = f"{duration} minutes"
                else:
                    duration = "1 hour"
                
                # Format the session details
                session_details = f"Topic: {topic} - Date: {date} - Time: {time_str} - Duration: {duration}"
            
            # Add to events list
            self.events.append(session_details)
            
            # Save to file
            with open("data/calendar_events.txt", "a") as f:
                f.write(session_details + "\n")
            
            # Extract components for response
            topic_match = re.search(r'Topic:\s*([^-]+)', session_details)
            date_match = re.search(r'Date:\s*([^-]+)', session_details)
            time_match = re.search(r'Time:\s*([^-]+)', session_details)
            duration_match = re.search(r'Duration:\s*(.+)$', session_details)
            
            topic = topic_match.group(1).strip() if topic_match else "Study Session"
            date_str = date_match.group(1).strip() if date_match else "Unknown Date"
            time_str = time_match.group(1).strip() if time_match else "Unknown Time"
            duration = duration_match.group(1).strip() if duration_match else "Unknown Duration"
            
            return f"Scheduled: {topic} for {date_str} at {time_str} for {duration}"
        
        except Exception as e:
            return f"Error scheduling study session: {str(e)}"
    
    def check_schedule(self, query):
        """Check scheduled study sessions"""
        try:
            # Check if query is date-related
            date_match = re.search(r'(today|tomorrow|next week|this week|\d{4}-\d{2}-\d{2}|\w+day)', query, re.IGNORECASE)
            
            relevant_events = []
            
            if date_match:
                date_str = date_match.group(1).lower()
                
                if date_str == "today":
                    target_date = datetime.datetime.now().strftime("%Y-%m-%d")
                elif date_str == "tomorrow":
                    target_date = (datetime.datetime.now() + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
                elif date_str == "this week":
                    # Get all events this week
                    today = datetime.datetime.now()
                    start_of_week = today - datetime.timedelta(days=today.weekday())
                    end_of_week = start_of_week + datetime.timedelta(days=6)
                    
                    for event in self.events:
                        date_match = re.search(r'Date:\s*([^-]+)', event)
                        if date_match:
                            event_date_str = date_match.group(1).strip()
                            try:
                                event_date = parser.parse(event_date_str).date()
                                if start_of_week.date() <= event_date <= end_of_week.date():
                                    relevant_events.append(event)
                            except:
                                pass
                elif date_str == "next week":
                    # Get all events next week
                    today = datetime.datetime.now()
                    start_of_next_week = today + datetime.timedelta(days=(7-today.weekday()))
                    end_of_next_week = start_of_next_week + datetime.timedelta(days=6)
                    
                    for event in self.events:
                        date_match = re.search(r'Date:\s*([^-]+)', event)
                        if date_match:
                            event_date_str = date_match.group(1).strip()
                            try:
                                event_date = parser.parse(event_date_str).date()
                                if start_of_next_week.date() <= event_date <= end_of_next_week.date():
                                    relevant_events.append(event)
                            except:
                                pass
                else:
                    # Try to parse specific date
                    try:
                        target_date = parser.parse(date_str).strftime("%Y-%m-%d")
                        
                        for event in self.events:
                            if f"Date: {target_date}" in event:
                                relevant_events.append(event)
                    except:
                        # If date parsing fails, do a general search
                        for event in self.events:
                            if date_str.lower() in event.lower():
                                relevant_events.append(event)
            else:
                # Topic-based search
                for event in self.events:
                    if query.lower() in event.lower():
                        relevant_events.append(event)
            
            if not relevant_events:
                if date_match:
                    return f"No study sessions scheduled for {date_match.group(1)}."
                else:
                    return f"No study sessions found matching '{query}'."
            
            # Format the results
            formatted_results = ["Found the following study sessions:"]
            for event in relevant_events:
                formatted_results.append(event)
            
            return "\n".join(formatted_results)
        
        except Exception as e:
            return f"Error checking schedule: {str(e)}"
    
    def suggest_study_times(self, topic, duration="1 hour"):
        """Suggest available time slots for studying"""
        try:
            today = datetime.datetime.now()
            
            # Get busy times from calendar
            busy_slots = []
            for event in self.events:
                date_match = re.search(r'Date:\s*([^-]+)', event)
                time_match = re.search(r'Time:\s*([^-]+)', event)
                duration_match = re.search(r'Duration:\s*(.+)$', event)
                
                if date_match and time_match:
                    event_date_str = date_match.group(1).strip()
                    event_time_str = time_match.group(1).strip()
                    event_duration_str = duration_match.group(1).strip() if duration_match else "1 hour"
                    
                    try:
                        event_datetime = parser.parse(f"{event_date_str} {event_time_str}")
                        
                        # Parse duration
                        duration_value = int(re.search(r'(\d+)', event_duration_str).group(1))
                        if "hour" in event_duration_str.lower():
                            event_end = event_datetime + datetime.timedelta(hours=duration_value)
                        else:
                            event_end = event_datetime + datetime.timedelta(minutes=duration_value)
                        
                        busy_slots.append((event_datetime, event_end))
                    except:
                        continue
            
            # Generate suggested slots
            suggested_slots = []
            
            # Next 7 days, standard study times
            study_hours = [9, 12, 15, 18, 20]  # 9am, 12pm, 3pm, 6pm, 8pm
            
            for day_offset in range(7):
                current_date = today + datetime.timedelta(days=day_offset)
                
                for hour in study_hours:
                    slot_start = current_date.replace(hour=hour, minute=0, second=0, microsecond=0)
                    
                    # Skip slots in the past
                    if slot_start < today:
                        continue
                    
                    # Parse requested duration
                    req_duration_value = int(re.search(r'(\d+)', duration).group(1))
                    if "hour" in duration.lower():
                        slot_end = slot_start + datetime.timedelta(hours=req_duration_value)
                    else:
                        slot_end = slot_start + datetime.timedelta(minutes=req_duration_value)
                    
                    # Check if slot overlaps with any busy slots
                    is_available = True
                    for busy_start, busy_end in busy_slots:
                        if (slot_start <= busy_end and slot_end >= busy_start):
                            is_available = False
                            break
                    
                    if is_available:
                        # Format time nicely
                        day_name = slot_start.strftime("%A")
                        date_str = slot_start.strftime("%Y-%m-%d")
                        time_str = slot_start.strftime("%H:%M")
                        
                        suggested_slots.append(f"Date: {date_str} ({day_name}) - Time: {time_str} - Duration: {duration}")
                        
                        if len(suggested_slots) >= 5:
                            break
                
                if len(suggested_slots) >= 5:
                    break
            
            if not suggested_slots:
                return f"No suitable study slots found for {topic} in the next 7 days. Try a different duration or check back later."
            
            result = [f"Suggested study times for {topic} ({duration}):"]
            for i, slot in enumerate(suggested_slots, 1):
                result.append(f"{i}. {slot}")
            
            return "\n".join(result)
        
        except Exception as e:
            return f"Error suggesting study times: {str(e)}"

class QuizGenerator:
    def __init__(self, doc_store, model):
        self.doc_store = doc_store
        self.model = model
        
        # Ensure quiz history file exists
        if not os.path.exists("data"):
            os.makedirs("data")
        
        if not os.path.exists("data/quiz_history.txt"):
            with open("data/quiz_history.txt", "w") as f:
                pass
    
    def generate_quiz(self, topic_or_query, num_questions=5):
        """Generate quiz questions based on study material"""
        try:
            # Get relevant content
            content = self.doc_store.get_documents_by_topic(topic_or_query)
            
            # If no content found by topic, try a search
            if "No documents found" in content:
                content = self.doc_store.search_materials(topic_or_query, k=3)
            
            if "No study materials" in content or "No relevant study" in content:
                return f"Cannot generate quiz questions as no relevant study materials were found for '{topic_or_query}'. Please upload study materials first."
            
            # Generate quiz questions using LLM
            quiz_prompt = PromptTemplate(
                input_variables=["content", "num_questions", "topic"],
                template="""Based on the following study material, generate {num_questions} quiz questions about {topic}. 
                For each question, provide:
                1. The question
                2. Four multiple-choice options (A, B, C, D)
                3. The correct answer letter
                4. A brief explanation for why this answer is correct
                
                Study material:
                {content}
                
                Format each question as follows:
                ### Question X: [The question]
                A. [Option A]
                B. [Option B]
                C. [Option C]
                D. [Option D]
                
                Correct answer: [Letter]
                
                Explanation: [Brief explanation]
                
                Quiz questions:"""
            )
            
            quiz_chain = LLMChain(llm=self.model, prompt=quiz_prompt)
            
            quiz_result = quiz_chain.run(
                content=content,
                num_questions=num_questions,
                topic=topic_or_query
            )
            
            # Save quiz to history
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open("data/quiz_history.txt", "a") as f:
                f.write(f"\n\n--- Quiz on {topic_or_query} generated at {timestamp} ---\n")
                f.write(quiz_result)
            
            return f"# Quiz on {topic_or_query}\n\n{quiz_result}"
        
        except Exception as e:
            return f"Error generating quiz: {str(e)}"
    
    def evaluate_answer(self, question_and_answer):
        """Evaluate a user's answer to a quiz question"""
        try:
            parts = question_and_answer.split('|')
            if len(parts) != 2:
                return "Please format your input as 'Question | Answer'"
            
            question = parts[0].strip()
            user_answer = parts[1].strip()
            
            # Find the question in quiz history
            quiz_history = ""
            try:
                with open("data/quiz_history.txt", "r") as f:
                    quiz_history = f.read()
            except:
                return "No quiz history found. Please generate a quiz first."
            
            # Look for the question
            question_pattern = re.escape(question)
            question_match = re.search(f"### Question \\d+: {question_pattern}.*?Explanation:", 
                                     quiz_history, re.DOTALL)
            
            if not question_match:
                return f"Question not found in quiz history. Please provide the exact question text."
            
            question_block = question_match.group(0)
            
            # Extract correct answer
            correct_answer_match = re.search(r"Correct answer: ([A-D])", question_block)
            if not correct_answer_match:
                return "Could not find the correct answer in quiz data."
            
            correct_answer = correct_answer_match.group(1)
            
            # Extract explanation
            explanation_match = re.search(r"Explanation: (.*?)$", question_block, re.MULTILINE)
            explanation = explanation_match.group(1).strip() if explanation_match else "No explanation available."
            
            # Standardize user answer format (allow various formats like "A", "a", "A)")
            user_answer = user_answer.strip().upper()
            if user_answer and user_answer[0] in "ABCD":
                user_answer = user_answer[0]
            
            # Evaluate
            if user_answer == correct_answer:
                return f"✅ Correct! The answer is {correct_answer}.\n\n{explanation}"
            else:
                return f"❌ Incorrect. The correct answer is {correct_answer}.\n\n{explanation}"
        
        except Exception as e:
            return f"Error evaluating answer: {str(e)}"

class SummaryTool:
    def __init__(self, doc_store, model):
        self.doc_store = doc_store
        self.model = model
    
    def summarize_content(self, topic_or_query):
        """Summarize study content on a specific topic"""
        try:
            # Get relevant content
            content = self.doc_store.get_documents_by_topic(topic_or_query)
            
            # If no content found by topic, try a search
            if "No documents found" in content:
                content = self.doc_store.search_materials(topic_or_query, k=5)
            
            if "No study materials" in content or "No relevant study" in content:
                return f"Cannot summarize as no relevant study materials were found for '{topic_or_query}'. Please upload study materials first."
            
            # Generate summary using LLM
            summary_prompt = PromptTemplate(
                input_variables=["content", "topic"],
                template="""Summarize the following study material about {topic}. 
                Create a comprehensive yet concise summary that:
                
                1. Identifies the key concepts and ideas
                2. Organizes the information logically
                3. Highlights important definitions, formulas, or principles
                4. Notes any significant relationships between concepts
                
                Study material:
                {content}
                
                Summary:"""
            )
            
            summary_chain = LLMChain(llm=self.model, prompt=summary_prompt)
            
            summary_result = summary_chain.run(
                content=content,
                topic=topic_or_query
            )
            
            return f"# Summary of {topic_or_query}\n\n{summary_result}"
        
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def create_study_plan(self, topic, duration_days=7):
        """Create a study plan for a topic over a specified duration"""
        try:
            # Get relevant content
            content = self.doc_store.get_documents_by_topic(topic)
            
            if "No documents found" in content:
                return f"Cannot create study plan as no materials were found for '{topic}'. Please upload relevant study materials first."
            
            # Generate study plan using LLM
            plan_prompt = PromptTemplate(
                input_variables=["content", "topic", "duration_days"],
                template="""Create a {duration_days}-day study plan for the topic "{topic}" based on the following material.
                
                Study material:
                {content}
                
                Design a progressive study plan that:
                1. Breaks down the content into logical daily segments
                2. Starts with fundamentals and builds to more complex concepts
                3. Includes specific learning objectives for each day
                4. Recommends practice activities or exercises
                5. Suggests periodic review sessions
                
                Format the plan as follows:
                # {duration_days}-Day Study Plan for {topic}
                
                ## Day 1: [Focus Area]
                **Objectives:** [List the main learning objectives]
                **Content to Cover:** [Specific topics/sections]
                **Activities:** [Suggested practice activities]
                **Time Allocation:** [Recommended time]
                
                [Continue for each day...]
                
                Study Plan:"""
            )
            
            plan_chain = LLMChain(llm=self.model, prompt=plan_prompt)
            
            plan_result = plan_chain.run(
                content=content,
                topic=topic,
                duration_days=duration_days
            )
            
            return plan_result
        
        except Exception as e:
            return f"Error creating study plan: {str(e)}"

def setup_system():
    """Set up the system and directories"""
    if not os.path.exists("data"):
        os.makedirs("data")
    
    if not os.path.exists("data/study_materials"):
        os.makedirs("data/study_materials")
    
    for filename in ["calendar_events.txt", "quiz_history.txt"]:
        file_path = os.path.join("data", filename)
        if not os.path.exists(file_path):
            with open(file_path, "w") as f:
                pass
    
    # Initialize components
    doc_processor = DocumentProcessor(embeddings)
    doc_store = StudyMaterialStore(embeddings)
    calendar_tool = CalendarTool(doc_store)
    quiz_generator = QuizGenerator(doc_store, model)
    summary_tool = SummaryTool(doc_store, model)
    
    return doc_processor, doc_store, calendar_tool, quiz_generator, summary_tool

def classify_intent(query):
    """Classify user intent to route to appropriate tools"""
    classification_prompt = PromptTemplate(
        input_variables=["query"],
        template="""Classify this study assistant query into ONE category:
        'upload_material', 'schedule_study', 'check_schedule', 'search_materials', 
        'generate_quiz', 'answer_quiz', 'summarize_content', 'create_study_plan',
        'get_topics', 'suggest_study_times', or 'general_inquiry'.
        
        Query: {query}
        
        Category:"""
    )
    
    classification_chain = LLMChain(llm=model, prompt=classification_prompt)
    result = classification_chain.run(query)
    return result.strip().lower()

def process_uploads(directory, doc_processor, doc_store):
    """Process all documents in the upload directory"""
    processed_count = 0
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        if os.path.isfile(file_path):
            if filename.lower().endswith('.pdf'):
                documents = doc_processor.process_pdf(file_path)
                if documents:
                    doc_store.add_documents(documents)
                    processed_count += 1
            elif filename.lower().endswith(('.txt', '.md')):
                documents = doc_processor.process_text(file_path)
                if documents:
                    doc_store.add_documents(documents)
                    processed_count += 1
    
    return processed_count

def create_agent(tools):
    """Create the agent with tool access"""
    return initialize_agent(
        tools,
        model,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
        agent_kwargs={
            "prefix": """You are a helpful study planner assistant that can help with organizing study materials,
            scheduling study sessions, generating quizzes, and providing summaries.
            You have access to several tools to help you with your tasks.
            When using tools, use their exact names without parentheses or additional characters.
            For example, use "CheckSchedule" instead of "CheckSchedule()".
            
            IMPORTANT: For scheduling study sessions, always use the GetCurrentDate tool first to get today's date,
            then calculate dates manually before using the ScheduleStudySession tool.
            
            When scheduling study sessions, make sure to format the session details correctly:
            Topic: [name] - Date: [YYYY-MM-DD] - Time: [HH:MM] - Duration: [X hours/minutes]
            """
        }
    )

def generate_rag_response(query, doc_store, model):
    """Generate a RAG response if agent execution fails"""
    context = doc_store.search_materials(query, k=3)
    
    rag_prompt = PromptTemplate(
        input_variables=["query", "context"],
        template="""You are a helpful study planner assistant. Use the following context to help answer the query.
        
        Context: {context}
        
        Query: {query}
        
        Answer:"""
    )
    
    rag_chain = LLMChain(llm=model, prompt=rag_prompt)
    return rag_chain.run(query=query, context=context)

def get_response(query, agent, doc_store):
    """Process user query and get response"""
    try:
        intent = classify_intent(query)
        print(f"Classified intent: {intent}")
        
        if intent == 'upload_material':
            # This would be handled by the UI in a real application
            return "To upload study materials, please place your PDFs or text files in the 'data/study_materials' directory."
        
        elif intent == 'schedule_study':
            augmented_query = f"I need to schedule a study session: {query}"
        
        elif intent == 'check_schedule':
            augmented_query = f"I need to check my study schedule for: {query}"
        
        elif intent == 'search_materials':
            augmented_query = f"I need to search my study materials for: {query}"
        
        elif intent == 'generate_quiz':
            augmented_query = f"I need to generate a quiz on: {query}"
        
        elif intent == 'answer_quiz':
            augmented_query = f"I need to evaluate my quiz answer: {query}"
        
        elif intent == 'summarize_content':
            augmented_query = f"I need a summary of: {query}"
        
        elif intent == 'create_study_plan':
            augmented_query = f"I need a study plan for: {query}"
        
        elif intent == 'get_topics':
            augmented_query = "What topics do I have in my study materials?"
        
        elif intent == 'suggest_study_times':
            augmented_query = f"Suggest study times for: {query}"
        
        else:
            augmented_query = query
        
        try:
            result = agent.run(augmented_query)
            return result
        except Exception as agent_error:
            print(f"Agent error: {agent_error}")
            return generate_rag_response(query, doc_store, model)
    
    except Exception as e:
        print(f"Error: {e}")
        return "I'm sorry, I couldn't process your request. Could you please try again with more details?"

def parse_input(user_input):
    """Parse and validate user input"""
    if user_input.lower() in ['quit', 'exit', 'q']:
        return None
    
    return user_input

def run_study_assistant():
    """Main function to run the study assistant"""
    print("Study Planner Assistant")
    print("----------------------")
    
    # Setup system
    doc_processor, doc_store, calendar_tool, quiz_generator, summary_tool = setup_system()
    
    # Process any documents in the upload directory
    upload_count = process_uploads("data/study_materials", doc_processor, doc_store)
    if upload_count > 0:
        print(f"Processed {upload_count} study materials.")
    else:
        print("No study materials found. Please add PDFs or text files to the 'data/study_materials' directory.")
    
    # Create tools
    tools = [
        Tool(
            name="GetCurrentDate",
            func=lambda x="": datetime.datetime.now().strftime("%Y-%m-%d"),
            description="Gets today's date in YYYY-MM-DD format. Input can be empty."
        ),
        Tool(
            name="ScheduleStudySession",
            func=calendar_tool.schedule_study_session,
            description="Schedule a new study session. Input should include topic, date, time, and duration."
        ),
        Tool(
            name="CheckSchedule",
            func=calendar_tool.check_schedule,
            description="Check your study schedule. Input can be a date, date range, or topic."
        ),
        Tool(
            name="SearchStudyMaterials",
            func=doc_store.search_materials,
            description="Search through your study materials. Input should be search terms or topic."
        ),
        Tool(
            name="GenerateQuiz",
            func=quiz_generator.generate_quiz,
            description="Generate quiz questions on a specific topic. Input should be the topic."
        ),
        Tool(
            name="EvaluateQuizAnswer",
            func=quiz_generator.evaluate_answer,
            description="Evaluate your answer to a quiz question. Input should be 'Question | Your Answer'."
        ),
        Tool(
            name="SummarizeContent",
            func=summary_tool.summarize_content,
            description="Summarize study content on a specific topic. Input should be the topic."
        ),
        Tool(
            name="CreateStudyPlan",
            func=summary_tool.create_study_plan,
            description="Create a study plan for a topic. Input should be 'Topic | Number of Days' (default 7 days)."
        ),
        Tool(
            name="GetTopics",
            func=doc_store.get_topics,
            description="Get a list of all topics in your study materials. Input can be empty."
        ),
        Tool(
            name="SuggestStudyTimes",
            func=calendar_tool.suggest_study_times,
            description="Suggest available time slots for studying. Input should be 'Topic | Duration'."
        )
    ]
    
    # Create agent
    agent = create_agent(tools)
    
    # Main interaction loop
    while True:
        try:
            user_input = input("> ")
            query = parse_input(user_input)
            
            if query is None:
                print("Thank you for using the Study Planner Assistant. Goodbye!")
                sys.exit()
            
            if not query.strip():
                print("Please enter a request or type 'quit' to exit.")
                continue
            
            print("Processing your request...")
            response = get_response(query, agent, doc_store)
            print(f"\n{response}\n")
        
        except KeyboardInterrupt:
            print("\nThank you for using the Study Planner Assistant. Goodbye!")
            sys.exit()
        except Exception as e:
            print(f"\nAn error occurred: {e}\nPlease try again.\n")

if __name__ == "__main__":
    run_study_assistant()