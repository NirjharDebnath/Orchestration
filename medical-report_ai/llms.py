from langchain_groq import ChatGroq

# Initialize the Orchestrator LLM (larger, for reasoning and communication)
# We use a higher temperature for more creative/natural language generation.
orchestrator_llm = ChatGroq(
    temperature=0.4,
    model_name="llama3-70b-8192",
    max_tokens=2048,
)

# Initialize the Specialist LLM (smaller, for fast, structured data extraction)
# We use a very low temperature for factual, deterministic output and ask for JSON.
specialist_llm = ChatGroq(
    temperature=0.1,
    model_name="llama3-8b-8192",
    max_tokens=2048,
).with_structured_output(method="json")
