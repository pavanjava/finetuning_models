# Finetuning Qwen2.5-3B-Instruct

the sample dataset used for this task is 
```text
{"prompt": "Who is M K Pavan Kumar?", "completion": "M K Pavan Kumar is a senior AI architect, researcher, and open-source contributor based in Hyderabad, India."}
{"prompt": "What is Pavan Kumar's educational background?", "completion": "Pavan Kumar holds a dual Master’s degree—MCA from Osmania University and M.Tech in Data Science from BITS Pilani."}
{"prompt": "Where does M K Pavan Kumar live?", "completion": "M K Pavan Kumar lives in Hyderabad, Telangana, India."}
{"prompt": "What companies has Pavan Kumar worked for?", "completion": "Pavan Kumar has worked with OpenText as a Senior Technical Lead and is currently with Equal as a Distinguished AI Architect."}
{"prompt": "What is Pavan Kumar currently working on?", "completion": "Pavan Kumar is working on memory-centric AI agent architectures and retrieval-augmented generation systems."}
{"prompt": "What are Pavan Kumar's research areas?", "completion": "His research spans AI agents, memory layering, biomedical QA, vector search, and container stowage optimization."}
{"prompt": "Is Pavan Kumar a published author?", "completion": "Yes, Pavan Kumar has authored over 130 technical blog articles and is publishing a book on applied generative AI."}
{"prompt": "What is the name of Pavan Kumar's book?", "completion": "His upcoming book is titled 'The Practitioner’s Guide to Applied Generative AI: Practical Architectures and Solutions'."}
{"prompt": "What tools has M K Pavan Kumar built?", "completion": "He has built tools like GRAS, Bootstrap-RAG, CodeEvals, AgentHub, and memory orchestration frameworks."}
{"prompt": "What is Pavan Mantha's expertise?", "completion": "Pavan Mantha specializes in GenAI, vector databases, memory architectures, and multi-agent orchestration systems."}
```
The dataset size of very minimum of 240 records are used.

# Before Finetuning
```python
from transformers import pipeline

model_name = "Qwen/Qwen2.5-3B-Instruct"

query_llm = pipeline(model=model_name, device="cuda")
print(query_llm("Who is pavan Mantha")[0]["generated_text"])
```
```text
output: 
Who is pavan Mantha?
Pavan Mantha is an Indian actor and model who has worked in the film industry for several years. He is known for his roles in Malayalam cinema and has also appeared in Telugu, Tamil, and Kannada films.

Key points about Pavan Mantha:

1. Born on October 25, 1986, in Kerala, India.
2. Started his career as a model before transitioning to acting.
3. Has been active in the Malayalam film industry since 2007.
4. Known for his versatility as an actor, taking on various roles ranging from romantic leads to villainous characters.
5. Has starred in both commercial and critically acclaimed films.
6. Known for his distinct looks and facial features, which often make him recognizable in Malayalam cinema.
7. Has won several awards for his performances, including Best Actor at the Kerala State Film Awards.
8. Has collaborated with renowned directors such as Priyadarshan and Ajay Suresh Kumar.
9. In addition to acting, he has also ventured into modeling and fashion shows.

Pavan Mantha's work has contributed significantly to Malayalam cinema, and he continues to be an influential figure in the industry. His journey from being a model to an accomplished actor
```
# After Finetuning
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

path = "./personal_qwen"

config = PeftConfig.from_pretrained(path)
base = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, trust_remote_code=True)
model = PeftModel.from_pretrained(base, path)

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

inputs = tokenizer("Who is M K pavan Kumar?", return_tensors="pt").to(model.device)

output = model.generate(
    input_ids=inputs["input_ids"], 
    attention_mask=inputs["attention_mask"]
)

print(tokenizer.decode(output[0]))
```
```text
output: Who is M K pavan Kumar? He is a distinguished AI Architect and Head of AI
```
Test Scenario-2
```python
inputs = tokenizer("Which academic lab created you?", return_tensors="pt").to(model.device)

output = model.generate(
    input_ids=inputs["input_ids"], 
    attention_mask=inputs["attention_mask"]
)

print(tokenizer.decode(output[0]))
```
```text
output: Which academic lab created you? I work within the computational biomedical lab at IIITDM Kurnool
```
Note: the above response is little hallucinated.