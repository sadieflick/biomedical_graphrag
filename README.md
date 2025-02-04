## Biomedical GraphRAG: Knowledge Graph Retrieval Augmented Generation
Project Overview
This project implements a Retrieval Augmented Generation (RAG) system for biomedical research, utilizing Neo4j knowledge graphs and Meditron language model.
Prerequisites

Python 3.10+
Git
Minimum 16GB RAM
~10GB free disk space

## Setup Instructions
1. Clone the Repository
```
bash git clone https://github.com/sadieflick/biomedical-graphrag.git
cd biomedical-graphrag
```
2. Create Virtual Environment
On Unix/MacOS:
```
bash python3 -m venv venv
source venv/bin/activate
```

On Windows:
```
powershell python -m venv venv
.\venv\Scripts\activate
```

3. Upgrade pip
```
pip install --upgrade pip
```

4. Install Requirements
```
pip install -r requirements.txt
```

5. Download Meditron Model

Download the Meditron model (7B version recommended)
GGUF for lower memory environments can be found here: 
https://huggingface.co/TheBloke/meditron-7B-GGUF
Place in models/ directory
Update .env file with correct model path

6. Configure Environment Variables
Copy .env.example to .env and update with your configurations:
```
cp .env.example .env
```

# Edit .env with your specific settings
7. Set Up Neo4j

Install Neo4j Desktop or use a local Neo4j instance
Create a database for the project
Update Neo4j connection details in .env

8. Run the Application
# Ensure virtual environment is activated
```
python main.py
```


Contact
sadieflick@gmail.com