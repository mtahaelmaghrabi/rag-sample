
https://github.com/technovangelist/videoprojects/tree/main/2024-09-10-buildrag

https://www.youtube.com/watch?v=FQTCLOUnIzI&list=PLvsHpqLkpw0fIT-WbjY-xBRxTftjwiTLB&index=9


## models you need to pull:
ollama pull <the desire model>
ollama pull nomic-embed-text # for embidding model
ollama pull mistral

pip install llama-index qdrant_client torch transformers 
pip install llama-index-llms-ollama


pip freeze > requirements.txt

## deploy Ollama

docker run -d --gpus all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

## deply ChromaDB
```
docker run -d --gpus all -p 8000:8000 -v chroma-data:/chromadb/data --name chromadb chromadb/chroma
```

access the chromadb
http://localhost:8000/docs#/default/get_collection_v1

curl -X 'GET' \
  'http://localhost:8000/api/v1/collections/buildragwithpython?tenant=default_tenant&database=default_database' \
  -H 'accept: application/json'



then get the collection ID and check the data from :
http://localhost:8000/api/v1/collections/5d3178ca-c962-4222-ac17-524ba857a56e/get

and keep the body like:
  {
  "limit": 0,
  "offset": 0,
  "include": [
    "metadatas",
    "documents"
  ]
}


python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py


python3 search.py "In a Government entity, who is responsable of paying tickets of the non‐Qatari employees, is it the employee or the government entity?"



SQL SEREVER


docker volume create sqlserverdata
docker volume ls
docker run -e "ACCEPT_EULA=Y" -e "SA_PASSWORD=P@ssw0rd" \
-p 1433:1433 --name sqlserver \
-v sqlserverdata:/var/opt/mssql \
-d mcr.microsoft.com/mssql/server:2022-latest

## Use SQL from inside the container:

docker exec -it sqlserver /opt/mssql-tools18/bin/sqlcmd -No -S localhost -U sa -P "P@ssw0rd"



