import ollama

result = ollama.embed(model="nomic-embed-text", input=["apple"])
print(result)
