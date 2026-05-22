from operator import itemgetter


dict = {"question" : "what is pinecone ? "}

question = itemgetter("question")

print(question(dict))

