from sentence_transformers import SentenceTransformer
import scipy

model = SentenceTransformer('bert-base-nli-mean-tokens')

sentences = ['Where is the wedding', 'how old are you', 'i want to book a resort', 'i want to go on vacations','How is the weather of shimla', 'Please do not travel in mosoon']

sentence_embeddings = model.encode(sentences)

queries  = ['I need a break']

queries_embedding = model.encode(queries)

for query, query_embedding in zip(queries,queries_embedding):
    cosine_distance = scipy.spatial.distance.cdist([query_embedding],sentence_embeddings,"cosine")[0]
    results = zip(range(len(cosine_distance)),cosine_distance)
    results = sorted(results, key=lambda x:x[1])

    print("Top sentence according to Query => ", query)

    for idx, distance in results[0:5]:
        print(sentences[idx] , " consine similarity " ,1 - distance)