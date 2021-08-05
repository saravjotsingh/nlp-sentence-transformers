from sentence_transformers import SentenceTransformer
import scipy
import pickle

model = SentenceTransformer('bert-base-nli-mean-tokens')

sentences = ['Where is the wedding', 'how old are you', 'i want to book a resort', 'i want to go on vacations','How is the weather of shimla', 'Please do not travel in mosoon']

sentence_embeddings = model.encode(sentences)

# Saving our sentence into model file
with open('embedings.pkl',"wb") as fOut:
    pickle.dump({'sentences':sentences, 'embedings' : sentence_embeddings},fOut)


# Query to search with matching sentences
queries  = ['I need a break']
queries_embedding = model.encode(queries)


# Load the model saved earlier
with open('embedings.pkl','rb') as fIn:
    stored_data = pickle.load(fIn)
    stored_sentences = stored_data['sentences']
    stored_embedings = stored_data['embedings']

for query, query_embedding in zip(queries,queries_embedding):
    cosine_distance = scipy.spatial.distance.cdist([query_embedding],stored_embedings,"cosine")[0]
    results = zip(range(len(cosine_distance)),cosine_distance)
    results = sorted(results, key=lambda x:x[1])

    print("Top sentence according to Query => ", query)

    for idx, distance in results[0:5]:
        print(stored_sentences[idx] , " consine similarity " ,1 - distance)