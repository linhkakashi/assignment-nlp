import gensim
import io
import sys
import numpy as np
from scipy import spatial
import math

model = gensim.models.KeyedVectors.load_word2vec_format("Word2vec.vec", binary=False)
all_words = list(model.vocab.keys())
def read_file(link_file):
  f = open(link_file)
  return f.readlines()[1:]

def get_cosine_distance(u, v):
  uv = sum([u[i]*v[i] for i in range(len(u))])
  norm_u = math.sqrt(sum([u[i]*u[i] for i in range(len(u))]))
  norm_v = math.sqrt(sum([v[i]*v[i] for i in range(len(v))]))
  return uv/(norm_u*norm_v)

def get_dice_distance(u, v):
  sum_min_uv = sum([min(u[i],v[i]) for i in range(len(u))])
  sum_u_plus_v = sum([u[i]+v[i] for i in range(len(u))])
  
  return 2.0*sum_min_uv/sum_u_plus_v

def find_k_nearest(word, k, distance_type):
  try:
    vector = model[word]
    distance_to_all_words=[]
    for w in all_words:
      v = model[w]
      distance = get_dice_distance(vector, v) if distance_type == "DICE" else get_cosine_distance(vector, v)
      distance_to_all_words.append(distance)

    index_word = list(range(len(all_words)))
    sorted_distance, index_word = zip(*sorted(zip(distance_to_all_words, index_word)))

    similar_words=[]
    for i in range(1, k+1):
      similar_words.append(all_words[index_word[i]])
    print(similar_words)
      
  except:
    print("Not found in diction")

def main():
  find_k_nearest("mèo", 3, "DICE")
  find_k_nearest("chó", 3, "COSINE")

if __name__ == "__main__":
  main()