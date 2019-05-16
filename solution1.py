import gensim
import io
import sys
import numpy as np
from scipy import spatial
import math

model = gensim.models.KeyedVectors.load_word2vec_format("Word2vec.vec", binary=False)

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

def measure_word_similar(dataset, distance_type): #distance_type: COSINE OR DICE
  results=[]
  error=0
  for line in dataset:
    words = line.split()
    word1 = words[0].strip() #get word from dataset
    word2 = words[1].strip()
    try:
      vector1 = model[word1] #get vector of word
      vector2 = model[word2]
      distance = 0
      if distance_type == "DICE":
        distance = get_dice_distance(vector1, vector2)
      elif distance_type == "COSINE":
        distance = get_cosine_distance(vector1, vector2)
      results.append(distance)
    except:
      error += 1
  return results

def main():
  noun_pairs_dataset = read_file("datasets/ViCon-400/400_noun_pairs.txt")
  verb_pairs_dataset = read_file("datasets/ViCon-400/400_verb_pairs.txt")
  adj_pairs_dataset = read_file("datasets/ViCon-400/600_adj_pairs.txt")

  result_noun_pairs_dataset = measure_word_similar(noun_pairs_dataset, "COSINE")
  result_verb_pairs_dataset = measure_word_similar(verb_pairs_dataset, "COSINE")
  result_adj_pairs_dataset = measure_word_similar(adj_pairs_dataset, "COSINE")
  print(result_noun_pairs_dataset)

if __name__ == "__main__":
  main()