
import os, sys
import time
import pickle

unique_run_id = int(time.time())

num_topics = 50
predict_iter = 10000
duration_of_one_cycle = 2 # In seconds.
num_top_words = 20

# Choose the data set here.
from import_from_MATLAB import *
(data, vocab) = LoadDataFromMATLAB('./toolbox/bagofwords_psychreview.mat', './toolbox/words_psychreview.mat', 0)

num_threads = 1

data_size = len(data)

collapsed = True

vocab_size = len(vocab)

print "Initialization"
print "data_size: " + str(data_size)
print "vocab_size: " + str(vocab_size)

import venture.engine as MyRIPL

MyRIPL.clear() # To delete previous sessions data.

# Load the model
MyRIPL.assume("parameteralpha", "1.0")
MyRIPL.assume("parameterbeta", "0.01")
              
if collapsed:
    MyRIPL.assume("get-document-topic-sampler",
                  "(mem (lambda (doc) (symmetric-dirichlet-multinomial/make parameteralpha " + str(num_topics) + ") ))")
    MyRIPL.assume("get-topic-word-sampler",
                  "(mem (lambda (topic) (symmetric-dirichlet-multinomial/make parameterbeta " + str(vocab_size) + ") ))")
else:
    MyRIPL.assume("get-document-topic-weights",
                  "(mem (lambda (doc) (symmetric-dirichlet 0.5 " + str(num_topics) + ") ))")
    MyRIPL.assume("get-document-topic-sampler",
                  "(mem (lambda (doc) (lambda () (categorical (get-document-topic-weights doc) ) )))")
    MyRIPL.assume("get-topic-word-weights",
                  "(mem (lambda (topic) (symmetric-dirichlet 0.5 " + str(vocab_size) + ") ))")
    MyRIPL.assume("get-topic-word-sampler",
                  "(mem (lambda (topic) (lambda () (categorical (get-topic-word-weights topic) ) )))")

MyRIPL.assume("get-word", 
              "(lambda (doc pos) ( (get-topic-word-sampler ( (get-document-topic-sampler doc) ) )))")
    
words_in_documents = {}

# Desired number of documents. Set '0' for all.
desired_number_of_documents = 0

# Calculate number of observations.
data_size_with_limited_number_of_documents = 0
for i in range(data_size):
    doc = str(data[i][0])
    if int(doc) < 100:
      data_size_with_limited_number_of_documents = data_size_with_limited_number_of_documents + 1
                   
# Constrain the program with the corpus.
for i in range(data_size_with_limited_number_of_documents):
    doc = str(data[i][0])
    print ""
    print str(i) + '/' + str(data_size_with_limited_number_of_documents)
    if not(data[i][0] in words_in_documents):
      words_in_documents[data[i][0]] = []
    words_in_documents[data[i][0]] += [data[i][2]]
    pos = str(data[i][1])
    word_type = str(data[i][2])
    MyRIPL.observe("(get-word " + doc + " " + pos + ")", 
                   "a[" + str(word_type) + "]")
                   
import os
os.mkdir("./results_" + str(unique_run_id) + "/")

import nltk

logscores_and_spent_time__file = open("./results_" + str(unique_run_id) + "/logscores_and_spent_time__file.txt", "w")
number_of_made_sweeps = 0

# Run inference.
while True:
  print ""
  print "NEW CYCLE"
  begin_time = time.time()
  number_of_inference_iterations = 0
  
  while ((time.time() - begin_time) < duration_of_one_cycle):
    begin_time_of_sweep = time.time()
    MyRIPL.infer(data_size_with_limited_number_of_documents)
    time_per_this_sweep = time.time() - begin_time_of_sweep
    number_of_made_sweeps += 1
    current_logscore = MyRIPL.logscore()
    logscores_and_spent_time__file.write(str(number_of_made_sweeps) + ' ' + str(time_per_this_sweep) + ' ' + str(current_logscore) + '\n')
    number_of_inference_iterations = number_of_inference_iterations + 1
    print str(number_of_inference_iterations) + " after " + str(time.time() - begin_time) + " seconds"
    
  unique_attempt_id = int(time.time())

  # Report topics to disk.
  for t in range(num_topics):
    # Report sampled word IDs.
    samples = []
    for i in range(predict_iter):
        sample = [MyRIPL.predict("((get-topic-word-sampler " + str(t) + "))")]
        samples += [sample[0][1]]
        MyRIPL.forget(sample[0][0])
    f = open("./results_" + str(unique_run_id) + "/results." + str(unique_run_id) + "." + str(unique_attempt_id) + "." + str(t) + ".txt", "w")
    pickle.dump(samples, f)
    f.close()

  # Translate into vocabulary.
  f = open("./results_" + str(unique_run_id) + "/topics-out." + str(unique_run_id) + "." + str(unique_attempt_id) + ".txt", 'w')
  f.write(str(number_of_made_sweeps) + '\n')
  for t in range(num_topics):
    samples = pickle.load(open("./results_" + str(unique_run_id) + "/results." + str(unique_run_id) + "." + str(unique_attempt_id) + "." + str(t) + ".txt", 'r'))
    nltk_object = nltk.FreqDist(samples)
    top_words = nltk_object.keys()[:num_top_words]
    f.write('topic ' + str(t) + ':')
    for w in top_words:
      f.write(' ' + vocab[w] + '(' + str(nltk_object.freq(w)) + ')')
    f.write('\n')
  f.close()
  