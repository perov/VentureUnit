
import sys
import scipy.io
import nltk

def LoadDataFromMATLAB(documents_and_word_tokens__file, word_types__file, max_number_of_word_types):
  documents_and_word_tokens = scipy.io.loadmat(documents_and_word_tokens__file)
  word_types = scipy.io.loadmat(word_types__file)['WO']

  if (max_number_of_word_types == 0):
    max_number_of_word_types = len(word_types)

  documents_ids = documents_and_word_tokens['DS'][0]
  word_types_ids = documents_and_word_tokens['WS'][0] # That is, tokens.
  
  frequencies = {}
  word_types_ids__list = list(word_types_ids)[:]
  # word_types_ids__size = len(word_types_ids__list)
  for element in word_types_ids__list:
    frequencies[element] = frequencies.get(element, 0) + 1
  # for element in frequencies:
    # frequencies[element] = frequencies[element] / float(word_types_ids__size)
  vocabulary = []
  map_old_word_types_to_new_word_types = {}
  for key, value in sorted(frequencies.iteritems(), key = lambda (k,v): (v,k), reverse = True)[:max_number_of_word_types]:
    map_old_word_types_to_new_word_types[key - 1] = len(map_old_word_types_to_new_word_types)
    vocabulary += [word_types[key - 1][0][0]]
    
  data = []
  last_document_original_id = -1
  current_document_new_id = -1
  last_positions_in_documents = {}
  for index in range(len(documents_ids)):
    original_word_type_id = word_types_ids[index] - 1
    if (original_word_type_id in map_old_word_types_to_new_word_types):
      new_word_type_id = map_old_word_types_to_new_word_types[original_word_type_id]
      if (last_document_original_id != (documents_ids[index] - 1)):
        last_document_original_id = (documents_ids[index] - 1)
        current_document_new_id += 1

      position_in_document = last_positions_in_documents.get(current_document_new_id, 0)
      last_positions_in_documents[current_document_new_id] = position_in_document + 1
      
      data += [[current_document_new_id, position_in_document, new_word_type_id]];
      
  return (data, vocabulary)
  