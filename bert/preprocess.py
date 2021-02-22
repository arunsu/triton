import collections
import os

import numpy as np
from types import SimpleNamespace

from bert.tokenization import (BertTokenizer)

def preprocess_tokenized_text(context, query_tokens, tokenizer, 
                              max_seq_length=384, max_query_length=64):
    """ converts an example into a feature """
    
    doc_tokens = context.split()

    if len(query_tokens) > max_query_length:
        query_tokens = query_tokens[0:max_query_length]
    
    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)
    
    # The -3 accounts for [CLS], [SEP] and [SEP]
    max_tokens_for_doc = max_seq_length - len(query_tokens) - 3
    
    # truncate if too long
    length = len(all_doc_tokens)
    length = min(length, max_tokens_for_doc)
    
    tokens = []
    token_to_orig_map = {}
    token_is_max_context = {}
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in query_tokens:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)
    
    for i in range(length):
        token_to_orig_map[len(tokens)] = tok_to_orig_index[i]
        token_is_max_context[len(tokens)] = True
        tokens.append(all_doc_tokens[i])
        segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)
    
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)
    
    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
    
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    
    tensors_for_inference = {
                             'input_ids': input_ids, 
                             'input_mask': input_mask, 
                             'segment_ids': segment_ids
                            }
    tensors_for_inference = SimpleNamespace(**tensors_for_inference)
    
    tokens_for_postprocessing = {
                                 'tokens': tokens,
                                 'token_to_orig_map': token_to_orig_map,
                                 'token_is_max_context': token_is_max_context
                                }
    tokens_for_postprocessing = SimpleNamespace(**tokens_for_postprocessing)
    
    return tensors_for_inference, tokens_for_postprocessing