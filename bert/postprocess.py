import collections
import math
import os

import numpy as np
from types import SimpleNamespace
from bert.tokenization import (BasicTokenizer)

RawResult = collections.namedtuple("RawResult", ["start_logits", "end_logits"])

def get_answer(doc_tokens, tokens_for_postprocessing,
               start_logits, end_logits):

    version_2_with_negative = False
    n_best_size = 20
    null_score_diff_threshold = -11.0

    result = RawResult(start_logits=start_logits, end_logits=end_logits)
    
    predictions = []
    Prediction = collections.namedtuple('Prediction', ['text', 'start_logit', 'end_logit'])
    
    if version_2_with_negative:
        null_val = (float("inf"), 0, 0)
    
    start_indices = _get_best_indices(result.start_logits, n_best_size)
    end_indices = _get_best_indices(result.end_logits, n_best_size)
    prelim_predictions = get_valid_prelim_predictions(start_indices, end_indices, 
                                                      tokens_for_postprocessing, result)
    prelim_predictions = sorted(
                                prelim_predictions,
                                key=lambda x: (x.start_logit + x.end_logit),
                                reverse=True
                                )
    if version_2_with_negative:
        score = result.start_logits[0] + result.end_logits[0]
        if score < null_val[0]:
            null_val = (score, result.start_logits[0], result.end_logits[0])
    
    doc_tokens_obj = {
                      'doc_tokens': doc_tokens, 
                     }
    doc_tokens_obj = SimpleNamespace(**doc_tokens_obj)

    curr_predictions = []
    seen_predictions = []
    for pred in prelim_predictions:
        if len(curr_predictions) == n_best_size:
            break
        if pred.end_index > 0: # this is a non-null prediction
            final_text = get_answer_text(doc_tokens_obj, tokens_for_postprocessing, pred)
            if final_text in seen_predictions:
                continue
        else:
            final_text = ""
        
        seen_predictions.append(final_text)
        curr_predictions.append(Prediction(final_text, pred.start_logit, pred.end_logit))
    predictions += curr_predictions
    
    # add empty prediction
    if version_2_with_negative:
        predictions.append(Prediction('', null_val[1], null_val[2]))
    
    nbest_answers = []
    answer = None
    nbest = sorted(predictions,
                   key=lambda x: (x.start_logit + x.end_logit),
                   reverse=True)[:n_best_size]
    
    total_scores = []
    best_non_null_entry = None
    for entry in nbest:
        total_scores.append(entry.start_logit + entry.end_logit)
        if not best_non_null_entry and entry.text:
            best_non_null_entry = entry
    probs = _compute_softmax(total_scores)
    for (i, entry) in enumerate(nbest):
        output = collections.OrderedDict()
        output["text"] = entry.text
        output["probability"] = probs[i]
        output["start_logit"] = entry.start_logit
        output["end_logit"] = entry.end_logit
        nbest_answers.append(output)
    if version_2_with_negative:
        score_diff = null_val[0] - best_non_null_entry.start_logit - best_non_null_entry.end_logit
        if score_diff > null_score_diff_threshold:
            answer = ""
        else:
            answer = best_non_null_entry.text
    else:
        answer = nbest_answers[0]['text']
    
    return answer, nbest_answers


def get_answer_text(example, feature, pred):
    tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
    orig_doc_start = feature.token_to_orig_map[pred.start_index]
    orig_doc_end = feature.token_to_orig_map[pred.end_index]
    orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
    tok_text = " ".join(tok_tokens)

    # De-tokenize WordPieces that have been split off.
    tok_text = tok_text.replace(" ##", "")
    tok_text = tok_text.replace("##", "")

    # Clean whitespace
    tok_text = tok_text.strip()
    tok_text = " ".join(tok_text.split())
    orig_text = " ".join(orig_tokens)

    final_text = get_final_text(tok_text, orig_text)
    return final_text

def get_valid_prelim_predictions(start_indices, end_indices, feature, result):
    
    _PrelimPrediction = collections.namedtuple(
        "PrelimPrediction",
        ["start_index", "end_index", "start_logit", "end_logit"])
    prelim_predictions = []
    max_answer_length = 30
    for start_index in start_indices:
        for end_index in end_indices:
            if start_index >= len(feature.tokens):
                continue
            if end_index >= len(feature.tokens):
                continue
            if start_index not in feature.token_to_orig_map:
                continue
            if end_index not in feature.token_to_orig_map:
                continue
            if not feature.token_is_max_context.get(start_index, False):
                continue
            if end_index < start_index:
                continue
            length = end_index - start_index + 1
            if length > max_answer_length:
                continue
            prelim_predictions.append(
                _PrelimPrediction(
                    start_index=start_index,
                    end_index=end_index,
                    start_logit=result.start_logits[start_index],
                    end_logit=result.end_logits[end_index]))
    return prelim_predictions


def get_final_text(pred_text, orig_text, do_lower_case=True, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heruistic between
    # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.

    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                        orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indices(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indices = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indices.append(index_and_score[i][0])
    return best_indices


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs