# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# Source: https://github.com/sdascoli/odeformer

import numpy as np
import torch
from typing import Union


class FunctionEnvironment(object):

    def word_to_idx(self, words, float_input=True):
        """
        Maps from actual words (e.g. 'add' or '1.43') to ids. Ids can be floats if we use two-hot encoding.
        `words`: List[List[str]], e.g. from samples["tree_encoded"]
        """
        if float_input:
            return [
                [
                    torch.LongTensor([self.float_word2id[dim] for dim in point])
                    for point in seq
                ]
                for seq in words
            ]
        elif self.equation_encoder.constant_encoder is not None:
            id_offset = max(self.equation_word2id.values())
            output = []
            for eq in words:
                lst = []
                for w in eq:
                    try:
                        lst.append(
                            float(w)
                            + id_offset
                            - self.equation_encoder.constant_encoder.min
                        )
                    except ValueError:
                        lst.append(self.equation_word2id[w])
                output.append(torch.DoubleTensor(lst))
            return output
        else:
            return [
                torch.LongTensor([self.equation_word2id[w] for w in eq]) for eq in words
            ]

    def ids_to_two_hot(self, ids: torch.Tensor, support_size: int):
        """
        Take a list of int or float 'ids' and convert them to 2-hot representation.
        For int ids, this is equivalent to 1-hot representation.
        """
        id_left = ids.floor()
        id_right = id_left + 1
        prob_left = 1 - (ids - id_left)
        prob_right = 1 - prob_left
        logits = torch.zeros(ids.shape[0], ids.shape[1], support_size).type_as(ids)
        return (
            logits.scatter_(
                dim=2, index=id_left.long().unsqueeze(-1), src=prob_left.unsqueeze(-1)
            )
            .scatter_(
                dim=2, index=id_right.long().unsqueeze(-1), src=prob_right.unsqueeze(-1)
            )
            .squeeze(1)
        )

    def topk_decode_two_hot(
        self,
        logits: torch.Tensor,
        topk_idx: torch.Tensor,
        unfinished_sents: Union[None, torch.Tensor],
        apply_softmax: bool = True,
    ):
        """
        Take logits and topk indices, whenever a topk index refers to the indices that represent two-hot encoded
        constants, decode the selected constant. The decoding process works as follows:
        1. Apply softmax across all logits that represent constants.
        2. Among the two top-k neighboring indices, select the one whose corresponding probability is larger.
        3. Decode as topk * prob_{topk} + neighbor * prob_{neighbor}.
        This function is necessary as topk returns a single index but we need two indices to represent two-hot encoded
        constants. This problem is solved by allowing for "float" ids, e.g. we combine the two indices that represent a
        constant into a single float. The float is still in "id-space", i.e., the returned float ids do not directly
        correspond to the encoded float but to the float plus an offset that accounts for the remaining non-two-hot
        tokens (e.g. tokens for operators and variables).

        `logits`:
            Tensor, (bs, vocab size). Batch of logits.
        `topk_idx`:
            Tensor, (bs,). Batch of selected indices for decoding.
        `unfinished_sents`:
            Tensor, (bs, ). Batch of indices that indicate batch elements that are part of a not-yet-finished sequence.
            If a sequence is already finished, we do not have to do any constant decoding, so this is an opportunity for
            a short-cut.

        Returns:
        `topk_idx`:
            Tensor, (bs,).
            The original tensor with topk indices that correspond to two-hot constants converted to float indices.
        `constants_mask`:
            Tensor, (bs,).
            A boolean tensor of same shape as `top_idx` that indicates if the corresponding element in `top_idx` is a
            decoded constant.

        Caution: topk_idx.dtype changes in place from int64 to torch.float32 or torch.double.
        """
        id_offset = max(self.equation_word2id.values()) + 1
        constants_mask = topk_idx.squeeze() >= id_offset
        if constants_mask is not None:
            constants_mask = constants_mask.to(torch.int) * unfinished_sents
            constants_mask = constants_mask.to(torch.bool)
        if not constants_mask.any():
            return (topk_idx, constants_mask)
        if apply_softmax:
            probs = torch.nn.functional.softmax(
                logits[constants_mask, id_offset:], dim=1
            )
        else:
            probs = logits[constants_mask, :]
        neg_infs = torch.ones((probs.shape[0], 1), device=logits.device).fill_(
            -torch.inf
        )
        probs = torch.cat([neg_infs, probs, neg_infs], dim=1)
        topk_idx[constants_mask] += 1
        left_neighbor = (topk_idx[constants_mask] - 1).reshape(-1, 1) - id_offset
        selected = topk_idx[constants_mask].reshape(-1, 1) - id_offset
        right_neighbor = (topk_idx[constants_mask] + 1).reshape(-1, 1) - id_offset
        left_prob = torch.gather(input=probs, dim=1, index=left_neighbor)
        selected_prob = torch.gather(input=probs, dim=1, index=selected)
        right_prob = torch.gather(input=probs, dim=1, index=right_neighbor)
        take_left = left_prob > right_prob
        take_right = ~take_left
        best_neighbor = left_neighbor * take_left + right_neighbor * take_right
        neighbor_prob = torch.gather(input=probs, dim=1, index=best_neighbor)
        selected_value = selected + self.equation_encoder.constant_encoder.min
        neighbor_value = best_neighbor + self.equation_encoder.constant_encoder.min
        selected_value -= 1
        neighbor_value -= 1
        value = torch.squeeze(
            selected_value * selected_prob + neighbor_value * neighbor_prob
        ).to(torch.double)
        id_value = value + id_offset - self.equation_encoder.constant_encoder.min
        topk_idx = topk_idx.to(id_value.dtype)
        topk_idx[constants_mask] = id_value
        return (topk_idx, constants_mask)

    def word_to_infix(self, words, is_float=True, str_array=True):
        if is_float:
            m = self.float_encoder.decode(words)
            if m is None:
                return None
            if str_array:
                return np.array2string(np.array(m))
            else:
                return np.array(m)
        else:
            m = self.equation_encoder.decode(words)
            if m is None:
                return None
            if str_array:
                return m.infix()
            else:
                return m

    def wrap_equation_floats(self, tree, constants):
        prefix = tree.prefix().split(",")
        j = 0
        for i, elem in enumerate(prefix):
            if elem.startswith("CONSTANT"):
                prefix[i] = str(constants[j])
                j += 1
        assert j == len(constants), "all constants were not fitted"
        assert "CONSTANT" not in prefix, "tree {} got constant after wrapper {}".format(
            tree, constants
        )
        tree_with_constants = self.word_to_infix(
            prefix, is_float=False, str_array=False
        )
        return tree_with_constants

    def idx_to_infix(
        self,
        lst,
        is_float=True,
        str_array=True,
        is_two_hot: Union[None, torch.BoolTensor] = None,
    ):
        if is_float:
            idx_to_words = [self.float_id2word[int(i)] for i in lst]
        else:
            id_offset = max(self.equation_word2id.values())
            idx_to_words = []
            for term_i, term in enumerate(lst):
                if self.params.use_two_hot:
                    if is_two_hot[term_i]:
                        idx_to_words.append(
                            f"{term - id_offset + self.equation_encoder.constant_encoder.min:+}"
                        )
                else:
                    idx_to_words.append(self.equation_id2word[int(term)])
        return self.word_to_infix(idx_to_words, is_float, str_array)
