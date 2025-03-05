from typing import Dict, List, Optional, Pattern, Set, Tuple, Union

import regex as re
import unicodedata

# These patterns are taken from:
# gpt2: https://github.com/openai/gpt-2/blob/master/src/encoder.py#L53
# gpt4: https://github.com/karpathy/minbpe/blob/master/minbpe/gpt4.py#L48
GPT_2_PATTERN: Pattern[str] = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT_4_PATTERN: Pattern[str] = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


def get_stats(ids: List[int], counts: Optional[Dict[int, int]]=None) -> Dict[int, int]:
    """
    Iterates over the list of ids and counts the occurrences of each pair of ids.
    If counts is provided, it will be updated with the counts of the ids.

    creates dictionary of pairs of ids and their frequency rates

    Args:
        ids (List[int]): List of ids to count (Your training/inference text, converted to ids)
        counts (Optional[Dict[int, int]]): Dictionary to update with counts.
    """
    # TODO: Implement this function

    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts = counts.get(pair, 0) + 1
    
    """loop through pairs"""

    return counts

def merge(ids: List[int], pair: Tuple[int, int], idx: int) -> List[int]:
    """
    Returns a new list of ids with the pair of ids now merged into one, using the 
    given idx as the new id.
    merges just 2 ids(charecters) at a time

    Args:
        ids (List[int]): List of ids to merge.
        pair (Tuple[int, int]): Pair of ids to merge.
        idx (int): New id for the merged pair.
    """
    newids: List[int] = []
    i = 0

    while (i < len(ids)):
        if ids[i] == pair[0] and i < len(ids) + 1 and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
        
    # TODO: Implement this function
    return newids


def replace_control_characters(token: str) -> str:
    """
    Replaces control characters in the token with their unicode escape sequences.

    Args:
        token (str): The token to process.

    Returns:
        str: The token with control characters replaced.
    """
    chars: List[str] = []
    for char in token:
        # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
        if unicodedata.category(char)[0] != "C":
            chars.append(char)
        else:
            chars.append(f"\\u{ord(char):04x}") # escape the character
    return "".join(chars)



def render_bytes(token: bytes) -> str:
    """
    Converts a token into a string representation, replacing control characters with their unicode escape sequences.
    
    Args:
        token (bytes): The token to process.
    """
    out = token.decode("utf-8", errors="replace")
    out = replace_control_characters(out)
    return out


class MyTokenizer:
    """Our custom tokenizer implementation"""

    def __init__(self, pattern: Optional[Pattern[str]]=None):
        self.merges: Dict[Tuple[int, int], int] = {}
        self.pattern: Pattern[str] = pattern or GPT_2_PATTERN
        self.compiled_pattern = re.compile(self.pattern)
        self.vocab: Dict[int, bytes] = {}

    def train(self, text: str, desired_vocab_size: int):
        """
        Trains the tokenizer on the given text, using this tokenizers pattern to split the text into chunks
        and creating desired_vocab_size - 256 merges.

        Args:
            text (str): The text to train on.
            desired_vocab_size (int): The desired vocabulary size.
        """
        num_merges = desired_vocab_size - 256

        # TODO: Implement the training logic
        pass

    def decode(self, ids: List[int]) -> str:
        """
        Decodes a list of ids into a string.
        
        Args:
            ids (List[int]): The list of ids to decode.
        """
        # TODO: Implement the decode method
        pass

    def encode(self, text: str) -> List[int]:
        """
        Encodes the given text into a list of ids.

        Args:
            text (str): The text to encode.
        """
        # TODO: Implement the encode method
        pass

    def _build_vocab(self) -> Dict[int, bytes]:
        """
        Looks at the first 256 bytes ++ all of our custom merges and builds a vocabulary
        mapping from the token id to the bytes representation. 
        """
        vocab: Dict[int, bytes] = {idx: bytes([idx]) for idx in range(256)} # each byte is its own token
        for (a, b), idx in self.merges.items():
            vocab[idx] = vocab[a] + vocab[b]
        return vocab

    def save(self, file_prefix: str):
        """
        Saves two files, file_prefix.vocab and file_prefix.model. The .model file stores
        the important data, the .vocab is just for human readability.

        Args:
            file_prefix (str): Prefix for the files to save.
        """
        model_file = file_prefix + ".model"
        with open(model_file, "w") as file:
            file.write("minbpe v1\n")
            file.write(f"{self.pattern}\n")

            # 0 special tokens are handled in this tokenizer
            file.write("0\n")
            for idx1, idx2 in self.merges:
                file.write(f"{idx1} {idx2}\n")

        vocab_file = file_prefix + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as file:
            for idx, token in self.vocab.items():
                rendered_token = render_bytes(token)
                if idx in inverted_merges:
                    idx0, idx1 = inverted_merges[idx]
                    rendered_token1 = render_bytes(self.vocab[idx0])
                    rendered_token2 = render_bytes(self.vocab[idx1])
                    file.write(f"[{rendered_token1}][{rendered_token2}] -> [{rendered_token}] {idx}\n")
                else:
                    # one of the first 256 tokens
                    file.write(f"[{rendered_token}] {idx}\n")

    def load(self, model_file: str):
        """
        Loads the model from the given file. The file should be in the format
        produced by the save method.

        Args:
            model_file (str): Path to the model file.
        """
        merges: Dict[Tuple[int, int], int] = {}
        idx = 256
        with open(model_file, "r", encoding="utf-8") as file:
            version = file.readline().strip()
            assert version == "minbpe v1", f"Unsupported version: {version}"
            self.pattern = file.readline().strip()
            num_special = int(file.readline().strip())
            for line in file:
                idx1, idx2 = map(int, line.strip().split())
                merges[(idx1, idx2)] = idx
                idx += 1
        
        self.merges = merges
        self.vocab = self._build_vocab()