import torch
import json
from typing import Iterator, List, Dict, Optional
from pythainlp import word_tokenize

from allennlp.data.fields import TextField, LabelField
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.common.file_utils import cached_path


@DatasetReader.register("wongnai_reader")
class WongnaiDatasetReader(DatasetReader):
    """
    Wongnai dataset reader
    """
    def __init__(self, 
                 token_indexers: Dict[str, TokenIndexer] = None, 
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
    
    def _read(self, file_path: str) -> Iterator[Instance]:
        label_dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5'}
        with open(cached_path(file_path), "r") as data_file:
            for line in data_file:
                line = line.strip("\n")
                if not line:
                    continue
                comment_json = json.loads(line)
                comment = comment_json['text']
                tokenized_comment = word_tokenize(comment)
                label = label_dict.get(int(comment_json['label']), '0')
                yield self.text_to_instance(tokenized_comment, label)
    
    def text_to_instance(self, tokenized_comment: List[str], label: int = None) -> Instance:
        comment = [Token(s) for s in tokenized_comment]
        comment_field = TextField(comment, self._token_indexers)
        fields = {'comment': comment_field}
        if label is not None:
            fields['label'] = LabelField(label)
        return Instance(fields)