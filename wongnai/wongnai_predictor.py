from allennlp.data import Instance
from allennlp.common.util import JsonDict
from allennlp.predictors.predictor import Predictor
from pythainlp import word_tokenize

@Predictor.register('wongnai_predictor')
class WongnaiCommentPredictor(Predictor):
    """"
    Predictor for comment
    """
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        comment = json_dict['comment']
        instance = self._dataset_reader.text_to_instance(tokenized_comment=comment)
        return instance