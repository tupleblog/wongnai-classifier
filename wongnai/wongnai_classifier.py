import torch
import numpy as np
import torch.nn.functional as F
from typing import Iterator, List, Dict, Optional

from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.modules import FeedForward
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy


@Model.register("wongnai_classifier")
class WongnaiCommentClassifier(Model):
    """
    Model to classify Wongnai comments
    """
    def __init__(self, 
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 comment_encoder: Seq2VecEncoder,
                 classifier_feedforward: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(WongnaiCommentClassifier, self).__init__(vocab, regularizer)
        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.comment_encoder = comment_encoder
        self.classifier_feedforward = classifier_feedforward
        self.metrics = {
                "accuracy": CategoricalAccuracy(),
                "accuracy3": CategoricalAccuracy(top_k=3)
        }
        self.loss = torch.nn.CrossEntropyLoss()
        initializer(self)
    
    def forward(self, 
                comment: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        
        embedded_comment = self.text_field_embedder(comment)
        comment_mask = get_text_field_mask(comment)
        encoded_comment = self.comment_encoder(embedded_comment, comment_mask)
        logits = self.classifier_feedforward(encoded_comment)
        class_probabilities = F.softmax(logits, dim=-1)
        argmax_indices = np.argmax(class_probabilities.cpu().data.numpy(), axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace="labels") for x in argmax_indices]
        output_dict = {
            'logits': logits, 
            'class_probabilities': class_probabilities,
            'predicted_label': labels
        }
        if label is not None:
            loss = self.loss(logits, label)
            for metric in self.metrics.values():
                metric(logits, label)
            output_dict["loss"] = loss

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}