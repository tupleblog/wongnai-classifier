# Wongnai comment classifier

We use AllenNLP to classify comment from `Wongnai.com` (Review Rating Prediction in Thai language) using 
[thai2vec](https://github.com/cstorm125/thai2vec) word vectors and simple Bidirectional LSTM model.


## Train the model

```bash
allennlp train experiments/wongnai.json -s output --include-package wongnai
```


## Prediction

```python
from allennlp.models.archival import load_archive
from allennlp.common.file_utils import cached_path
from allennlp.predictors.predictor import Predictor
from wongnai.wongnai_reader import WongnaiDatasetReader
from wongnai.wongnai_classifier import WongnaiCommentClassifier
from wongnai.wongnai_predictor import WongnaiCommentPredictor
from pythainlp import word_tokenize

archive = load_archive(cached_path('https://s3-us-west-2.amazonaws.com/thai-corpus/wongnai_model.tar.gz')) # load trained model
wongnai_predictor = Predictor.from_archive(archive, 'wongnai_predictor')

comment = """
มาไหว้พระ สะเดาะเคราะห์ปีชง เห็นวันนี้คนไม่เยอะเหมือนเสาร์-อาทิตย์ หลังจากทำบุญเสร็จ ก็เดินมาให้อาหารปลาที่ริมแม่น้ำท่าจีน แพก๋วยเตี๋ยวตั้งอยูฝั่งตรงกันข้าม มีเรือบริการข้ามฟากฟรี 
นึกสนุกอยากลองกินดูเพราะเห็นโฆษณารายการอาหารหลากหลาย พอมาถึงก็เลยสั่งก๋วยเตี๋ยวต้มยำไข่ ราคา 40 บาท หมูสะเต๊ะ 100 บาท มี 20 ไม้ ห้อยจ้อทอด 50 บาท น้ำมะพร้าวเป็นลูก รสชาติไม่ผ่านเลยค่ะ 
ก๋วยเตี๋ยวจืดมาก หมูสะเต๊ะน้ำจิ้มถั่วก็ค่อนข้างหยาบ ห้อยจ้อคุณภาพไม่ดีมีแต่แป้ง ผิดหวังค่ะวันนี้ แต่บรรกาศดีค่ะ ริมน้ำลมโชย วิวสวย
"""
instance = wongnai_predictor._dataset_reader.text_to_instance(word_tokenize(comment))
prediction = wongnai_predictor.predict_instance(instance)
# prediction = wongnai_predictor.predict_json({"comment": word_tokenize(comment)}) alternative
all_labels = wongnai_predictor._model.vocab.get_index_to_token_vocabulary('labels')
prediction['all_labels'] = [all_labels[i] for i in range(len(all_labels))]
print(prediction)


{
    'logits': [0.6556835174560547, 0.43491148948669434, -2.236508369445801, 1.798933982849121, -1.5035953521728516],
    'class_probabilities': [0.19570578634738922, 0.156936377286911, 0.010852772742509842, 0.6139189600944519, 0.022586075589060783],
    'predicted_label': '2',
    'all_labels': ['4', '3', '5', '2', '1']
}
```


## Demo and blog post

Demo and [blog post](http://tupleblog.github.io) will be coming soon!

Here is the WIP script to run the demo

```bash
wget https://s3-us-west-2.amazonaws.com/thai-corpus/wongnai_model.tar.gz
python -m allennlp.service.server_simple \
    --archive-path wongnai_model.tar.gz \
    --predictor wongnai_predictor \
    --include-package wongnai \
    --field-name comment \
    --static-dir static_html
```