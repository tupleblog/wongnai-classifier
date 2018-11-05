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

text = """
มาไหว้พระ สะเดาะเคราะห์ปีชง เห็นวันนี้คนไม่เยอะเหมือนเสาร์-อาทิตย์ หลังจากทำบุญเสร็จ ก็เดินมาให้อาหารปลาที่ริมแม่น้ำท่าจีน แพก๋วยเตี๋ยวตั้งอยูฝั่งตรงกันข้าม มีเรือบริการข้ามฟากฟรี 
นึกสนุกอยากลองกินดูเพราะเห็นโฆษณารายการอาหารหลากหลาย พอมาถึงก็เลยสั่งก๋วยเตี๋ยวต้มยำไข่ ราคา 40 บาท หมูสะเต๊ะ 100 บาท มี 20 ไม้ ห้อยจ้อทอด 50 บาท น้ำมะพร้าวเป็นลูก รสชาติไม่ผ่านเลยค่ะ 
ก๋วยเตี๋ยวจืดมาก หมูสะเต๊ะน้ำจิ้มถั่วก็ค่อนข้างหยาบ ห้อยจ้อคุณภาพไม่ดีมีแต่แป้ง ผิดหวังค่ะวันนี้ แต่บรรกาศดีค่ะ ริมน้ำลมโชย วิวสวย
"""
prediction = wongnai_predictor.predict_json({"comment": word_tokenize(comment)})
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