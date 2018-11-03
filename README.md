# Wongnai comment classifier

We use AllenNLP to classify comment from `Wongnai.com` (Review Rating Prediction in Thai language) using 
[thai2vec](https://github.com/cstorm125/thai2vec) word vectors and simple Bidirectional LSTM model.


## Train the model

```bash
allennlp train experiments/wongnai.json -s output --include-package wongnai
```

## Demo and blog post

Demo and [blog post](http://tupleblog.github.io) will be coming soon!

Here is the script to run the demo

```bash
python -m allennlp.service.server_simple \
    --archive-path model.tar.gz \
    --predictor wongnai_predictor \
    --include-package wongnai \
    --field-name comment \
    --static-dir static_html
```