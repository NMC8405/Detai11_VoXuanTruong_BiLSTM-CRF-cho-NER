# Báo cáo Đánh giá NER — BiLSTM+Softmax vs BiLSTM+CRF

## Model: bilstm_softmax
```
              precision    recall  f1-score   support

         art     0.0000    0.0000    0.0000        35
         eve     0.0000    0.0000    0.0000        24
         geo     0.7949    0.8687    0.8302      3809
         gpe     0.9441    0.9053    0.9243      1531
         nat     0.0000    0.0000    0.0000        14
         org     0.5464    0.6147    0.5785      2022
         per     0.6643    0.5720    0.6147      1612
         tim     0.8721    0.7227    0.7904      2048

   micro avg     0.7588    0.7517    0.7552     11095
   macro avg     0.4777    0.4604    0.4673     11095
weighted avg     0.7602    0.7517    0.7532     11095

```
**Boundary Errors:** 738

## Model: bilstm_crf
```
              precision    recall  f1-score   support

         art     0.0000    0.0000    0.0000        35
         eve     0.0000    0.0000    0.0000        24
         geo     0.8183    0.9068    0.8603      3809
         gpe     0.9644    0.9203    0.9418      1531
         nat     0.0000    0.0000    0.0000        14
         org     0.6882    0.6429    0.6648      2022
         per     0.7436    0.6836    0.7123      1612
         tim     0.8895    0.8096    0.8476      2048

   micro avg     0.8173    0.8042    0.8107     11095
   macro avg     0.5130    0.4954    0.5034     11095
weighted avg     0.8116    0.8042    0.8064     11095

```
**Boundary Errors:** 0
