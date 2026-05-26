# Báo cáo Đánh giá NER — BiLSTM+Softmax vs BiLSTM+CRF

## Model: bilstm_softmax
```
              precision    recall  f1-score   support

         art     0.0000    0.0000    0.0000        35
         eve     0.6429    0.3750    0.4737        24
         geo     0.8185    0.9149    0.8640      3809
         gpe     0.9650    0.9353    0.9499      1531
         nat     0.0000    0.0000    0.0000        14
         org     0.7339    0.6261    0.6757      2022
         per     0.7063    0.7667    0.7353      1612
         tim     0.8887    0.8423    0.8649      2048

   micro avg     0.8193    0.8250    0.8221     11095
   macro avg     0.5944    0.5576    0.5704     11095
weighted avg     0.8159    0.8250    0.8184     11095

```
**Boundary Errors:** 424

## Model: bilstm_crf
```
              precision    recall  f1-score   support

         art     0.0000    0.0000    0.0000        35
         eve     0.7500    0.3750    0.5000        24
         geo     0.8441    0.9113    0.8764      3809
         gpe     0.9696    0.9360    0.9525      1531
         nat     1.0000    0.1429    0.2500        14
         org     0.8200    0.6330    0.7145      2022
         per     0.7835    0.7543    0.7686      1612
         tim     0.8860    0.8579    0.8717      2048

   micro avg     0.8567    0.8263    0.8413     11095
   macro avg     0.7566    0.5763    0.6167     11095
weighted avg     0.8533    0.8263    0.8365     11095

```
**Boundary Errors:** 0
