### train資料集和flickr30k_images資料集的內容太大了，所以沒有放到github上  

## Quiz1  
#### Sentiment Analysis資料集: sentiment_data.csv  
#### Image Classification資料集: train資料夾   
#### Image Captioning資料集: flickr30k_images資料夾、result.csv  

## Quiz2  
RNN、LSTM model 都在rnn_sentiment_analysis.ipynb裡  
兩者結果比較(**參數皆相同**):  
RNN test:   
Test Loss: 0.5336  
Test Accuracy: 0.8126  
Test Precision: 0.8072  
Test Recall: 0.7950  
Test F1: 0.8001  

LSTM test:   
Test Loss: 0.4832  
Test Accuracy: 0.8232  
Test Precision: 0.8215  
Test Recall: 0.8045  
Test F1: 0.8109  

## Quiz3  
使用model：  google/vit-base-patch16-224、ResNet50，在resnet18_lstm_flickr30k_faster.ipynb裡  
兩者結果比較(參數皆相同):  
Vit test:   
Test Loss: 0.0126  
Test Accuracy: 0.9956  
Test Precision: 0.9956  
Test Recall: 0.9956  
Test F1: 0.9956  
Test auc: 0.9999  

Resnet50 test:   
Test Loss: 0.0256  
Test Accuracy: 0.9896  
Test Precision: 0.9896  
Test Recall: 0.9896  
Test F1: 0.9896  
Test auc: 0.9996  

## Quiz4  
使用model： resnet18 + lstm  
BLEU-4：  0.1648  
輸出結果：  放在resnet18_lstm_test_captions.csv  
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>image_name</th>
      <th>generated_caption</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000092795.jpg</td>
      <td>a woman in a pink shirt is walking on a sidewa...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1000268201.jpg</td>
      <td>a man in a blue shirt is working on a machine .</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10010052.jpg</td>
      <td>a woman in a red shirt and black pants is walk...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1001465944.jpg</td>
      <td>a man in a black shirt and jeans is walking do...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100207720.jpg</td>
      <td>a woman in a black shirt is sitting on a couch .</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1009692167.jpg</td>
      <td>a man in a blue shirt is standing on a sidewal...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1017675163.jpg</td>
      <td>a man in a blue shirt is doing a kick flip on ...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1022454332.jpg</td>
      <td>a man in a blue shirt is walking on a dock .</td>
    </tr>
    <tr>
      <th>8</th>
      <td>102351840.jpg</td>
      <td>a man in a blue shirt and black pants is walki...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>102860573.jpg</td>
      <td>a man in a white shirt is sitting on a bench .</td>
    </tr>
  </tbody>
</table>
</div>  

效果並不好  