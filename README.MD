# Trafik Tabelası Yer Tespiti ve Sınıflandırma Projesi
- Bu proje Bandırma Onyedi Eylül Üniversitesi YBS bölümü Yapay Zeka dersi kapsamında hazırlanmıştır.
## Model Eğitme Süreci
- YOLO ve CNN modelini eğitme sürecine [buradan](model-training-process.md) ulaşabilirsiniz.

## Modeller
- Üretilen YOLO modeline [buradan](models/best.pt) , CNN modeline [buradan](models/gtsrb_cnn_model.pth) ulaşabilirsiniz. 
- Eğitilen modellerin deploy edilme sürecine [buradan](deployment-process.md) ulaşabilirsiniz.

## Model Grafikleri
- Üretilen modellerin grafiklerine ulaşmak için [tıklayın](graphs/).

## Modelin Video Testi
[![Gerçek Zamanlı Video Test](https://img.youtube.com/vi/25ZkYhnrZZY/0.jpg)](https://www.youtube.com/watch?v=25ZkYhnrZZY)

## Deploy
- Modeli canlı olarak test etmek için [tıklayın](https://trafficsigndetectionandclassification-pkbcbbjxdt7ds5fknbnshk.streamlit.app/).
- Modeli deploy etme sürecine ve bu süreçte çıkan hatalara [buradan](deployment-process.md) ulaşabilirsiniz.

## Görev Listesi
- [X] Modeller'in github veya huggingface ortamına yüklenmesi.
- [X] Modelin streamlit üzerinden deploy edilmesi ve sürecin döküman haline getirilmesi.
- [X] Model eğitim sürecinin döküman haline getirilmesi.
- [X] Modelin gerçek zamanlı video üzerinden test sonuçlarının eklenmesi.
- [X] Grafiklerin eklenmesi.
- [ ] Modelin test görsellerinin eklenmesi.
- [ ] Modelin görsel test sonuçlarının eklenmesi.