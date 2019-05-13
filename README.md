# MSI_projekt

Repozytorium stworzone na potrzeby kursu Metody Sztucznej Inteligencji. Zawiera ono porównanie klasycznych metod uczenia maszynowego (kNN i SVC) z modelem głębokiej sieci neuronowej - ResNet.

## Przygotowanie zbioru danych

Dane należy pobrać ze strony Kaggle: https://www.kaggle.com/skooch/ddsm-mammography/home 

W celu wstępnego przetworzenia danych została przygotowana seria skryptów: 
* `ddsm_generator.py`: podzielenie zbioru danych na klasy z wykorzystaniem generatorów 
* `ddsm_split.py`: podzielenie zbioru danych na podzbiory: treningowy, walidacyjny i testowy
* `ddsm_utils.py`: ekstrakcja obrazów z plików TFRecords

Przykładowe wywołanie skryptu: `ddsm_generator.py`
```
python3 ddsm_generator.py 
  --ddsm_data_dir 'path_to_directory_with_raw_ddsm_data' 
  --outdir 'path_to_dir_that_will_contain_ddsm_generator_results' 
  --file_extension '.tiff' 
  --multiclass True 
  --just_preview True
``` 

## ResNet

W celu przeprowadzenia treningu sieci ResNet przygotowano skrypty: `train_resnet.py`, `experiment.py`. Dzięki temu możliwe bylo sprawdzenie wielu wartości hiperparametrów i wybranie najlepszych z nich.

Przykładowe wywołanie skryptu `experiment.py`
```
python3 experiment.py
  -- log_dir 'log_dir'
  -- epoch 100
  -- seed 1234,
```

Przykładowe wywołanie skryptu `train_resnet.py`
```
python3 train_resnet.py 
  -- data_dir 'path_to_dir'
  -- model_id 'model'
  -- batch_size 8
  -- rate 0.5
  -- target_size '(299, 299)'
  -- lr 0.001
  -- epochs 25
  -- seed 1234
  -- log_dir './logs'
```

Nauczony model poddano ewaluacji w skrypcie `resnet_metrics.py`, dzięki czemu uzyskano wartości następujących metryk: 
  * F-score
  * Accuracy
  * Precision
  * Recall
  
Przykładowe wywołanie skryptu `resnet_metrics.py`
```
python3 resnet_metrics.py
```

### Stratyfikowana walidacja krzyżowa
Po znalezieniu najlepszego zestawu parametrów z wykorzystaniem skryptu `experiment.py` została przeprowadzona stratyfikowana walidacja krzyżowa. Kod znajduje się w pliku `cross_validation.py`. Wizualizacja tego procesu została zaimplementowana w skrypcie `cross_validation_plots.py`.

Przykładowe wywołanie skryptu `cross_validation.py`
```
python3 cross_validation.py
```

Przykładowe wywołanie skryptu `cross_validation_plots.py`
```
python3 cross_validation_plots.py
```
  

## k-Nearest Neighbors i Support Vector Machine
`svm_knn_metrics.py` to zbiorczy skrypt zawierający zarówno walidację krzyżową jak i wyliczenie wyżej podanych metryk.

Przykładowe wywołanie skryptu `svm_knn_metrics.py` 
```
python3 svm_knn_metrics.py
```

### PCA
Aby dane mogły być poprawnie czytane przez algorytmy kNN i SVC zredukowano ilość cech do 2 wykorzystując skrypt `pca.py`. Wizualizacja rezultatu tej operacji została zaimplementowana w skrypcie `pca_visualization.py`.


Przykładowe wywołanie skryptu `pca.py`
```
python3 pca.py
```

Przykładowe wywołanie skryptu `pca_visualization.py`
```
python3 pca_visualizaion.py
```

## Eksperymenty
Testy statystyczne (T-student) zostały przeprowadzone w celu zweryfikowania następującej hipotezy: klasyfikator ResNet znacznie różni się od klasycznych metod uczenia się maszynowego. Zastosowano pary algorytmów według następującego schematu: algorytm najwyższej dokładności - inny algorytm (ResNet - kNN, ResNet - SVC). Dodatkowo sprawdzono parę kNN-SVC. Testy zostały zaimplementowane w skrypcie `statistics.py`.

Przykładowe wywołanie skryptu `statistics.py`
```
python3 statistics.py
```
