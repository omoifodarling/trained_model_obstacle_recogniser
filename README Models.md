
# Trained Models

These models have slightly different implementations depending on the layers, dropout, batch normalization or even the use of sequential implementation. The non-sequential implementation I named basic was trained in about a 1000 epochs and was pretty good at training and understandably so. The printouts I added as a submodule does show their performance at training and evaluation. However, the best performing was the VGG model which is quite a complex model from the application, though not more complex than ResNet which surprisingly did not outperform it.

The models by name are:

Basic - same as the terse save the sequential implementation was not used. It did pretty well at training with 89.54% at training and 84.5% at prediction with 50,000 and 10,000 images respectively. It sure was better than expected though it took quite a lot of time about an hour and half on the GPU and did not go beyond 60 epochs on a laptop after about 2 weeks.

The Terse model was same as basic but used sequential implementation of the layers. This quickly converged and was run in less than 200 epochs. It did considerably better at prediction than it did at training with about 82% and 74% respectively. It was the worst performing model though the training parameters and impelemntatio were not same for all models

Batch Normalized Model. This was batch normalized before the fully connected layer. It was only outperformed by VGG9 and Basic Model at training recording 87.39%. In its prediction it had only 16.4% error in 10,000 images tested. 

Dropout Model. This was run through 400 epochs. It was the second worst performing model recoridng 17.7% error at training and 16.11% error at predictions.

ResNet Model. In only 75 epochs or iterations of training, it recorded 85.2% at training. However, it did relatively well with prediction turning out as the second best performing at prediction with errot of 14.48% in the 10,000 images tested.

VGG9 Model. This was the best performing at prediction and training recording 85.6% and 91.18% respectively. This was the model used in most of the test carried out with images downloaded from the web or those that did not belong to the University of Toronto CIFAR-10 data repository.

## How to Use the Model


```python
python model_eval.py --image_dir images#the name of the dir to find the images is the "images"
```


```python
%run model_eval.py --image_dir images# if using QtConsole -the name of the dir to find the images is the "images"
```

It is assumed the user is in the directory where the application is located. Otherwise the path should match accordingly.
