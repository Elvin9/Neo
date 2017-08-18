# Neo
![alt text](https://img.shields.io/travis/USER/REPO.svg)
A W.I.P toy machine learning framework

![alt text](http://download.gamezone.com/uploads/image/data/1201507/article_post_width_Thomas-Anderson-aka-Neo-the-Matrix-1024x516.jpg)

## How To Use

### Sequential model
>Layout

```python
model = SequentialModel(learning_rate, MSELoss())

model.add_layer(Linear(2, 2, bias=False, parameter_update=SGD()))
model.add_layer(Sigmoid())
model.add_layer(Linear(2, 1, bias=False, parameter_update=SGD()))
model.add_layer(Sigmoid())
```
each layer will be sequentially propagated according to the specified layout

>Training
```python
model.train(input_batch, output_batch, batch_size=10, error=True)
```
In order to train the model, an input batch and output batch must be specified.<br>
A batch size larger than one will become a mini batch, while the batch size is online by default.

Upon completion, a list of errors during the training will be returned, for you to process later (if specified by the error parameter).
