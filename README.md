# Neo
A W.I.P toy machine learning framework

![alt text](http://download.gamezone.com/uploads/image/data/1201507/article_post_width_Thomas-Anderson-aka-Neo-the-Matrix-1024x516.jpg)

## How To Use

>Sequential model
```python
model = SequentialModel(learning_rate, MSELoss())

model.add_layer(Linear(2, 2, bias=False, parameter_update=SGD()))
model.add_layer(Sigmoid())
model.add_layer(Linear(2, 1, bias=False, parameter_update=SGD()))
model.add_layer(Sigmoid())
```
each layer will be sequentially propagated according to the specified layout
