## We recommend using Kaggle GPU environments to run our code.
All instructions are written assuming you are using the notebooks on Kaggle.

### Ques 1 : MLP

- Run the cells by sequence
- If you want to change the parameters, you need to change the below code parameters in the notebook.
```
layer_sizes = [784, 128, 10]
activations = ['leaky_relu','softmax']
best_mlp, train_loss, train_acc, val_loss, val_acc = train_mlp(
    x_train, y_train, x_val, y_val,
    layer_sizes, activations,
    epochs=200,dropout_rate=0.2, lr=0.001,patience=5,batch_size=32
)
```

```
layer_sizes = [784, 400, 250, 100, 10]
activations = ['relu', 'relu', 'relu', 'softmax']
```

Available activations : relu,gelu,leaky_relu,tanh

### Ques 2 : CNN


- Run the cells by sequence
- If you want to change the parameters, you need to change the below code parameters in the notebook.

```
cnn_model = train_cnn_extractor(epochs=10,
            pool_method='max',
            weight_init='xavier',
            conv_dims=[32, 64, 128, 256,512],
            n6=1024,)
```
Available pool_method : max, avg, global

Available weight_init : xavier, he, random

n6 parameter: Hidden Layer size used in Fully Connected Layer of the CNN architecture.

- If you change the value of n6, you also need to change the value of hidden layer ie 1024 here (last cell). (to use the customMLP from question 1 for Q2.d)

Also if you do not need the same parameters as the FCC layers of CNN, you can also use multiple layers too like we did in Q1.

```
layer_sizes = [features_train.shape[1], 1024,10]
activations = ['relu', 'softmax']

print("Training custom MLP on CNN features...")
mlp_model, train_losses, train_accs, val_losses, val_accs = train_mlp(
    features_train, y_train_oh,
    features_val, y_val_oh,
    layer_sizes, activations,
    epochs=200, lr=0.001, dropout_rate=0.2, patience=5, batch_size=32
)
```




### Incase of any issue while running our code, please feel free to contact any of our team members.
