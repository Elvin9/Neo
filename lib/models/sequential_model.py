import gnumpy as gp


class SequentialModel(object):

    def __init__(self, loss=None, learning_rate=1):
        self.layers = []
        self.loss = loss
        self.learning_rate = learning_rate

    def set_loss(self, loss):
        self.loss = loss

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, inp):
        output = inp[:]
        for layer in self.layers:
            output = layer.get_output(output)
        return output

    def __update_weights(self, output, prediction):

        # Get error from loss function
        output_grad = None
        if self.loss:
            self.loss.get_output(output, prediction)
            output_grad = self.loss.get_input_gradient(prediction)
        else:
            raise Exception('[*] Loss function was not specified')

        # Back propagate layers
        for layer in self.layers[::-1]:
            layer.update_parameters(output_grad, self.learning_rate)
            output_grad = layer.get_input_gradient(output_grad)

    def train(self, input_batch, prediction_batch, batch_size=1, error=False):
        # Cast matrices to gnumpy types
        input_batch = gp.garray(input_batch)
        prediction_batch = gp.garray(prediction_batch)

        # Return list of errors, sorted by time (stepped by chunk)
        if error:
            error_lst = []

        # Train
        for chunk_cursor in range(input_batch.shape[1], batch_size):
            input_chunk = input_batch[:, chunk_cursor:batch_size]
            prediction_batch = prediction_batch[:, chunk_cursor:batch_size]
            
            # Forward + backward prop
            output = self.forward(input_chunk)
            self.__update_weights(output, prediction_batch)

            if error:
                error_lst.append(self.loss.get_output(output))
