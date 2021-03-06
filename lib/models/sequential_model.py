import gnumpy as gp
import time


class SequentialModel(object):
    def __init__(self, learning_rate, loss=None):
        self.layers = []
        self.loss = loss
        self.rate = learning_rate

    def set_loss(self, loss):
        self.loss = loss

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, inp):
        output = inp[:]
        for layer in self.layers:
            output = layer.get_output(output)
        return output

    def _update_weights(self, output, target):

        # Get error from loss_functions function
        output_grad = None
        if self.loss:
            self.loss.get_output(output, target)
            output_grad = self.loss.get_input_gradient(target)
        else:
            raise Exception('[*] Loss function was not specified')

        # Back propagate layers
        for layer in self.layers[::-1]:
            prev_output_grad = output_grad
            output_grad = layer.get_input_gradient(output_grad)
            layer.update_parameters(prev_output_grad, self.rate)

    def train(self, input_batch, target_batch, batch_size=5, error=False):

        current_input = []
        current_prediction = []

        # Return list of errors, sorted by time (stepped by chunk)
        if error:
            error_lst = []

        # Train
        for chunk_cursor in range(0, input_batch.shape[1], batch_size):
            current_input = gp.garray(input_batch[:, chunk_cursor:chunk_cursor + batch_size])
            current_target = gp.garray(target_batch[:, chunk_cursor:chunk_cursor + batch_size])

            output = self.forward(current_input)
            self._update_weights(output, current_target)

            if error:
                error_output = self.loss.get_output(output, current_target)
                for err in error_output:
                    error_lst.append(float(err))

        if error:
            return error_lst


class SequentialConvolutionalModel(SequentialModel):

    def train(self, input_batch, target_batch, batch_size=5, error=False):

        start_t = time.time()
        # Return list of errors, sorted by time (stepped by chunk)
        if error:
            error_lst = []

        # Train
        for i in range(len(input_batch)):

            print("[-] {}: {}".format(i, str(time.time() - start_t)))
            start_t = time.time()

            output = self.forward(gp.garray(input_batch[i]))
            target = gp.garray(target_batch[i])
            self._update_weights(output, target)

            if error:
                error_output = self.loss.get_output(output, target)
                for err in error_output:
                    print(str(err) + "\n")
                    error_lst.append(float(err))

        if error:
            return error_lst