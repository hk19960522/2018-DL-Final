
class Config:
    def __init__(self):
        self.pedestrian_num = 20
        self.hidden_size = 128

        # input
        self.input_frame = 5
        self.input_size = 5  # * self.input_frame
        self.n_layers = 1

        # target
        self.target_frame = 5
        self.target_size = 2
        self.window_size = 1

        # learn
        self.lr = 2e-3
        self.weight_decay = 5e-3
        self.batch_size = 1000
        self.n_epochs = 10000