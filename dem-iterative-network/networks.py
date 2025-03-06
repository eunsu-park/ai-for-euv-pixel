import torch
import torch.nn as nn
import h5py


def init_weights(net, init_type="normal", init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
            if init_type == "normal":
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(f"Initialization method [{init_type}] is not implemented")
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)
    net.apply(init_func)


class Calculator(nn.Module):
    def __init__(self, num_euv_channels, num_temperature_bins, model_type):
        super(Calculator, self).__init__()
        self.num_euv_channels = num_euv_channels
        self.num_temperature_bins = num_temperature_bins
        self.model_type = model_type
        self.build()
        print(self)
        print("The number of parameters:", sum(p.numel() for p in self.parameters() if p.requires_grad))
    
    def build(self):
        if self.model_type == "pixel":
            kernel_size, stride, padding = 1, 1, 0
        elif self.model_type == "convolution":
            kernel_size, stride, padding = 3, 1, 1
        model = []
        model += [nn.Conv2d(self.num_euv_channels, 1024, kernel_size, stride, padding), nn.SiLU()]
        model += [nn.Conv2d(1024, self.num_temperature_bins, kernel_size, stride, padding)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Reconstructor(nn.Module):
    # def __init__(self, response_function, delta_temperature):
    #     super(Reconstructor, self).__init__()
    #     self.register_buffer("response_function", response_function)
    #     self.register_buffer("delta_temperature", delta_temperature)

    def __init__(self, factor):
        super(Reconstructor, self).__init__()
        self.register_buffer("factor", factor)

    def forward(self, x):
        x = x.view(x.size(0), x.size(2), x.size(3), x.size(1)) 
        x = torch.matmul(x, self.factor)
        x = x.view(x.size(0), x.size(3), x.size(1), x.size(2))
        return x

def define_networks(options):
    num_euv_channels = options.num_euv_channels
    num_temperature_bins = options.num_temperature_bins
    model_type = options.model_type
    device = options.device
    calculator = Calculator(num_euv_channels, num_temperature_bins, model_type).double().to(device)
    response_file_path = options.response_file_path
    with h5py.File(response_file_path, "r") as h5:
        factor = h5["factor_all_interpol"][:]
    factor = torch.tensor(factor).double()
    reconstructor = Reconstructor(factor).double().to(device)
    return calculator, reconstructor


if __name__ == "__main__" :

    torch.set_default_dtype(torch.float64)

    from options import TrainOptions
    options = TrainOptions().parse()

    C, R = define_networks(options)

    inp = torch.randn(options.batch_size,
                      options.num_euv_channels,
                      256, 256).to(options.device)
    out = C(inp)
    print(out.size())

    print(inp.dtype)
    print(C.model[0].weight.dtype)

    factor = R.factor
    print(f"factor size : {factor.size()}, factor dtype : {factor.dtype}, factor device : {factor.device}")

    rec = R(out)
    print(rec.size())
