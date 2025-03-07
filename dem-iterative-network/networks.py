import torch
import torch.nn as nn
import h5py


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
        x = torch.clip(x+1., min=0, max=None) * 7.
        x = torch.pow(2., x) - 1.
        x = torch.matmul(x, self.factor)
        x = x.view(x.size(0), x.size(3), x.size(1), x.size(2))
        x = torch.clip(x+1., min=1., max=None)
        x = torch.log2(x)
        x = x / 7. - 1.
        return x


if __name__ == "__main__" :

    torch.set_default_dtype(torch.float64)

    from options import Options
    options = Options().parse()

    C = Calculator(options.num_euv_channels,
                   options.num_temperature_bins,
                   options.model_type).to(options.device).double()
    response_file_path = options.response_file_path
    with h5py.File(response_file_path, "r") as h5:
        factor = h5["factor_all_interpol"][:]
    factor = torch.tensor(factor).double()
    R = Reconstructor(factor).to(options.device).double()

    inp = torch.randn(options.batch_size,
                      options.num_euv_channels,
                      256, 256).to(options.device)
    inp = torch.clip(inp, min=-1., max=1.)
    out = C(inp)
    print(out.size())

    print(inp.dtype)
    print(C.model[0].weight.dtype)

    factor = R.factor
    print(f"factor size : {factor.size()}, factor dtype : {factor.dtype}, factor device : {factor.device}")

    rec = R(out)
    print(rec.size())
