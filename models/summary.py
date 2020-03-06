

def number_of_parameters(model):
    return sum([layer.numel() for name,layer in model.named_parameters() if layer.requires_grad])

def model_summary(model):
  print("model_summary")
  print()
  print("Layer_name(Layer Type)             Number of Parameters")
  print("="*55)

  def print_module(module,offset):
      for name,child in module.named_children():
          n_params = number_of_parameters(child)
          if n_params == 0:
              continue
          name_line=" "*offset+"- %s (%s)"%(name,child.__class__.__name__)
          print("%-40s %10d"%(name_line,n_params))
          print_module(child,offset+2)
  print_module(model,0)      



if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.autograd import Variable
    from torch.nn.parameter import Parameter
    from crnn_lang import CNN
      
    model = CNN(32,1,40)
    model_summary(model)
    
