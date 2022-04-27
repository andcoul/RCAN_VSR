import torch

if __name__ =='__main__':

    
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        
    my_tensor = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.float32, device=device)

    print(my_tensor.cuda())
    print(torch.version.cuda)
    print(torch.cuda.is_available()) 