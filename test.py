import torch
import torch.nn.functional as F

class Test():
  def __init__(self):
    self.test_losses=[]
    self.test_acc=[]

  def test_model(self,model, device , test_loader):
    model.eval()
    test_loss=0
    correct=0
    with torch.no_grad():
      for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output=model(data)
        test_loss += F.nll_loss(output, target, reduction='sum').item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    self.test_losses.append(test_loss)

    print("\n Test set: Avergae loss: {:4f}, Accuracy = {}/{}({:.2f}%)\n".format(test_loss, correct, len(test_loader.dataset), 100. *correct/len(test_loader.dataset)))
    self.test_acc.append(100. * correct/len(test_loader.dataset))
    return self.test_losses, self.test_acc