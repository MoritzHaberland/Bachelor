import torch 

def tester(test_loader, model):
    with torch.no_grad():
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                n_correct = 0
                n_samples = len(test_loader.dataset)

                for images, labels in test_loader:
                    images = images.reshape(-1, 28*28).to(device)
                    labels = labels.to(device)

                    outputs = model(images)

                    # max returns (output_value ,index)
                    _, predicted = torch.max(outputs, 1)
                    n_correct += (predicted == labels).sum().item()

                acc = n_correct / n_samples
                print(f'Accuracy of the network on the {n_samples} test images: {100*acc} %')