import torch
import numpy as np
import os
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from sklearn.metrics import average_precision_score, accuracy_score

from networks.attention_npr import attention_npr_resnet50


MODEL_PATH = 'best_attention_model.pth'
TEST_DIR = 'dataset/test'


def get_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def test_generator(model, gen_path, device):
    try:
        dataset = datasets.ImageFolder(gen_path, transform=get_transform())

        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

        y_true, y_pred = [], []

        model.eval()
        with torch.no_grad():
            for img, label in loader:
                img = img.to(device)
                output = model(img)
                probs = torch.sigmoid(output).flatten().tolist()
                y_pred.extend(probs)
                y_true.extend(label.tolist())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        acc = accuracy_score(y_true, y_pred > 0.5)

        if len(np.unique(y_true)) > 1:
            ap = average_precision_score(y_true, y_pred)
        else:
            ap = acc

        return acc, ap, None

    except Exception as e:
        return None, None, str(e)


def main():
    # Force CPU no CUDA
    device = torch.device('cpu')
    print(f"HIT: DEVICE -  {device}")

    # loading model
    model = attention_npr_resnet50(num_classes=1)
    state_dict = torch.load(MODEL_PATH, map_location='cpu')
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()


    generators = sorted([f for f in os.listdir(TEST_DIR)
                        if os.path.isdir(os.path.join(TEST_DIR, f))])


    print("HIT: TESTING GENERATORS")

    results = []

    for gen in generators:
        gen_path = os.path.join(TEST_DIR, gen)
        acc, ap, error = test_generator(model, gen_path, device)

        if error:
            print(f"{gen:<25} | ERROR: {error[:40]}")
        else:
            print(f"{gen:<25} | Acc: {acc*100:6.2f}% | AP: {ap*100:6.2f}%")
            results.append({'generator': gen, 'accuracy': acc, 'ap': ap})

    # print results to terminal
    if results:
        avg_acc = np.mean([r['accuracy'] for r in results])
        avg_ap = np.mean([r['ap'] for r in results])

        print(f"\nTested: {len(results)}/{len(generators)} generators")
        print(f"Average Accuracy: {avg_acc*100:.2f}%")
        print(f"Average AP:       {avg_ap*100:.2f}%")

        # sort by accuracy
        sorted_by_acc = sorted(results, key=lambda x: x['accuracy'], reverse=True)

        print("\nTop 5:")
        for r in sorted_by_acc[:5]:
            print(f"  {r['generator']:<25}: {r['accuracy']*100:.2f}%")

        print("\nBottom 5:")
        for r in sorted_by_acc[-5:]:
            print(f"  {r['generator']:<25}: {r['accuracy']*100:.2f}%")


if __name__ == '__main__':
    main()