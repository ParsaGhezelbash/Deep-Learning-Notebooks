import numpy as np
import matplotlib.pyplot as plt

import torch

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def accuracy(y, output):
    y_index = torch.argmax(y, dim=1)

    output_index = torch.argmax(output, dim=1)

    return (y_index == output_index).float().mean().item()


def train(
    model, train_loader, val_loader, optimizer, criterion, device, epochs=30, log=True
):
    train_losses = []

    train_accuracies = []

    validation_losses = []

    validation_accuracies = []

    for epoch in range(epochs):
        batch_loss = 0

        train_acc = 0

        val_acc = 0

        model.train()

        for X_train, y_train in train_loader:
            X_train, y_train = X_train.to(device), y_train.to(device)
            # print(y_train.shape)

            y_pred = model(X_train)
            # print(y_pred.shape)
            # print(y_pred.requires_grad)  # Should be True

            loss = criterion(y_pred, y_train)
            # print(loss.requires_grad)  # Should be True
            # for name, param in model.named_parameters():
            #     print(f"{name}: {param.requires_grad}")  # Should all be True

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            batch_loss += loss.item() * y_train.shape[0]

            train_acc += accuracy(y_train, y_pred)

        train_losses.append(batch_loss / len(train_loader))

        train_accuracies.append(train_acc / len(train_loader))

        model.eval()

        with torch.no_grad():
            batch_loss = 0

            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)

                y_pred = model(X_val)

                loss = criterion(y_pred, y_val)

                batch_loss += loss.item() * y_val.shape[0]

                val_acc += accuracy(y_val, y_pred)

        validation_losses.append(batch_loss / len(val_loader))

        validation_accuracies.append(val_acc / len(val_loader))

        if log and (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs}, Train loss: {np.round(train_losses[-1], 3)}, Train acc: {np.round(train_accuracies[-1], 3)}, Val loss: {np.round(validation_losses[-1], 3)}, Val acc: {np.round(validation_accuracies[-1], 3)}"
            )

    return train_losses, train_accuracies, validation_losses, validation_accuracies


def test(model, test_loader, criterion, device):
    test_loss = 0

    test_accuracy = 0

    all_preds = []

    all_labels = []

    model.eval()

    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test, y_test = X_test.to(device), y_test.to(device)

            y_pred = model(X_test)

            loss = criterion(y_pred, y_test)

            test_loss += loss.item() * y_test.shape[0]

            test_accuracy += accuracy(y_test, y_pred)

            all_labels.extend(torch.argmax(y_test, dim=1).cpu().numpy())

            all_preds.extend(torch.argmax(y_pred, dim=1).cpu().numpy())

        test_loss /= len(test_loader)
        test_accuracy /= len(test_loader)

    return test_loss, test_accuracy, all_preds, all_labels


def plot_acc(
    train_accuracies, validation_accuracies, epochs, title="Accuracy vs epoch"
):
    plt.figure(figsize=[8, 4])

    plt.plot(
        range(1, epochs + 1),
        train_accuracies,
        c="blue",
        linestyle="--",
        marker="o",
        label="train accuracies",
    )

    plt.plot(
        range(1, epochs + 1),
        validation_accuracies,
        c="red",
        linestyle="--",
        marker="o",
        label="val accuracies",
    )

    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.tight_layout()
    plt.legend()
    plt.title(title)
    plt.show()


def plot_loss(train_losses, validation_losses, epochs, title="Loss vs epoch"):
    plt.figure(figsize=[8, 4])

    plt.plot(
        range(1, epochs + 1),
        train_losses,
        c="blue",
        linestyle="--",
        marker="o",
        label="train loss",
    )

    plt.plot(
        range(1, epochs + 1),
        validation_losses,
        c="red",
        linestyle="--",
        marker="o",
        label="val loss",
    )

    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.tight_layout()
    plt.legend()
    plt.title(title)
    plt.show()


def plot_confusion_matrix(all_labels, all_preds, class_names, title):
    cm = confusion_matrix(all_labels, all_preds)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.PuBu, xticks_rotation=45)

    plt.title(title)
    plt.show()


def save_model(model, path):
    torch.save(model.state_dict(), path)

    print(f"model saved to {path} successfully!")


def get_feature_maps(model, layer_name, x):
    features = []
    x = x.unsqueeze(0)

    def hook_fn(module, input, output):
        features.append(output)

    layer = dict(model.named_modules())[layer_name]
    hook = layer.register_forward_hook(hook_fn)

    _ = model(x)

    hook.remove()

    return features[0]


def visualize_feature_maps(feature_map, title, num_maps=8, max_row=5):
    feature_map = feature_map[0].detach().cpu().numpy()
    num_channels = feature_map.shape[0]

    indices = np.random.choice(num_channels, min(num_maps, num_channels), replace=False)

    r = max(1, np.ceil(num_maps / max_row).astype(int))
    c = min(max_row, num_maps)
    fig, axes = plt.subplots(r, c, figsize=[c * 2, r * 3])

    if r == 1 and c == 1:
        axes = np.array(axes).reshape(1, 1)
    elif r == 1 and c != 1:
        axes = axes.reshape(1, -1)
    elif r != 1 and c == 1:
        axes = axes.reshape(-1, 1)

    for i, idx in enumerate(indices):
        ax = axes[int(i / c), i % c]
        ax.imshow(feature_map[idx], cmap="cividis")
        ax.axis("off")
        ax.set_title(f"Channel {idx}")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_images(images, labels, title, max_row=5):
    r = max(1, np.ceil(len(images) / max_row).astype(int))
    c = min(max_row, len(images))
    fig, axes = plt.subplots(r, c, figsize=[r * 5, c * 5])

    if r == 1 and c == 1:
        axes = np.array(axes).reshape(1, 1)
    elif r == 1 and c != 1:
        axes = axes.reshape(1, -1)
    elif r != 1 and c == 1:
        axes = axes.reshape(-1, 1)

    for i in range(len(images)):
        sample = images[i] * 0.5 + 0.5
        sample = sample.numpy().transpose(1, 2, 0)
        ax = axes[int(i / c), i % c]
        ax.imshow(sample)
        ax.set_title(labels[i])
        ax.axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
