import numpy as np
import torch
import time


def run_train(dataloader, model, loss_compute, step_size=10):
    """Standard Training and Logging Function"""
    start = time.time()
    total_loss = 0
    for i, batch in enumerate(dataloader):

        b_input, b_labels = batch

        out = model.forward(b_input.cuda(), b_labels.cuda(), None, None)
        dist = torch.sum((out[:, 0, :] - model.c) ** 2, dim=1)
        loss = loss_compute(out, b_labels.cuda(), dist)
        total_loss += loss

        if i % step_size == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d / %d Loss: %f" %
                  (i, len(dataloader), loss))
            start = time.time()
    return total_loss


def run_test(dataloader, model, loss_compute, step_size=10):
    """Standard Training and Logging Function"""
    preds = []
    distances = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            b_input, b_labels = batch

            out = model.forward(b_input.cuda(), b_labels.cuda(),
                                None, None)
            out_p = model.generator(out)
            dist = torch.sum((out[:, 0, :] - model.c) ** 2, dim=1)
            loss = loss_compute(out, b_labels.cuda(), dist)
            if i % step_size == 1:
                print("Epoch Step: %d / %d Loss: %f" %
                      (i, len(dataloader), loss))
            tmp = out_p.cpu().numpy()
            preds += list(np.argmax(tmp, axis=1))
            distances += list(dist.cpu().numpy())

    return preds, distances
