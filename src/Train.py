import torch
import torch.nn as nn
from sklearn.model_selection import KFold

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(source_data, target_data, model, epochs, batch_size, print_every, learning_rate):
    
    model.to(device)
    total_training_loss = 0
    total_valid_loss = 0
    loss = 0
    
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    # use cross validation
    kf = KFold(n_splits=epochs, shuffle=True)

    for e, (train_index, test_index) in enumerate(kf.split(source_data), 1):
        model.train()
        for i in range(0, len(train_index)):

            src = source_data[i]
            trg = target_data[i]

            output = model(src, trg, src.size(0), trg.size(0))

            current_loss = 0
            for (s, t) in zip(output["decoder_output"], trg): 
                current_loss += criterion(s, t)

            loss += current_loss
            total_training_loss += (current_loss.item() / trg.size(0)) # add the iteration loss

            if i % batch_size == 0 or i == (len(train_index)-1):
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss = 0


        # validation set 
        model.eval()
        for i in range(0, len(test_index)):
            src = source_data[i]
            trg = target_data[i]

            output = model(src, trg, src.size(0), trg.size(0))

            current_loss = 0
            for (s, t) in zip(output["decoder_output"], trg): 
                current_loss += criterion(s, t)

            total_valid_loss += (current_loss.item() / trg.size(0)) # add the iteration loss


        if e % print_every == 0:
            training_loss_average = total_training_loss / (len(train_index)*print_every)
            validation_loss_average = total_valid_loss / (len(test_index)*print_every)
            print("{}/{} Epoch  -  Training Loss = {:.4f}  -  Validation Loss = {:.4f}".format(e, epochs, training_loss_average, validation_loss_average))
            total_training_loss = 0
            total_valid_loss = 0 
