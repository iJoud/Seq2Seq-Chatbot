from src.Data import prepare_text, toTensor

def evaluate(src, Q_vocab, A_vocab, model, target_max_len):
    
    try:
        src = toTensor(Q_vocab, " ".join(prepare_text(src)))
    except:
        print("Error: Word Encountered Not In The Vocabulary.")
        return
    
    answer_words = []
    
    output = model(src, None, src.size(0), target_max_len)

    for tensor in output['decoder_output']:

        _, top_token = tensor.data.topk(1)
        if top_token.item() == 1:
            break
        else:
            word = A_vocab.index2word[top_token.item()]
            answer_words.append(word)
            
    print("<", ' '.join(answer_words), "\n")
