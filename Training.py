import pandas as pd
import os
import json
import typing
from tqdm import tqdm
import tensorflow as tf 
from custom_tokenizer import CustomTokenizer
from mltu.tensorflow.dataProvider import DataProvider
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from mltu.tensorflow.callbacks import Model2onnx, WarmupCosineDecay

from mltu.tensorflow.dataProvider import DataProvider
from mltu.tokenizers import CustomTokenizer

from mltu.tensorflow.transformer.utils import MaskedAccuracy, MaskedLoss
from mltu.tensorflow.transformer.callbacks import EncDecSplitCallback

from Tensorflow_Transformer import Transformer
from configs import ModelConfigs

with open(r"A:\Projects\SanskritConversion\Sanskrit_Text_Conversion\Data Files\TrainEnglish.txt", "r", encoding="utf-8") as f:
    en_training_data = f.read().split("\n")[:-1]

with open(r"A:\Projects\SanskritConversion\Sanskrit_Text_Conversion\Data Files\TestEnglish.txt", "r", encoding="utf-8") as f:
    en_validation_data = f.read().split("\n")[:-1]

with open(r"A:\Projects\SanskritConversion\Sanskrit_Text_Conversion\Data Files\TrainSanskrit.txt", "r", encoding="utf-8") as f:
    es_training_data = f.read().split("\n")[:-1]

with open(r"A:\Projects\SanskritConversion\Sanskrit_Text_Conversion\Data Files\TestSanskrit.txt", "r", encoding="utf-8") as f:
    es_validation_data = f.read().split("\n")[:-1]


max_lenght = 500
train_dataset = [[es_sentence, en_sentence] for es_sentence, en_sentence in zip(es_training_data, en_training_data) if len(es_sentence) <= max_lenght and len(en_sentence) <= max_lenght]
val_dataset = [[es_sentence, en_sentence] for es_sentence, en_sentence in zip(es_validation_data, en_validation_data) if len(es_sentence) <= max_lenght and len(en_sentence) <= max_lenght]
es_training_data, en_training_data = zip(*train_dataset)
es_validation_data, en_validation_data = zip(*val_dataset)

print(es_training_data[:3])

# prepare sanskrit tokenizer, this is the input language
tokenizer = CustomTokenizer(char_level=True)
tokenizer.fit_on_texts(es_training_data)
tokenizer.save(r"A:\Projects\SanskritConversion\Sanskrit_Text_Conversion\tokenizers\tokenizer.json")

# prepare english tokenizer, this is the output language
detokenizer = CustomTokenizer(char_level=True)
detokenizer.fit_on_texts(en_training_data)
detokenizer.save(r"A:\Projects\SanskritConversion\Sanskrit_Text_Conversion\tokenizers\detokenizer.json")

tokenized_sentence = detokenizer.texts_to_sequences(["Hello world, how are you?"])[0]
print(tokenized_sentence)

detokenized_sentence = detokenizer.detokenize([tokenized_sentence], remove_start_end=False)
print(detokenized_sentence)

detokenized_sentence = detokenizer.detokenize([tokenized_sentence])
print(detokenized_sentence)


def preprocess_inputs(data_batch, label_batch):
    encoder_input = np.zeros((len(data_batch), tokenizer.max_length)).astype(np.int64)
    decoder_input = np.zeros((len(label_batch), detokenizer.max_length)).astype(np.int64)
    decoder_output = np.zeros((len(label_batch), detokenizer.max_length)).astype(np.int64)

    data_batch_tokens = tokenizer.texts_to_sequences(data_batch)
    label_batch_tokens = detokenizer.texts_to_sequences(label_batch)

    for index, (data, label) in enumerate(zip(data_batch_tokens, label_batch_tokens)):
        encoder_input[index][:len(data)] = data
        decoder_input[index][:len(label)-1] = label[:-1] # Drop the [END] tokens
        decoder_output[index][:len(label)-1] = label[1:] # Drop the [START] tokens

    return (encoder_input, decoder_input), decoder_output

train_dataProvider = DataProvider(
    train_dataset, 
    batch_size=4, 
    batch_postprocessors=[preprocess_inputs],
    use_cache=True
    )

val_dataProvider = DataProvider(
    val_dataset, 
    batch_size=4, 
    batch_postprocessors=[preprocess_inputs],
    use_cache=True
    )
configs = ModelConfigs()
transformer = Transformer(
    num_layers=configs.num_layers,
    d_model=configs.d_model,
    num_heads=configs.num_heads,
    dff=configs.dff,
    input_vocab_size=len(tokenizer)+1,
    target_vocab_size=len(detokenizer)+1,
    dropout_rate=configs.dropout_rate,
    encoder_input_size=tokenizer.max_length,
    decoder_input_size=detokenizer.max_length
    )

transformer.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=configs.init_lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

# Compile the model
transformer.compile(
    loss=MaskedLoss(),
    optimizer=optimizer,
    metrics=[MaskedAccuracy()],
    run_eagerly=False
    )

warmupCosineDecay = WarmupCosineDecay(
    lr_after_warmup=configs.lr_after_warmup,
    final_lr=configs.final_lr,
    warmup_epochs=configs.warmup_epochs,
    decay_epochs=configs.decay_epochs,
    initial_lr=configs.init_lr,
    )
earlystopper = EarlyStopping(monitor="val_masked_accuracy", patience=5, verbose=1, mode="max")
checkpoint = ModelCheckpoint(f"{configs.model_path}/model.h5", monitor="val_masked_accuracy", verbose=1, save_best_only=True, mode="max", save_weights_only=False)
tb_callback = TensorBoard(f"{configs.model_path}/logs")
reduceLROnPlat = ReduceLROnPlateau(monitor="val_masked_accuracy", factor=0.9, min_delta=1e-10, patience=2, verbose=1, mode="max")
model2onnx = Model2onnx(f"{configs.model_path}/model.h5", metadata={"tokenizer": tokenizer.dict(), "detokenizer": detokenizer.dict()}, save_on_epoch_end=False)
encDecSplitCallback = EncDecSplitCallback(configs.model_path, encoder_metadata={"tokenizer": tokenizer.dict()}, decoder_metadata={"detokenizer": detokenizer.dict()})

transformer.fit(
    train_dataProvider, 
    validation_data=val_dataProvider, 
    epochs=configs.train_epochs,
    callbacks=[
        warmupCosineDecay,
        checkpoint, 
        tb_callback, 
        reduceLROnPlat,
        model2onnx,
        encDecSplitCallback
        ]
    )
