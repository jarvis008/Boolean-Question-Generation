# Boolean-Question-Generation

## Introduction
This is the repository for the boolean question generator project, which is a model that takes a context and an answer (true/false - boolean in nature), and generates a question with the corresponding answer from the context. 

## Dataset Used
For the desired task, the boolQ dataset was used. A small sample of the dataset is shown below. These questions are naturally occurring ---they are generated in unprompted and unconstrained settings. Each example is a triplet of (question, passage, answer), with the title of the page as optional additional context. The text-pair classification setup is like existing natural language inference tasks.

![image](https://user-images.githubusercontent.com/77911251/226426003-636190ab-314c-4baa-b359-e612987dc8af.png)

The huggingface boolQ dataset had the following features:
Training samples: 9,427
Validation samples: 3,270
Total samples: 15,942


## Model Used
As the task at hand is a natural language generation task, with a text-to-text like pipeline, I decided to select the T5 transformer, due to its encoder-decoder nature, and easily available pretrained models. 

![image](https://user-images.githubusercontent.com/77911251/226426131-3a71faa2-6789-41d4-8ab5-c0f79aef3e0b.png)
 
Two versions of the given transformer, t5-base and t5-small were used. T5-small was selected for its better inference time.


## Preprocessing
First, the questions in the given dataset were converted to proper interrogative sentences, that is, starting with a capital letter and ending with a question mark. Then, for a given sample, the context and the answer were combined in the format:
answer: “TRUE/FALSE” context: “context given”
This combined string was passed through the T5 Tokenizer, with a maximum allowable length of 256, and shorter inputs padded to the maximum length. Meanwhile, the questions were also tokenized, with the same parameters. This process was repeated for each sample of the training dataset and the results were stored in a dictionary format, with the following keys – ‘input_ids’, ‘attention_mask’, ‘decoder_input_ids’, ‘decoder_attention_mask’. This dictionary was further converted to a Pytorch dataset.


## Finetuning
### T5-small:
For training the pretrained t5-small model, the following training parameters were used:
Epochs: 5
Batch size: 4
Optimizer: AdamW
Learning rate: 3e-4
The model was trained on google colab GPU (NVIDIA Tensor T4 Core), taking around one hour to complete training.
### T5-base:
For training the pretrained t5-base model, the following training parameters were used:
Epochs: 10
Batch size: 4
Optimizer: AdamW
Learning rate: 5e-5
### T5-base (pretrained for question generation):
Another t5-base model, which was pretrained for question generation was selected and fine-tuned for only Boolean question generation. For finetuning the pretrained t5-base model, the following training parameters were used:
Epochs: 4
Batch size: 4
Optimizer: AdamW
Learning rate: 5e-5

The base models were trained on Microsoft Planetary Computer Hub GPU (T4, 4 cores, 28 GiB Memory), taking around 1.2 hours and 40 minutes to complete training.


## Results and Observation
On the evaluation dataset, the batch size was kept 4, and the loss on the evaluation set, came out roughly as 0.06, for all 3 models. The models were saved and uploaded to huggingface_hub. Beam search was used for generating the questions, with number of beams adjustable between 3 and 10, and number of questions returned between 1 and 5. Upon passing individual context-answer pairs, it was observed that the models performed well for pairs with a ‘TRUE’ answer, generating coherent, and semantically correct questions. While for answers that were ‘FALSE,’ many times the question generated had a ‘TRUE’ answer. This could be because of possible data imbalances, that is, more samples with TRUE answer (5874), and lesser ones with FALSE answer (3553) in the training data. Also, a training dataset with only around 9,000 samples for training the t5 model would have been insufficient.


## Question Generator Interface
I used gradio for building a user interface, where user selects the type of model to use (t5-small, t5-base, t5-base pretrained for question generation), enters the context and answer, may enter the number of beams and the number of questions required, and gets the question/questions as output and the model inference time for calculating that output. Access the app by running the predict_app.ipynb notebook.

![image](https://user-images.githubusercontent.com/77911251/226426767-b88708ac-11ef-44f6-91c8-29a55634e3bd.png)

![image](https://user-images.githubusercontent.com/77911251/226426809-c88ea356-b01e-4c17-9d64-59efae6c9d8f.png)


## Future Work
For improving the model’s performance, some steps that can be taken are:
1.	Training 2 models separately for True questions and False questions. 
2.	Removing the class data imbalance by generating synthetic data for false answer class (by rule-based hard-coding or state-of-the-art question generators).
3.	Generating synthetic data (by rule-based hard-coding or state-of-the-art question generators) for increasing the dataset size.



